"""Tests for the parts the source distribution adds.

The prompt engine is covered by ``test_upstream_parity.py`` and must not change.
What is new here is everything around it: where an install lives now that there
is no frozen executable to anchor to, which artifacts a given GPU resolves to,
and the console setup that the one-click installer drives.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pytest

from prompt_master.core.models import CPU_INDEX, GpuInfo
from prompt_master.core.paths import DEFAULT_SUBDIR, ROOT_ENV, AppPaths
from prompt_master.inference import model_choice
from prompt_master.inference.device_detection import (CPU_DEVICE, CPU_RUNTIME, NO_OFFLOAD,
    PINNED, QUANTIZATIONS, SYSTEM_RAM_DEFAULT_QUANT, mixed_device, recommended_quantization,
    runtime_component_id, vram_shortfall_mb)
from prompt_master.provisioning import importer, installer, verifier
from prompt_master.provisioning.importer import LocalSource, SourceMismatch
from prompt_master.provisioning.manifest import Component
from prompt_master import setup_cli


def gpu(name="NVIDIA GeForce RTX 4090", total=24564, compute=None, index=0):
    return GpuInfo(index, f"GPU-{index}", name, total, total - 2000, "560.94", compute)


def cpu(name="Intel(R) Core(TM) i7-13700K", total=65413):
    """The processor, as ``detect_cpu`` reports it: no card, system RAM."""
    return GpuInfo(CPU_INDEX, "CPU", name, total, total - 20000, "AMD64", None)


# ── the two pinned cards must not move ───────────────────────────────────────

@pytest.mark.parametrize("name,runtime,quant", [
    ("NVIDIA GeForce RTX 3090", "llama-runtime-cuda12", "Q4_K_M"),
    ("NVIDIA GeForce RTX 5090", "llama-runtime-cuda13", "Q6_K_P"),
])
def test_pinned_cards_keep_their_upstream_provisioning(name, runtime, quant):
    """The 3090 and 5090 were the only supported cards upstream. Widening
    support must not change what either of them downloads."""
    card = gpu(name=name, total=24576 if "3090" in name else 32607)
    assert runtime_component_id(card) == runtime
    assert recommended_quantization(card) == quant


def test_pinned_mapping_wins_over_vram_heuristic():
    """A 3090 has enough raw VRAM to tempt the heuristic toward Q6_K_P once the
    thresholds are generic. The pin is what stops that."""
    assert PINNED["NVIDIA GeForce RTX 3090"][1] == "Q4_K_M"
    assert recommended_quantization(gpu(name="NVIDIA GeForce RTX 3090", total=24576)) == "Q4_K_M"


# ── other NVIDIA cards are now sized rather than refused ─────────────────────

@pytest.mark.parametrize("total,expected", [
    (81920, "Q8_K_P"),   # A100 80GB
    (49140, "Q8_K_P"),   # RTX 6000 Ada
    (32607, "Q6_K_P"),   # 32 GiB class
    (24564, "Q4_K_M"),   # 4090
    (16376, "Q4_K_M"),   # too small for a full offload, still offered
])
def test_quantization_scales_with_vram(total, expected):
    assert recommended_quantization(gpu(total=total)) == expected


def test_shortfall_is_zero_when_it_fits_and_positive_when_it_does_not():
    assert vram_shortfall_mb(gpu(total=24564), "Q4_K_M") == 0
    assert vram_shortfall_mb(gpu(total=16376), "Q4_K_M") == 22 * 1024 - 16376
    assert vram_shortfall_mb(gpu(total=24564), "Q8_K_P") > 0


def test_unknown_quantization_is_rejected():
    with pytest.raises(ValueError):
        vram_shortfall_mb(gpu(), "Q2_K")


@pytest.mark.parametrize("compute,expected", [
    (8.6, "llama-runtime-cuda12"),   # Ampere
    (8.9, "llama-runtime-cuda12"),   # Ada
    (9.0, "llama-runtime-cuda12"),   # Hopper
    (12.0, "llama-runtime-cuda13"),  # Blackwell
])
def test_runtime_follows_compute_capability(compute, expected):
    assert runtime_component_id(gpu(name="NVIDIA Whatever", compute=compute)) == expected


def test_runtime_falls_back_to_model_number_without_compute_capability():
    """Drivers too old to report compute_cap still have to land on a runtime
    that can target the card."""
    assert runtime_component_id(gpu(name="NVIDIA GeForce RTX 5080", compute=None)) == "llama-runtime-cuda13"
    assert runtime_component_id(gpu(name="NVIDIA GeForce RTX 4080", compute=None)) == "llama-runtime-cuda12"


# ── the manifest still resolves for every card and quantization ──────────────

@pytest.mark.parametrize("quant", QUANTIZATIONS)
@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 5090", "NVIDIA L40S"])
def test_every_supported_combination_resolves_to_four_pinned_components(name, quant):
    components = installer.resolve(gpu(name=name, compute=8.9), quant)
    assert len(components) == 4
    for component in components:
        component.validate()          # HTTPS, 64-char SHA-256, no "latest"
    known, exact = installer.download_estimate(gpu(name=name, compute=8.9), quant)
    # The llama.cpp archives publish no size, so the total is a floor, but the
    # multi-gigabyte weights are always counted and a number is always shown.
    assert known > 15 * 2 ** 30 and exact is False
    assert installer.format_download_size(gpu(name=name, compute=8.9), quant).startswith("at least")


def test_component_ids_pair_the_runtime_with_its_cuda_dlls():
    """The program archive and the CUDA runtime DLLs are separate upstream
    releases; a mismatched pair produces a runtime that cannot start."""
    ids = installer.component_ids(gpu(name="NVIDIA GeForce RTX 3090"), "Q4_K_M")
    assert ids == ("llama-runtime-cuda12", "llama-runtime-cuda12-cudart", "model-Q4_K_M", "mmproj")


# ── the processor, for a machine with no card to give the model ──────────────

def test_the_cpu_is_a_device_like_any_other():
    assert cpu().is_cpu and not gpu().is_cpu
    assert runtime_component_id(cpu()) == CPU_RUNTIME


def test_the_cpu_runtime_needs_no_cuda_dlls_beside_it():
    """The CUDA program archive is pinned separately from its cudart release
    and useless without it. The CPU archive carries everything it needs, so a
    third component would be a download with nothing to pair to."""
    ids = installer.component_ids(cpu(), "Q4_K_M")
    assert ids == (CPU_RUNTIME, "model-Q4_K_M", "mmproj")
    assert not any(key.endswith("-cudart") for key in ids)


@pytest.mark.parametrize("quant", QUANTIZATIONS)
def test_every_cpu_combination_resolves_to_pinned_components(quant):
    components = installer.resolve(cpu(), quant)
    assert len(components) == 3
    for component in components:
        component.validate()          # HTTPS, 64-char SHA-256, no "latest"
    assert installer.format_download_size(cpu(), quant).startswith("at least")


def test_the_cpu_is_never_short_of_memory_at_any_quantization():
    """A shortfall measures what would spill out of VRAM into system RAM. In
    CPU mode the weights are in system RAM already, so there is nothing to
    measure and nothing for either front end to warn about — including on a
    machine with far less RAM than the weights are large."""
    assert all(vram_shortfall_mb(cpu(), quant) == 0 for quant in QUANTIZATIONS)
    assert vram_shortfall_mb(cpu(total=8192), "Q8_K_P") == 0
    with pytest.raises(ValueError):
        vram_shortfall_mb(cpu(), "Q2_K")      # an unknown name is still an error


def test_the_cpu_default_is_the_lightest_build_not_the_largest_that_fits():
    """A card takes the biggest quantization its VRAM holds. Every byte of the
    weights crosses the memory bus on a processor, so the default there is the
    smallest pinned build — and it does not move with how much RAM is fitted."""
    assert recommended_quantization(cpu(total=65413)) == SYSTEM_RAM_DEFAULT_QUANT
    assert recommended_quantization(cpu(total=262144)) == SYSTEM_RAM_DEFAULT_QUANT
    assert recommended_quantization(gpu(total=262144)) == "Q8_K_P"


def test_cpu_state_records_no_offload_and_asks_llama_nothing(tmp_path, monkeypatch):
    """--device none is the whole answer, so the CUDA device probe must not run
    — there is no CUDA device for it to find — and a layer count passed in from
    the command line must not survive into the state as a fiction."""
    monkeypatch.setattr(installer, "list_llama_devices",
                        lambda *_a, **_k: pytest.fail("must not probe for a CUDA device"))
    installed = installer.Installed("runtime/llama-server.exe", "models/m.gguf", "models/p.gguf")

    state = installer.write_state(AppPaths(tmp_path), cpu(), "Q4_K_M", installed,
                                  gpu_layers=installer.FULL_OFFLOAD)

    assert state["gpu_device"] == CPU_DEVICE == "none"
    assert state["gpu_layers"] == NO_OFFLOAD == "0"
    assert state["gpu_index"] == CPU_INDEX
    assert state["gpu_device_name"] == cpu().name
    assert json.loads((tmp_path / "data" / "setup-state.json").read_text(encoding="utf-8")) == state


def test_a_cpu_install_hides_every_card_from_llama_server(tmp_path, monkeypatch):
    """CUDA_VISIBLE_DEVICES is set from the GPU index for a card. In CPU mode
    the index is -1, which as a value would hide nothing meaningful, so the
    variable is emptied instead: a card in the machine must not be picked up."""
    from prompt_master.inference.llama_process import LlamaProcess

    captured = {}

    class FakePopen:
        def __init__(self, command, env=None, **_kwargs):
            captured["command"], captured["env"] = command, env

        def poll(self): return None

    monkeypatch.setattr("prompt_master.inference.llama_process.subprocess.Popen", FakePopen)
    LlamaProcess().start(tmp_path / "llama-server.exe", tmp_path / "m.gguf", tmp_path / "p.gguf",
                         CPU_INDEX, CPU_DEVICE, 16384, tmp_path / "log.txt",
                         gpu_layers=NO_OFFLOAD)

    command = captured["command"]
    assert command[command.index("--device") + 1] == "none"
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_a_cpu_install_is_given_longer_to_load_before_it_is_called_dead(tmp_path, monkeypatch):
    """17-27 GiB into system RAM takes longer than filling VRAM does. The wait
    is a start-up allowance, not a warning and not a limit on generation."""
    from prompt_master.inference import service as service_module

    waited = []
    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    for relative in ("runtime/llama-server.exe", "models/m.gguf", "models/p.gguf"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stand-in")
    state = {"runtime": "runtime/llama-server.exe", "model": "models/m.gguf",
             "mmproj": "models/p.gguf", "gpu_index": CPU_INDEX, "gpu_device": CPU_DEVICE,
             "context_size": 16384, "gpu_layers": NO_OFFLOAD}
    paths.state_file.write_text(json.dumps(state), encoding="utf-8")

    inference = service_module.InferenceService(paths)
    monkeypatch.setattr(inference.process, "start", lambda *_a, **_k: None)
    monkeypatch.setattr(inference.process, "wait_ready", lambda timeout: waited.append(timeout))
    inference.client()
    assert waited == [service_module.CPU_READY_TIMEOUT]

    paths.state_file.write_text(json.dumps({**state, "gpu_index": 0, "gpu_device": "CUDA0",
                                            "gpu_layers": installer.FULL_OFFLOAD}), encoding="utf-8")
    inference.client()
    assert waited[-1] == service_module.GPU_READY_TIMEOUT

    # Mixed mode is a CUDA device with nothing resident on it, and loads out of
    # system RAM exactly as the CPU install above does.
    paths.state_file.write_text(json.dumps({**state, "gpu_index": 0, "gpu_device": "CUDA0"}),
                                encoding="utf-8")
    inference.client()
    assert waited[-1] == service_module.CPU_READY_TIMEOUT


# ── mixed: the card does the work, system RAM holds the model ────────────────

def mixed(name="NVIDIA GeForce RTX 4090", total=24564, index=0):
    return mixed_device(gpu(name=name, total=total, index=index))


def test_mixed_is_the_same_card_asked_to_do_something_else():
    card = gpu()
    assert not card.is_mixed and mixed_device(card).is_mixed
    # Same hardware: everything nvidia-smi reported survives the choice.
    assert dataclasses.replace(mixed_device(card), mixed=False) == card
    assert card.mode == "gpu" and mixed_device(card).mode == "mixed" and cpu().mode == "cpu"


def test_mixed_needs_a_card_to_hand_work_to():
    with pytest.raises(ValueError, match="CUDA GPU"):
        mixed_device(cpu())


def test_mixed_installs_the_cards_own_cuda_runtime_not_the_cpu_one():
    """The card is still doing the compute, so the CUDA build is still what has
    to be downloaded — cudart and all. Only where the weights sit changes."""
    assert runtime_component_id(mixed(name="NVIDIA GeForce RTX 3090")) == "llama-runtime-cuda12"
    assert runtime_component_id(mixed(name="NVIDIA GeForce RTX 5090")) == "llama-runtime-cuda13"
    ids = installer.component_ids(mixed(name="NVIDIA GeForce RTX 3090"), "Q4_K_M")
    assert ids == ("llama-runtime-cuda12", "llama-runtime-cuda12-cudart", "model-Q4_K_M", "mmproj")
    assert len(installer.resolve(mixed(), "Q4_K_M")) == 4


def test_mixed_is_sized_like_system_ram_not_like_a_card():
    """A 6 GiB card cannot hold any of the three, which in GPU mode is a warning
    on every option. In mixed mode it is not a fact about anything: the weights
    are not going there."""
    small = mixed(total=6144)
    assert small.weights_in_system_ram and cpu().weights_in_system_ram
    assert not gpu().weights_in_system_ram
    assert all(vram_shortfall_mb(small, quant) == 0 for quant in QUANTIZATIONS)
    assert vram_shortfall_mb(gpu(total=6144), "Q4_K_M") > 0        # the same card, held to VRAM
    assert recommended_quantization(small) == SYSTEM_RAM_DEFAULT_QUANT
    # Even a pinned card, which in GPU mode takes its pinned quantization.
    assert recommended_quantization(mixed(name="NVIDIA GeForce RTX 5090")) == SYSTEM_RAM_DEFAULT_QUANT
    assert recommended_quantization(gpu(name="NVIDIA GeForce RTX 5090", total=32607)) == "Q6_K_P"


def test_mixed_state_keeps_the_cuda_device_and_offloads_no_layers(tmp_path, monkeypatch):
    """The CUDA device probe still runs — there is a real device to name, and
    llama-server is still pointed at it. What changes is that nothing is
    resident on it, so a layer count from the command line is replaced."""
    monkeypatch.setattr(installer, "list_llama_devices", lambda *_a, **_k: ("CUDA0", "RTX 4090"))
    installed = installer.Installed("runtime/llama-server.exe", "models/m.gguf", "models/p.gguf")

    state = installer.write_state(AppPaths(tmp_path), mixed(index=1), "Q4_K_M", installed,
                                  gpu_layers=installer.FULL_OFFLOAD)

    assert state["mode"] == "mixed"
    assert state["gpu_device"] == "CUDA0"
    assert state["gpu_layers"] == NO_OFFLOAD == "0"
    assert state["gpu_index"] == 1


def test_gpu_mode_still_records_what_it_always_did(tmp_path, monkeypatch):
    """The mode is new state beside the old, not a change to it: a plain card
    keeps its device, its index and the caller's layer count."""
    monkeypatch.setattr(installer, "list_llama_devices", lambda *_a, **_k: ("CUDA0", "RTX 4090"))
    installed = installer.Installed("runtime/llama-server.exe", "models/m.gguf", "models/p.gguf")

    state = installer.write_state(AppPaths(tmp_path), gpu(index=1), "Q4_K_M", installed)

    assert state["mode"] == "gpu"
    assert state["gpu_device"] == "CUDA0" and state["gpu_index"] == 1
    assert state["gpu_layers"] == installer.FULL_OFFLOAD == "all"


def test_a_mixed_install_points_llama_server_at_the_card_with_nothing_on_it(tmp_path, monkeypatch):
    """The command is the whole mechanism: the card is named on --device, so
    llama.cpp keeps handing it the work it can take, and --n-gpu-layers 0 keeps
    the weights in system RAM. Unlike CPU mode, the card stays visible."""
    from prompt_master.inference.llama_process import LlamaProcess

    captured = {}

    class FakePopen:
        def __init__(self, command, env=None, **_kwargs):
            captured["command"], captured["env"] = command, env

        def poll(self): return None

    monkeypatch.setattr("prompt_master.inference.llama_process.subprocess.Popen", FakePopen)
    LlamaProcess().start(tmp_path / "llama-server.exe", tmp_path / "m.gguf", tmp_path / "p.gguf",
                         1, "CUDA0", 16384, tmp_path / "log.txt", gpu_layers=NO_OFFLOAD)

    command = captured["command"]
    assert command[command.index("--device") + 1] == "CUDA0"
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_every_card_is_offered_both_ways_with_its_own_entry_first(monkeypatch):
    """The order is what a machine that never wanted this depends on: option A
    is the first card holding the model, exactly as before."""
    from prompt_master.inference import device_detection

    monkeypatch.setattr(device_detection, "detect_gpus", lambda _timeout=15: [gpu(index=0), gpu(index=1)])
    monkeypatch.setattr(device_detection, "detect_cpu", cpu)

    offered = device_detection.detect_devices()
    assert [(device.physical_index, device.mode) for device in offered] == [
        (0, "gpu"), (0, "mixed"), (1, "gpu"), (1, "mixed"), (CPU_INDEX, "cpu")]


def test_missing_component_set_fails_before_any_download():
    with pytest.raises(RuntimeError, match="no complete"):
        installer.resolve(gpu(), "Q3_K_S")


def test_find_server_locates_the_executable_at_any_depth(tmp_path):
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    (nested / "llama-server.exe").write_bytes(b"exe")
    assert installer._find_server(tmp_path).name == "llama-server.exe"


def test_find_server_reports_a_runtime_without_one(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing useful", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no llama-server"):
        installer._find_server(tmp_path)


def test_fetch_rejects_a_zip_that_escapes_the_runtime_directory(tmp_path, monkeypatch):
    """extract_zips_atomic is what guards this, and fetch must keep using it."""
    from prompt_master.provisioning.extractor import extract_zips_atomic

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../../escaped.dll", b"bad")
    with pytest.raises(ValueError):
        extract_zips_atomic([archive], tmp_path / "runtime")


# ── install root discovery ───────────────────────────────────────────────────

def test_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV, str(tmp_path / "elsewhere"))
    assert AppPaths.discover().root == (tmp_path / "elsewhere").resolve()


def test_marker_points_at_another_drive(tmp_path, monkeypatch):
    """The whole reason the marker sits beside app.py: the models usually are
    not beside app.py."""
    monkeypatch.delenv(ROOT_ENV, raising=False)
    checkout, models = tmp_path / "checkout", tmp_path / "models"
    checkout.mkdir()
    monkeypatch.setattr("prompt_master.core.paths.application_dir", lambda: checkout)
    (checkout / "install.json").write_text(json.dumps({"install_root": str(models)}), encoding="utf-8")
    assert AppPaths.discover().root == models.resolve()


def test_corrupt_marker_falls_back_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.delenv(ROOT_ENV, raising=False)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr("prompt_master.core.paths.application_dir", lambda: checkout)
    (checkout / "install.json").write_text("{ not json", encoding="utf-8")
    assert AppPaths.discover().root == (checkout / DEFAULT_SUBDIR).resolve()


def test_default_root_is_under_the_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv(ROOT_ENV, raising=False)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr("prompt_master.core.paths.application_dir", lambda: checkout)
    assert AppPaths.discover().root == (checkout / DEFAULT_SUBDIR).resolve()


def test_configured_is_false_until_setup_writes_state(tmp_path):
    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    assert not paths.configured
    paths.state_file.write_text("{}", encoding="utf-8")
    assert paths.configured


def test_contained_still_refuses_escapes(tmp_path):
    with pytest.raises(ValueError):
        AppPaths(tmp_path).contained("../outside")


def test_locate_contains_a_relative_path_and_takes_an_absolute_one(tmp_path):
    """The weights may live outside the install root — see AppPaths.locate —
    and a relative path is still held to it exactly as it always was."""
    paths = AppPaths(tmp_path)
    outside = tmp_path.parent / "elsewhere" / "model.gguf"

    assert paths.locate("models/model.gguf") == (tmp_path / "models" / "model.gguf").resolve()
    assert paths.locate(outside) == outside.resolve()
    with pytest.raises(ValueError):
        paths.locate("../escape.gguf")


def test_record_keeps_our_own_files_relative_and_leaves_the_rest_where_they_are(tmp_path):
    """The install stays movable: everything under the root is recorded
    relative to it, and only a file somewhere else is pinned to a drive."""
    paths = AppPaths(tmp_path)
    outside = tmp_path.parent / "elsewhere" / "model.gguf"

    assert paths.record(tmp_path / "models" / "model.gguf") == "models/model.gguf"
    assert paths.record(outside) == str(outside.resolve())


# ── console setup ────────────────────────────────────────────────────────────

def test_setup_flags_cover_every_question():
    """An unattended reinstall must not need a human, or the installer cannot
    be scripted."""
    options = setup_cli.parse_args(["--dir", "D:/PM", "--gpu", "1", "--quant", "Q8_K_P", "--yes"])
    assert (options.directory, options.gpu, options.quant, options.yes) == ("D:/PM", 1, "Q8_K_P", True)
    assert options.context_size == installer.DEFAULT_CONTEXT_SIZE
    assert options.gpu_layers == installer.FULL_OFFLOAD
    assert options.cpu is False

    cpu_run = setup_cli.parse_args(["--dir", "D:/PM", "--cpu", "--quant", "Q4_K_M", "--yes"])
    assert cpu_run.cpu and cpu_run.gpu is None and cpu_run.mixed is False

    mixed_run = setup_cli.parse_args(["--dir", "D:/PM", "--gpu", "0", "--mixed", "--yes"])
    assert mixed_run.mixed and mixed_run.gpu == 0


def test_preselected_gpu_index_must_exist(monkeypatch):
    monkeypatch.setattr(setup_cli, "detect_devices", lambda: [gpu(index=0), cpu()])
    assert setup_cli.ask_device(0).physical_index == 0
    with pytest.raises(SystemExit, match="No GPU with index 3"):
        setup_cli.ask_device(3)


def test_a_missing_index_points_at_the_cpu_rather_than_stopping_there(monkeypatch):
    """--gpu 3 on a machine without one is now a wrong answer to a question
    that has another answer, so the message has to name it."""
    monkeypatch.setattr(setup_cli, "detect_devices", lambda: [cpu()])
    with pytest.raises(SystemExit, match="--cpu"):
        setup_cli.ask_device(0)


def test_preselected_quantization_must_be_known(monkeypatch):
    assert setup_cli.ask_quantization(gpu(), "Q8_K_P") == "Q8_K_P"
    with pytest.raises(SystemExit, match="Unknown quantization"):
        setup_cli.ask_quantization(gpu(), "Q2_K")


def test_choose_accepts_letters_and_numbers(monkeypatch):
    answers = iter(["b", "1", "?", "A"])
    monkeypatch.setattr(setup_cli, "ask", lambda *_args, **_kwargs: next(answers))
    assert setup_cli.choose("pick", ["one", "two"]) == 1   # "b"
    assert setup_cli.choose("pick", ["one", "two"]) == 1   # "1"
    assert setup_cli.choose("pick", ["one", "two"]) == 0   # "?" retried, then "A"


def test_a_single_gpu_is_still_the_default_answer(monkeypatch, capsys):
    """One card no longer means one option — it is offered twice and the
    processor beside it — but the card holding the model stays first, so
    pressing Enter installs what it always did."""
    monkeypatch.setattr(setup_cli, "detect_devices",
                        lambda: [gpu(index=2), mixed(index=2), cpu()])
    # What pressing Enter does: ask() returns the default it was offered.
    monkeypatch.setattr(setup_cli, "ask", lambda _prompt, default="": default)

    device = setup_cli.ask_device(None)

    assert device.physical_index == 2 and device.mode == "gpu"
    listed = capsys.readouterr().out
    assert "A) NVIDIA GeForce RTX 4090 — 24564 MiB" in listed and "(recommended)" in listed
    assert "B) NVIDIA GeForce RTX 4090 — mixed" in listed
    assert "C) Intel(R) Core(TM) i7-13700K" in listed


def test_the_mixed_flag_applies_to_the_named_card_or_the_first_one(monkeypatch, capsys):
    monkeypatch.setattr(setup_cli, "detect_devices",
                        lambda: [gpu(index=0), mixed(index=0), gpu(index=1), mixed(index=1), cpu()])

    assert setup_cli.ask_device(1, mixed=True).physical_index == 1
    assert setup_cli.ask_device(1, mixed=True).is_mixed
    # Without --gpu it is still an answer rather than a question: the first card.
    monkeypatch.setattr(setup_cli, "ask", lambda *_a, **_k: pytest.fail("should not ask"))
    chosen = setup_cli.ask_device(None, mixed=True)
    assert chosen.physical_index == 0 and chosen.is_mixed
    assert "mixed mode" in capsys.readouterr().out


def test_mixed_needs_a_card_and_says_so(monkeypatch):
    monkeypatch.setattr(setup_cli, "detect_devices", lambda: [cpu()])
    with pytest.raises(SystemExit, match="--cpu to run without a card"):
        setup_cli.ask_device(None, mixed=True)


def test_mixed_and_cpu_together_are_refused():
    with pytest.raises(SystemExit, match="--mixed hands work to a GPU"):
        setup_cli.run(["--cpu", "--mixed"])


def test_no_gpu_selects_the_cpu_instead_of_exiting(monkeypatch, capsys):
    """The machine this feature exists for: no card, no driver, one answer."""
    monkeypatch.setattr(setup_cli, "detect_devices", lambda: [cpu()])
    monkeypatch.setattr(setup_cli, "ask", lambda *_a, **_k: pytest.fail("should not ask"))

    device = setup_cli.ask_device(None)

    assert device.is_cpu
    assert "No CUDA GPU" in capsys.readouterr().out


def test_the_cpu_flag_skips_the_scan_entirely(monkeypatch):
    """--cpu is an answer, not a preference: a machine with a card that should
    stay free must not need nvidia-smi to say so."""
    monkeypatch.setattr(setup_cli, "detect_devices", lambda: pytest.fail("must not scan"))
    monkeypatch.setattr(setup_cli, "detect_cpu", cpu)
    assert setup_cli.ask_device(None, cpu=True).is_cpu


def test_cpu_and_gpu_together_are_refused():
    with pytest.raises(SystemExit, match="--cpu and --gpu"):
        setup_cli.run(["--cpu", "--gpu", "0"])


# ── supplying the model from disk instead of downloading it ──────────────────
#
# The model is one 16-27 GiB file, and a connection that cannot carry it is not
# something setup can fix. Anyone who already has it hands it over instead.

def pin_to_file(monkeypatch, path, key="model-Q6_K_P"):
    """Rewrite the manifest so ``key`` pins the bytes of ``path`` — a test can
    then supply a file that really is the pinned artifact without writing 21 GiB."""
    components = dict(installer.load_components())
    body = path.read_bytes()
    components[key] = dataclasses.replace(components[key], size=len(body),
                                          sha256=hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(installer, "load_components", lambda: components)
    return components


def pinned(path, component_id="model-Q6_K_P", destination="models/model.gguf"):
    """A manifest entry for a file that exists, so a supplied copy can match it."""
    body = path.read_bytes()
    return Component(component_id, "https://example.invalid/model.gguf", destination,
                     len(body), hashlib.sha256(body).hexdigest(), "pinned")


def their_file(tmp_path, name="Gemma4-Q6_K_P.gguf", body=b"pretend gguf" * 512):
    path = tmp_path / "downloads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_a_supplied_file_is_moved_not_copied(tmp_path):
    """The whole point of handing over a 21 GiB file already on the disk is not
    to end up with two of them."""
    source = their_file(tmp_path)
    destination = tmp_path / "user_data" / "models" / "model.gguf"
    installed = destination.with_name(source.name)

    assert importer.adopt(pinned(source), destination, LocalSource(source)) == installed
    assert installed.read_bytes() == b"pretend gguf" * 512
    assert not source.exists()


def test_a_supplied_file_keeps_the_name_it_arrived_with(tmp_path):
    """It goes into the folder the manifest names, not under the file name the
    manifest names. A models folder that renamed somebody's file to the pinned
    build's name would claim to hold something it does not."""
    source = their_file(tmp_path, name="MyMerge-Q5_K_M.gguf")
    destination = tmp_path / "user_data" / "models" / "Gemma4-26B-Q6_K_P.gguf"

    installed = importer.adopt(pinned(source), destination, LocalSource(source))

    assert installed.name == "MyMerge-Q5_K_M.gguf"
    assert installed.parent == destination.parent
    assert not destination.exists(), "nothing was written under the pinned name"


def test_keeping_the_source_copies_instead(tmp_path):
    source = their_file(tmp_path)
    destination = tmp_path / "user_data" / "models" / "model.gguf"

    installed = importer.adopt(pinned(source), destination, LocalSource(source, move=False))
    assert source.exists() and installed.read_bytes() == source.read_bytes()


def test_adopting_clears_the_abandoned_download_it_replaces(tmp_path):
    """The 175 MiB of a download that never finished is dead weight once the
    file arrives from disk — and a part file blocks a later resume anyway."""
    source = their_file(tmp_path)
    destination = tmp_path / "user_data" / "models" / "model.gguf"
    destination.parent.mkdir(parents=True)
    part = destination.with_name(destination.name + ".part")
    part.write_bytes(b"an interrupted download")

    importer.adopt(pinned(source), destination, LocalSource(source))
    assert not part.exists()


def test_a_file_already_at_the_destination_is_left_where_it_is(tmp_path):
    destination = tmp_path / "user_data" / "models" / "model.gguf"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"already installed")

    assert importer.adopt(pinned(destination), destination, LocalSource(destination)) == destination
    assert destination.read_bytes() == b"already installed"


def test_a_file_that_is_not_the_pinned_artifact_is_refused_and_left_alone(tmp_path):
    """Refusing before the move matters: their only copy must survive a no."""
    source = their_file(tmp_path)
    component = dataclasses.replace(pinned(source), sha256="0" * 64)
    destination = tmp_path / "user_data" / "models" / "model.gguf"

    with pytest.raises(SourceMismatch, match="pinned SHA-256"):
        importer.adopt(component, destination, LocalSource(source))
    assert source.exists() and not destination.exists()


def test_the_wrong_size_is_refused_without_reading_the_file(tmp_path, monkeypatch):
    """21 GiB takes minutes to hash. The two ordinary mistakes — the wrong
    quantization, a half-finished download — are visible from the size alone."""
    source = their_file(tmp_path)
    component = dataclasses.replace(pinned(source), size=22758955104)
    monkeypatch.setattr(verifier, "digest_of", lambda *_a, **_k: pytest.fail("must not hash"))

    problem = importer.size_problem(component, source)
    assert "6.00 KiB" in problem and "21.20 GiB" in problem


def test_a_cross_drive_move_copies_then_removes_the_original(tmp_path, monkeypatch):
    """os.replace cannot cross volumes, which is the ordinary case here: the
    model was downloaded to C: and the install root is on D:."""
    source = their_file(tmp_path)
    destination = tmp_path / "user_data" / "models" / "model.gguf"
    installed = destination.with_name(source.name)
    monkeypatch.setattr(importer.os, "replace", _refusing_replace(installed))
    seen = []

    importer.adopt(pinned(source), destination, LocalSource(source), lambda done, total: seen.append(done))
    assert installed.read_bytes() == b"pretend gguf" * 512
    assert not source.exists()
    assert seen and seen[-1] == installed.stat().st_size   # the copy reported progress


def _refusing_replace(destination):
    """os.replace that fails for the destination itself, as a cross-volume
    rename does, but still works for the .part the copy renames into place."""
    real = importer.os.replace

    def replace(source, target):
        if Path(target) == destination and Path(source).suffix != ".part":
            raise OSError(17, "cross-device link")
        return real(source, target)

    return replace


def test_a_cross_drive_move_checks_for_room_first(tmp_path, monkeypatch):
    source = their_file(tmp_path)
    destination = tmp_path / "user_data" / "models" / "model.gguf"
    destination.parent.mkdir(parents=True)
    monkeypatch.setattr(importer.os, "replace", _refusing_replace(destination.with_name(source.name)))
    monkeypatch.setattr(importer.shutil, "disk_usage", lambda _path: _Usage(64))

    with pytest.raises(OSError, match="free"):
        importer.adopt(pinned(source), destination, LocalSource(source))
    assert source.exists() and not destination.exists()


@dataclasses.dataclass
class _Usage:
    free: int


def test_the_manifest_can_name_what_a_file_actually_is():
    """"That is the Q4_K_M build" ends an investigation that "hash mismatch"
    only starts."""
    components = installer.load_components()
    assert installer.identify(components["model-Q4_K_M"].sha256) == "model-Q4_K_M"
    assert installer.identify(components["mmproj"].sha256.upper()) == "mmproj"
    assert installer.identify("0" * 64) is None


def test_a_supplied_component_drops_out_of_the_download_estimate():
    card = gpu(name="NVIDIA GeForce RTX 5090", total=32607)
    whole, _ = installer.download_estimate(card, "Q6_K_P")
    without_model, _ = installer.download_estimate(card, "Q6_K_P", {"model-Q6_K_P"})

    assert whole - without_model == installer.load_components()["model-Q6_K_P"].size
    assert installer.format_download_size(card, "Q6_K_P", {"model-Q6_K_P", "mmproj"}) == "the runtime archives only"


def test_fetch_installs_a_supplied_component_and_downloads_the_rest(tmp_path, monkeypatch):
    downloaded, adopted = [], []

    def fake_download(component, target, progress=None, notice=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target, "w") as bundle: bundle.writestr("llama-server.exe", b"exe")
        else:
            target.write_bytes(b"downloaded")
        downloaded.append(component.component_id); return target

    def fake_adopt(component, target, source, progress=None):
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"from disk")
        adopted.append((component.component_id, source.path.name)); return target

    monkeypatch.setattr(installer, "download", fake_download)
    monkeypatch.setattr(installer, "adopt", fake_adopt)
    card = gpu(name="NVIDIA GeForce RTX 5090", total=32607)

    installed = installer.fetch(AppPaths(tmp_path), card, "Q6_K_P",
                                sources={"model-Q6_K_P": LocalSource(tmp_path / "mine.gguf", checked=True)})

    assert adopted == [("model-Q6_K_P", "mine.gguf")]
    assert downloaded == ["llama-runtime-cuda13", "llama-runtime-cuda13-cudart", "mmproj"]
    assert (tmp_path / installed.model).read_bytes() == b"from disk"

    # A file for a component this install does not use is refused rather than
    # ignored — otherwise it would be quietly downloaded instead.
    with pytest.raises(RuntimeError, match="model-Q4_K_M"):
        installer.fetch(AppPaths(tmp_path), card, "Q6_K_P",
                        sources={"model-Q4_K_M": LocalSource(tmp_path / "mine.gguf", checked=True)})


# ── the question that offers it ──────────────────────────────────────────────

def answers(monkeypatch, *replies):
    queue = iter(replies)
    monkeypatch.setattr(setup_cli, "ask", lambda *_args, **_kwargs: next(queue))


def test_declining_the_question_downloads_exactly_as_before(monkeypatch):
    answers(monkeypatch, "N")
    assert setup_cli.ask_local_model("Q6_K_P", setup_cli.Steps(4)) == ("Q6_K_P", {})


def test_an_accepted_file_is_recorded_as_checked_and_moved(monkeypatch, tmp_path, capsys):
    source = their_file(tmp_path)
    monkeypatch.setattr(setup_cli, "accept_file", lambda component, path: (None, component.component_id))
    answers(monkeypatch, "Y", f'"{source}"')      # Windows "Copy as path" quotes it

    quant, sources = setup_cli.ask_local_model("Q6_K_P", setup_cli.Steps(4))

    assert quant == "Q6_K_P"
    supplied = sources["model-Q6_K_P"]
    assert supplied.path == source and supplied.move and supplied.checked
    assert "MOVED" in capsys.readouterr().out       # said before the file is taken


def test_a_file_that_is_a_different_quantization_offers_to_install_that_instead(monkeypatch, tmp_path):
    """Better to install what they have as what it is than as what was asked for."""
    source = their_file(tmp_path, name="Gemma4-Q4_K_M.gguf")
    monkeypatch.setattr(setup_cli, "accept_file",
                        lambda component, path: ("not Q6_K_P — it is model-Q4_K_M", "model-Q4_K_M"))
    answers(monkeypatch, "Y", str(source), "Y")

    quant, sources = setup_cli.ask_local_model("Q6_K_P", setup_cli.Steps(4))
    assert quant == "Q4_K_M" and set(sources) == {"model-Q4_K_M"}


def test_a_projector_beside_the_model_is_offered_too(monkeypatch, tmp_path):
    """Both files come from the same repository, so having one usually means
    having the other — and it is another download from the host that failed."""
    source = their_file(tmp_path)
    projector = source.parent / Path(installer.load_components()["mmproj"].destination).name
    projector.write_bytes(b"a small stand-in for the projector")
    pin_to_file(monkeypatch, projector, "mmproj")
    monkeypatch.setattr(setup_cli, "accept_file", lambda component, path: (None, component.component_id))
    answers(monkeypatch, "Y", str(source), "Y")

    _quant, sources = setup_cli.ask_local_model("Q6_K_P", setup_cli.Steps(4))
    assert sources["mmproj"].path == projector


def test_a_named_file_that_is_not_the_pinned_one_stops_an_unattended_run(monkeypatch, tmp_path):
    """--yes asks nothing, so there is no "use it anyway" to fall back on."""
    source = their_file(tmp_path)
    monkeypatch.setattr(setup_cli, "accept_file", lambda component, path: ("that is model-Q4_K_M", "model-Q4_K_M"))
    options = setup_cli.parse_args(["--model-file", str(source), "--yes"])

    with pytest.raises(SystemExit, match="model-Q4_K_M"):
        setup_cli.supplied_files(options, "Q6_K_P")


def test_the_wizard_vets_a_supplied_model_the_same_way(monkeypatch, tmp_path):
    """The wizard is what an existing install re-runs setup from, so it has to
    offer the same way out of a download that will not finish."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from prompt_master.ui.setup_wizard import SetupWizard

    source = their_file(tmp_path)
    pin_to_file(monkeypatch, source)
    widgets.QApplication.instance() or widgets.QApplication([])
    wizard = SetupWizard(AppPaths(tmp_path / "user_data"))
    wizard.quant.setCurrentText("Q6_K_P")

    assert wizard._vet_model_file() == {}                     # empty field still downloads
    wizard.model_file.setText(f'"{source}"')
    supplied = wizard._vet_model_file()["model-Q6_K_P"]
    assert supplied.path == source and supplied.move and supplied.checked

    monkeypatch.setattr(widgets.QMessageBox, "question",
                        staticmethod(lambda *_a, **_k: widgets.QMessageBox.No))
    wizard.model_file.setText(str(tmp_path / "not-the-pinned-one.gguf"))
    assert wizard._vet_model_file() is None                   # refused, and the page holds


def open_wizard(monkeypatch, tmp_path, devices):
    """The wizard on its hardware page, with the scan answered by ``devices``."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from prompt_master.ui import setup_wizard as wizard_module

    monkeypatch.setattr(wizard_module, "detect_devices", lambda: list(devices))
    widgets.QApplication.instance() or widgets.QApplication([])
    wizard = wizard_module.SetupWizard(AppPaths(tmp_path / "user_data"))
    wizard._page_changed(1)
    return wizard


def test_the_wizard_offers_every_mode_a_card_has(monkeypatch, tmp_path):
    """Settings → Models and Hardware is where an installed app goes back to
    change its mind, so all three have to be on that page too."""
    wizard = open_wizard(monkeypatch, tmp_path, [gpu(), mixed(), cpu()])

    assert [wizard.gpu.itemData(row).mode for row in range(wizard.gpu.count())] == [
        "gpu", "mixed", "cpu"]
    assert "mixed" in wizard.gpu.itemText(1)
    assert "Intel(R) Core(TM) i7-13700K" in wizard.gpu.itemText(2)
    assert wizard.gpu.currentIndex() == 0                  # the card is still the default


@pytest.mark.parametrize("row,expected", [(1, "system RAM and uses"), (2, "processor and system RAM")])
def test_the_wizard_describes_a_system_ram_install_without_warning_about_it(
        monkeypatch, tmp_path, row, expected):
    wizard = open_wizard(monkeypatch, tmp_path, [gpu(), mixed(), cpu()])

    wizard.gpu.setCurrentIndex(row)
    wizard._page_changed(2)

    assert wizard.quant.currentText() == SYSTEM_RAM_DEFAULT_QUANT
    described = wizard.recommendation.text()
    assert expected in described
    # A disclaimer, not a warning: nothing about memory size or speed.
    assert "Warning" not in described and "slow" not in described


def test_the_wizard_still_warns_a_card_that_cannot_hold_its_quantization(monkeypatch, tmp_path):
    """The VRAM warning is the reason mixed mode is worth offering, so it has to
    survive being sat next to it."""
    wizard = open_wizard(monkeypatch, tmp_path, [gpu(total=8192), mixed(total=8192), cpu()])

    wizard._page_changed(2)

    assert "Warning" in wizard.recommendation.text()


def test_the_wizard_falls_back_to_the_processor_when_the_scan_fails(monkeypatch, tmp_path):
    """A missing driver used to leave an empty combo box on a page that could
    not be left. It now leaves the one device that machine actually has."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from prompt_master.ui import setup_wizard as wizard_module

    def explode():
        raise RuntimeError("nvidia-smi is not available.")

    monkeypatch.setattr(wizard_module, "detect_devices", explode)
    monkeypatch.setattr(wizard_module, "detect_cpu", cpu)
    widgets.QApplication.instance() or widgets.QApplication([])
    wizard = wizard_module.SetupWizard(AppPaths(tmp_path / "user_data"))

    wizard._page_changed(1)
    assert wizard.gpu.count() == 1 and wizard.gpu.currentData().is_cpu
    assert "nvidia-smi" in wizard.hardware_status.text()


def test_named_files_can_be_kept_where_they_are(monkeypatch, tmp_path):
    source, projector = their_file(tmp_path), their_file(tmp_path, name="mmproj.gguf")
    monkeypatch.setattr(setup_cli, "accept_file", lambda component, path: (None, component.component_id))
    options = setup_cli.parse_args(["--model-file", str(source), "--mmproj-file", str(projector), "--keep-source"])

    sources = setup_cli.supplied_files(options, "Q6_K_P")
    assert set(sources) == {"model-Q6_K_P", "mmproj"}
    assert not any(source.move for source in sources.values())


# ── the one-click installer's choice of interpreter ──────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def one_click():
    """``one_click.py`` loaded as a module — it sits beside app.py, not in src/."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prompt_master_one_click", REPO_ROOT / "one_click.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandboxed(one_click, tmp_path, monkeypatch):
    """The installer with the directories it writes to redirected into tmp_path."""
    files = tmp_path / "installer_files"
    files.mkdir()
    monkeypatch.setattr(one_click, "INSTALLER_FILES", files)
    monkeypatch.setattr(one_click, "ENV_DIR", files / "env")
    monkeypatch.setattr(one_click, "CONDA_DIR", files / "conda")
    monkeypatch.setattr(one_click, "REQUIREMENTS_MARKER", files / "requirements.installed")
    return one_click


@pytest.mark.parametrize("version,ok", [
    ((3, 11), False),   # too old for the application
    ((3, 12), True),
    ((3, 13), True),
    ((3, 14), False),   # too new for the pinned wheels
    ((4, 0), False),
    (None, False),      # an interpreter that would not report a version
])
def test_supported_pythons_are_a_range_not_a_floor(one_click, version, ok):
    """PySide6 6.8.1 declares Requires-Python <3.14 and Pillow 11.0.0 and numpy
    2.2.1 publish no 3.14 wheels, so a newer Python is a reason to skip an
    interpreter rather than to prefer it."""
    assert one_click.supported(version) is ok


def test_pyproject_declares_the_same_range_the_installer_enforces(one_click):
    expected = 'requires-python = ">={0}.{1},<{2}.{3}"'.format(
        *one_click.MIN_PYTHON, *one_click.MAX_PYTHON)
    assert expected in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_find_system_python_skips_a_python_the_pins_do_not_cover(one_click, monkeypatch, tmp_path):
    """An interpreter that is too new fails at pip rather than at import, so it
    has to be rejected before an environment is built on it."""
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    monkeypatch.setattr(one_click.sys, "platform", "linux")
    monkeypatch.setattr(one_click.sys, "version_info", (3, 14, 0))
    monkeypatch.setattr(one_click.shutil, "which", lambda name: str(interpreter))

    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 14))
    assert one_click.find_system_python() is None

    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 13))
    assert one_click.find_system_python() == interpreter


def test_py_launcher_is_asked_only_for_versions_in_the_range(one_click, monkeypatch):
    """The launcher sees installs that are not on PATH, so what it is asked for
    decides which Python a machine with several of them ends up using."""
    probes = []

    def fake_run(command, **_kwargs):
        probes.append(command)
        raise OSError("no interpreter here")

    monkeypatch.setattr(one_click.sys, "platform", "win32")
    monkeypatch.setattr(one_click.shutil, "which",
                        lambda name: r"C:\Windows\py.exe" if name == "py" else None)
    monkeypatch.setattr(one_click.subprocess, "run", fake_run)

    assert one_click.find_system_python() is None
    assert [command[1] for command in probes if command[0] == r"C:\Windows\py.exe"] == ["-3.13", "-3.12"]


def test_the_private_python_is_reused_only_while_the_pins_cover_it(sandboxed, monkeypatch):
    """The bootstrap is pinned to a 3.12 build, so reusing whatever is in
    installer_files\\conda is safe only as long as it is still that build."""
    one_click = sandboxed
    private = one_click.CONDA_DIR / "python.exe"
    private.parent.mkdir(parents=True)
    private.write_text("bootstrap", encoding="utf-8")

    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 12))
    assert one_click.bootstrap_miniconda() == private

    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 14))
    monkeypatch.setattr(one_click.sys, "platform", "linux")
    with pytest.raises(SystemExit, match="3.12 or 3.13"):
        one_click.bootstrap_miniconda()
    assert private.exists()      # nothing is deleted where nothing can replace it


# ── the environment it builds ────────────────────────────────────────────────

def stale_env(one_click, layout="Scripts/python.exe"):
    interpreter = one_click.ENV_DIR.joinpath(*layout.split("/"))
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("stale", encoding="utf-8")
    one_click.REQUIREMENTS_MARKER.write_text("hash of an earlier requirements.txt", encoding="utf-8")
    return interpreter


def creating_venv(one_click):
    """A stand-in for ``python -m venv`` that produces an interpreter."""
    def fake_run(command, **_kwargs):
        created = Path(command[-1]) / "bin" / "python"
        created.parent.mkdir(parents=True)
        created.write_text("fresh", encoding="utf-8")

    return fake_run


def test_create_env_rebuilds_an_environment_built_on_an_unsupported_python(sandboxed, monkeypatch, capsys):
    """The 3.14 case, which is what a machine that upgraded its Python ends up
    with: the environment exists and its interpreter runs, so only the version
    tells the installer that pip is about to fail."""
    one_click = sandboxed
    stale = stale_env(one_click)
    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 14))
    monkeypatch.setattr(one_click, "run", creating_venv(one_click))

    interpreter = one_click.create_env("python3.12")

    assert interpreter == one_click.ENV_DIR / "bin" / "python"
    assert not stale.exists()
    # The marker lives outside the environment: left behind, it would tell the
    # next step that the new, empty environment already has its dependencies.
    assert not one_click.REQUIREMENTS_MARKER.exists()
    assert "3.14" in capsys.readouterr().out


def test_create_env_keeps_an_environment_the_pins_cover(sandboxed, monkeypatch):
    one_click = sandboxed
    existing = stale_env(one_click)
    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 13))
    monkeypatch.setattr(one_click, "run", lambda *_a, **_k: pytest.fail("must not rebuild"))

    assert one_click.create_env("python3.12") == existing
    assert one_click.REQUIREMENTS_MARKER.is_file()


def test_create_env_rebuilds_an_environment_whose_interpreter_no_longer_runs(sandboxed, monkeypatch):
    one_click = sandboxed
    stale_env(one_click)
    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: None)
    monkeypatch.setattr(one_click, "run", creating_venv(one_click))

    assert one_click.create_env("python3.12") == one_click.ENV_DIR / "bin" / "python"


def test_create_env_refuses_to_delete_the_interpreter_it_is_running(sandboxed, monkeypatch):
    """Windows cannot unlink a running python.exe. An installer started from
    the stale environment has to say which file to run instead of leaving a
    half-deleted one behind."""
    one_click = sandboxed
    stale = stale_env(one_click)
    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 14))
    monkeypatch.setattr(one_click.sys, "executable", str(stale))

    with pytest.raises(SystemExit, match="running from the environment"):
        one_click.create_env("python3.12")
    assert stale.exists()


def test_install_requirements_refuses_an_environment_the_pins_cannot_fill(sandboxed, monkeypatch):
    """pip's own diagnosis is forty lines of ignored versions ending in "No
    matching distribution found", which is the error this replaces."""
    one_click = sandboxed
    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 14))
    monkeypatch.setattr(one_click, "run", lambda *_a, **_k: pytest.fail("pip must not run"))

    with pytest.raises(SystemExit, match="3.14"):
        one_click.install_requirements("python")


def test_install_requirements_records_what_it_installed(sandboxed, monkeypatch):
    one_click = sandboxed
    commands = []
    monkeypatch.setattr(one_click, "interpreter_version", lambda executable: (3, 12))
    monkeypatch.setattr(one_click, "run", lambda command, **_k: commands.append(command))

    one_click.install_requirements("python")
    assert commands[-1][:4] == ["python", "-m", "pip", "install"]
    assert one_click.REQUIREMENTS_MARKER.is_file()

    commands.clear()
    one_click.install_requirements("python")
    assert commands == []


def test_start_windows_bat_enforces_the_same_range(one_click):
    """The .bat chooses the interpreter that runs one_click.py, so a range that
    disagrees with MIN_PYTHON/MAX_PYTHON reinstates the failure before any of
    the Python above gets a chance to reject it."""
    text = (REPO_ROOT / "start_windows.bat").read_text(encoding="utf-8")
    minors = range(one_click.MAX_PYTHON[1] - 1, one_click.MIN_PYTHON[1] - 1, -1)

    probed = re.search(r"for %%V in \(([^)]*)\) do", text).group(1).split()
    assert probed == ["{0}.{1}".format(one_click.MIN_PYTHON[0], minor) for minor in minors]

    guard = "({0},{1}) <= sys.version_info[:2] < ({2},{3})".format(
        *one_click.MIN_PYTHON, *one_click.MAX_PYTHON)
    assert guard in text
    # Both interpreters a previous run may have left behind are version-tested,
    # or a stale environment is picked up again and pip fails exactly as before.
    assert text.count("call :supported") == 2


# ── launcher ─────────────────────────────────────────────────────────────────

def test_launcher_imports_only_the_standard_library():
    """``python app.py`` runs before the environment exists, so every top-level
    import in it has to resolve against a bare interpreter. A third-party import
    added here turns "not installed yet" into a traceback."""
    import ast

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    allowed = {"os", "sys", "pathlib", "subprocess", "__future__"}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            assert {alias.name.split(".")[0] for alias in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed


def test_launcher_reexec_is_guarded_against_looping(monkeypatch, tmp_path):
    """If the environment exists but is broken, re-exec must stop after one
    hop rather than spawn interpreters forever."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prompt_master_launcher", Path(__file__).resolve().parents[1] / "app.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    monkeypatch.setattr(launcher, "env_python", lambda: tmp_path / "python")
    monkeypatch.setenv(launcher.REEXEC_GUARD, "1")
    launcher.reexec_if_needed()      # returns instead of re-executing


# ── changing what runs the model, without touching the model ─────────────────

def _installed_state(tmp_path, monkeypatch=None, *, quantization="Q4_K_M",
                     runtime_id="llama-runtime-cuda12", gpu_layers="all"):
    """An install that has already been provisioned, on disk.

    Written out rather than provisioned: this is the starting position for the
    switch under test, so it must not depend on the code path being tested or
    on the CUDA probe the tests around it disallow.
    """
    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    for relative in ("models/model.gguf", "models/mmproj.gguf"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")
    server = tmp_path / "runtime" / "llama-server.exe"
    server.parent.mkdir(parents=True, exist_ok=True)
    server.write_bytes(b"")
    paths.state_file.write_text(json.dumps({
        "runtime": "runtime/llama-server.exe", "runtime_id": runtime_id,
        "model": "models/model.gguf", "mmproj": "models/mmproj.gguf",
        "mode": "gpu", "gpu_index": 0, "gpu_uuid": "GPU-0",
        "gpu_name": "NVIDIA GeForce RTX 4090", "gpu_device": "CUDA0",
        "gpu_device_name": "RTX 4090", "quantization": quantization,
        "context_size": 16384, "gpu_layers": gpu_layers,
    }), encoding="utf-8")
    return paths


def test_a_device_switch_needing_no_new_runtime_downloads_nothing(tmp_path, monkeypatch):
    """The same card in mixed mode is the same llama.cpp build and the same
    model: there is nothing to fetch and nothing to unpack, only state to
    rewrite."""
    monkeypatch.setattr(installer, "list_llama_devices", lambda *_a, **_k: ("CUDA0", "RTX 4090"))
    monkeypatch.setattr(installer, "download", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(installer, "extract_zips_atomic", lambda *a, **k: pytest.fail("extracted"))
    paths = _installed_state(tmp_path, monkeypatch)

    state = installer.switch_device(paths, mixed())

    assert state["mode"] == "mixed"
    assert state["gpu_layers"] == NO_OFFLOAD
    # The 16-27 GiB that was not re-downloaded is the point of the whole thing.
    assert state["model"] == "models/model.gguf" and state["mmproj"] == "models/mmproj.gguf"
    assert state["quantization"] == "Q4_K_M"


def test_switching_to_the_processor_fetches_only_the_cpu_runtime(tmp_path, monkeypatch):
    fetched, extracted = [], []
    components = {key: Component(key, f"https://example.invalid/{key}.zip", f"cache/downloads/{key}.zip",
                                 None, "a" * 64, "1")
                  for key in (CPU_RUNTIME, "llama-runtime-cuda12", "llama-runtime-cuda12-cudart",
                              "model-Q4_K_M", "mmproj")}
    monkeypatch.setattr(installer, "load_components", lambda: components)
    monkeypatch.setattr(installer, "download",
                        lambda component, target, *a, **k: fetched.append(component.component_id) or target)
    monkeypatch.setattr(installer, "extract_zips_atomic",
                        lambda archives, target: extracted.append(target))
    monkeypatch.setattr(installer, "list_llama_devices",
                        lambda *_a, **_k: pytest.fail("must not probe for a CUDA device"))
    paths = _installed_state(tmp_path, monkeypatch)

    state = installer.switch_device(paths, cpu())

    assert fetched == [CPU_RUNTIME], "only the runtime, never the model"
    assert extracted == [tmp_path / "runtime"]
    assert state["mode"] == "cpu" and state["gpu_device"] == CPU_DEVICE
    assert state["runtime_id"] == CPU_RUNTIME
    assert state["model"] == "models/model.gguf"


def test_a_switch_back_to_the_same_card_keeps_its_partial_offload(tmp_path, monkeypatch):
    """A card set up with a layer count should still have it when it is chosen
    again — and must not hand that number to a different card."""
    monkeypatch.setattr(installer, "list_llama_devices", lambda *_a, **_k: ("CUDA0", "RTX 4090"))
    paths = _installed_state(tmp_path, monkeypatch, gpu_layers="40")

    assert installer.switch_device(paths, gpu())["gpu_layers"] == "40"
    assert installer.switch_device(paths, gpu(index=1))["gpu_layers"] == installer.FULL_OFFLOAD


def test_a_switch_refuses_when_the_model_it_would_keep_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "list_llama_devices", lambda *_a, **_k: ("CUDA0", "RTX 4090"))
    paths = _installed_state(tmp_path, monkeypatch)
    (tmp_path / "models" / "model.gguf").unlink()

    with pytest.raises(RuntimeError, match="model is missing"):
        installer.switch_device(paths, mixed())


def test_a_switch_refuses_on_an_install_that_never_finished(tmp_path):
    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    with pytest.raises(RuntimeError, match="no recorded model"):
        installer.switch_device(paths, cpu())


def test_the_state_records_which_runtime_is_unpacked(tmp_path, monkeypatch):
    """Every build extracts to the same llama-server path, so without this a
    later switch cannot tell what is on disk."""
    monkeypatch.setattr(installer, "list_llama_devices", lambda *_a, **_k: ("CUDA0", "RTX 5090"))
    installed = installer.Installed("runtime/llama-server.exe", "models/m.gguf", "models/p.gguf")

    state = installer.write_state(AppPaths(tmp_path), gpu(name="NVIDIA GeForce RTX 5090"),
                                  "Q6_K_P", installed)

    assert state["runtime_id"] == "llama-runtime-cuda13"
    assert installer.runtime_component_ids(cpu()) == (CPU_RUNTIME,)
    assert installer.runtime_component_ids(gpu()) == ("llama-runtime-cuda12",
                                                      "llama-runtime-cuda12-cudart")


# ── running a model setup did not download ───────────────────────────────────

def test_choosing_a_model_records_it_and_changes_nothing_else(tmp_path):
    """The narrow half of setup: which weights, and nothing about the machine
    they run on. No download, no runtime change, no device change."""
    paths = _installed_state(tmp_path)
    before = json.loads(paths.state_file.read_text(encoding="utf-8"))
    other = tmp_path / "models" / "Gemma-3-27B-Q6_K_P.gguf"
    other.write_bytes(b"other weights")

    state = model_choice.choose(paths, other)

    assert state["model"] == "models/Gemma-3-27B-Q6_K_P.gguf"
    assert state["quantization"] == "Q6_K_P"
    untouched = ("runtime", "runtime_id", "mode", "gpu_index", "gpu_device", "gpu_device_name",
                 "context_size", "gpu_layers")
    assert {key: state[key] for key in untouched} == {key: before[key] for key in untouched}


def test_a_model_outside_the_install_root_is_recorded_and_found_where_it_is(tmp_path):
    """16-27 GiB is not moved into the install root to satisfy a rule about
    tidiness. It stays where it was chosen, and is still refused as a managed
    path, which is what keeps the runtime check meaning something."""
    paths = _installed_state(tmp_path / "install")
    outside = tmp_path / "elsewhere" / "Llama-3.1-8B-Q5_K_M.gguf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"weights")

    state = model_choice.choose(paths, outside)

    assert Path(state["model"]) == outside.resolve()
    assert paths.locate(state["model"]) == outside.resolve()
    assert state["quantization"] == "Q5_K_M"
    with pytest.raises(ValueError):
        paths.contained(state["model"])


def test_a_projector_can_be_given_with_the_model_and_taken_away_again(tmp_path):
    """Optional in both directions: supplying one turns image input on, and
    leaving it out turns it off rather than keeping the last model's."""
    paths = _installed_state(tmp_path)
    model = tmp_path / "models" / "other.gguf"
    model.write_bytes(b"weights")
    projector = tmp_path / "models" / "other-mmproj-f16.gguf"
    projector.write_bytes(b"projector")

    assert model_choice.choose(paths, model, projector)["mmproj"] == "models/other-mmproj-f16.gguf"
    assert model_choice.choose(paths, model)["mmproj"] == ""


def test_a_file_that_is_not_there_is_refused_before_anything_is_written(tmp_path):
    paths = _installed_state(tmp_path)
    before = paths.state_file.read_text(encoding="utf-8")
    model = tmp_path / "models" / "other.gguf"
    model.write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="no model file"):
        model_choice.choose(paths, tmp_path / "missing.gguf")
    with pytest.raises(RuntimeError, match="no vision projector"):
        model_choice.choose(paths, model, tmp_path / "missing-mmproj.gguf")
    assert paths.state_file.read_text(encoding="utf-8") == before


def test_choosing_a_model_needs_a_runtime_to_run_it_with(tmp_path):
    """This is not a way around setup: without llama.cpp on disk there is
    nothing to hand the weights to."""
    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="no llama.cpp runtime"):
        model_choice.choose(paths, model)


@pytest.mark.parametrize("stem,expected", [
    ("Gemma-3-27B-it-Q6_K_P", "Q6_K_P"),
    ("mistral-7b-instruct-v0.2.Q4_K_M", "Q4_K_M"),
    ("Llama-3.1-8B-IQ3_XXS", "IQ3_XXS"),
    ("qwen2.5-7b-instruct-q8_0", "Q8_0"),
    ("some-merge-bf16", "BF16"),
    ("a-finetune-nobody-labelled", "a-finetune-nobody-labelled"),
])
def test_a_model_is_named_by_its_quantization_whenever_the_file_name_says_one(stem, expected):
    """"Model: Q6_K_P" is what the status bar has always said, and a file
    chosen by hand should read the same way rather than as a long stem."""
    assert model_choice.describe(Path(f"/models/{stem}.gguf")) == expected


def test_a_projector_beside_the_model_is_suggested_and_only_suggested(tmp_path):
    """A model and its projector come from the same repository and land in the
    same folder. That is a suggestion for a box, not a pairing to assume."""
    folder = tmp_path / "gguf"
    folder.mkdir()
    model = folder / "Gemma-3-27B-Q6_K_P.gguf"
    model.write_bytes(b"weights")

    assert model_choice.projector_beside(model) is None
    projector = folder / "mmproj-Gemma-3-27B-f16.gguf"
    projector.write_bytes(b"projector")
    assert model_choice.projector_beside(model) == projector


def test_what_is_recorded_now_is_what_the_dialog_opens_on(tmp_path):
    paths = _installed_state(tmp_path)

    current = model_choice.recorded(paths)

    assert current.model == (tmp_path / "models" / "model.gguf").resolve()
    assert current.sees and current.mmproj == (tmp_path / "models" / "mmproj.gguf").resolve()
    assert model_choice.recorded(AppPaths(tmp_path / "never-set-up")) is None


def test_a_device_switch_keeps_a_hand_picked_model_and_its_absent_projector(tmp_path, monkeypatch):
    """Changing what runs the model must not quietly put the downloaded one
    back, and must not invent a projector the chosen model never had."""
    monkeypatch.setattr(installer, "list_llama_devices", lambda *_a, **_k: ("CUDA0", "RTX 4090"))
    monkeypatch.setattr(installer, "download", lambda *a, **k: pytest.fail("downloaded"))
    paths = _installed_state(tmp_path / "install")
    outside = tmp_path / "elsewhere" / "Chosen-Q4_K_M.gguf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"weights")
    model_choice.choose(paths, outside)

    state = installer.switch_device(paths, mixed())

    assert Path(state["model"]) == outside.resolve()
    assert state["mmproj"] == "" and state["quantization"] == "Q4_K_M"
    assert state["mode"] == "mixed"


# ── a model with no projector is a model that cannot be shown a picture ──────

def _runnable(tmp_path, *, projector=True):
    """An install whose files are all present, ready for InferenceService."""
    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    for relative in ("runtime/llama-server.exe", "models/m.gguf", "models/p.gguf"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stand-in")
    paths.state_file.write_text(json.dumps({
        "runtime": "runtime/llama-server.exe", "model": "models/m.gguf",
        "mmproj": "models/p.gguf" if projector else "", "gpu_index": 0,
        "gpu_device": "CUDA0", "context_size": 16384, "gpu_layers": "all",
        "quantization": "Q4_K_M"}), encoding="utf-8")
    return paths


def test_the_server_is_started_without_mmproj_when_there_is_no_projector(tmp_path, monkeypatch):
    """--mmproj is left off the command rather than passed something empty,
    which llama-server would refuse to start on."""
    from prompt_master.inference.llama_process import LlamaProcess

    captured = {}

    class FakePopen:
        def __init__(self, command, env=None, **_kwargs):
            captured["command"] = command

        def poll(self): return None

    monkeypatch.setattr("prompt_master.inference.llama_process.subprocess.Popen", FakePopen)
    LlamaProcess().start(tmp_path / "llama-server.exe", tmp_path / "m.gguf", None,
                         0, "CUDA0", 16384, tmp_path / "log.txt")

    command = captured["command"]
    assert "--mmproj" not in command
    assert command[command.index("--model") + 1] == str(tmp_path / "m.gguf")


def test_text_still_runs_on_a_model_that_has_no_projector(tmp_path, monkeypatch):
    from prompt_master.inference.service import InferenceService

    started = []
    service = InferenceService(_runnable(tmp_path, projector=False))
    monkeypatch.setattr(service.process, "start", lambda *args, **kwargs: started.append(args))
    monkeypatch.setattr(service.process, "wait_ready", lambda timeout: None)

    service.client()

    assert started[0][2] is None, "no projector was handed to the server"
    assert not service.vision_ready()


def test_an_image_request_without_a_projector_says_which_menu_fixes_it(tmp_path):
    """Never a silent downgrade to text — the prompt this app would write with
    the still off the wire describes a scene nobody chose."""
    from prompt_master.inference.service import InferenceService

    with pytest.raises(RuntimeError, match="no vision projector"):
        InferenceService(_runnable(tmp_path, projector=False)).client(needs_vision=True)


def test_a_projector_that_was_recorded_and_then_deleted_is_still_an_error(tmp_path):
    """Absent by choice and absent by accident are different things: one is a
    model that answers text, the other is a broken install."""
    from prompt_master.inference.service import InferenceService

    paths = _runnable(tmp_path)
    (tmp_path / "models" / "p.gguf").unlink()

    with pytest.raises(RuntimeError, match="vision projector is missing"):
        InferenceService(paths).client()


# ── a supplied file keeps its own name ───────────────────────────────────────

def test_fetch_records_the_supplied_file_under_the_name_it_arrived_with(tmp_path, monkeypatch):
    """End to end, with the real adopt: what setup writes into the state file
    is the path the model actually went to. Nothing renames a file somebody
    handed over, and nothing needs it renamed."""
    components = installer.load_components()

    def fake_download(component, target, progress=None, notice=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target, "w") as bundle:
                bundle.writestr("llama-server.exe", b"exe")
        else:
            target.write_bytes(b"downloaded")
        return target

    monkeypatch.setattr(installer, "download", fake_download)
    mine = tmp_path / "downloads" / "MyMerge-Q5_K_M.gguf"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"pretend gguf")
    paths = AppPaths(tmp_path / "install")

    installed = installer.fetch(paths, gpu(name="NVIDIA GeForce RTX 5090", total=32607), "Q6_K_P",
                                sources={"model-Q6_K_P": LocalSource(mine, checked=True)})

    assert installed.model == "models/MyMerge-Q5_K_M.gguf"
    assert (paths.root / installed.model).is_file()
    assert not (paths.root / components["model-Q6_K_P"].destination).exists()
    assert not mine.exists(), "it was moved, not copied"
    # What was downloaded still lands on the pinned name, which is what the
    # resume and the download cache are keyed on.
    assert installed.mmproj == components["mmproj"].destination


def test_a_model_with_any_name_at_all_is_what_the_server_is_pointed_at(tmp_path, monkeypatch):
    """Nothing downstream parses a model's name: the recorded path is handed to
    llama-server as it stands, spaces, brackets and all."""
    from prompt_master.inference.service import InferenceService

    odd = "Some Merge (v2) [uncensored].Q5_K_M.gguf"
    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    for relative in ("runtime/llama-server.exe", f"models/{odd}"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stand-in")
    paths.state_file.write_text(json.dumps({
        "runtime": "runtime/llama-server.exe", "model": f"models/{odd}", "mmproj": "",
        "gpu_index": 0, "gpu_device": "CUDA0", "context_size": 16384, "gpu_layers": "all",
    }), encoding="utf-8")

    started = []
    service = InferenceService(paths)
    monkeypatch.setattr(service.process, "start", lambda *args, **kwargs: started.append(args))
    monkeypatch.setattr(service.process, "wait_ready", lambda timeout: None)
    service.client()

    assert started[0][1] == tmp_path / "models" / odd


def test_a_projector_that_is_not_named_like_the_pinned_one_is_still_found(tmp_path):
    """A model supplied by hand carries whatever naming its publisher chose.
    Insisting on the pinned projector's name would mean only the pinned pair is
    ever offered."""
    component = installer.load_components()["mmproj"]
    folder = tmp_path / "gguf"
    folder.mkdir()
    model = folder / "MyMerge-Q5_K_M.gguf"
    model.write_bytes(b"weights")

    assert setup_cli._projector_beside(component, model) is None
    theirs = folder / "MyMerge-mmproj-f16.gguf"
    theirs.write_bytes(b"a stand-in for the projector")
    assert setup_cli._projector_beside(component, model) == theirs


def test_a_projector_beside_an_unpinned_model_can_be_taken_anyway(monkeypatch, tmp_path):
    """The same question the model itself gets, for the same reason: the
    projector beside an unpinned model is the one that model needs, and it will
    not match a hash pinned to a different build."""
    source = their_file(tmp_path, name="MyMerge-Q5_K_M.gguf")
    projector = source.parent / "MyMerge-mmproj-f16.gguf"
    projector.write_bytes(b"a stand-in for the projector")
    monkeypatch.setattr(setup_cli, "accept_file",
                        lambda component, path: ("that is not the pinned projector", None))
    # Use a local model / the path / use it anyway / take the projector / anyway.
    answers(monkeypatch, "Y", str(source), "Y", "Y", "Y")

    quant, sources = setup_cli.ask_local_model("Q6_K_P", setup_cli.Steps(4))

    assert quant == "Q6_K_P" and set(sources) == {"model-Q6_K_P", "mmproj"}
    assert sources["mmproj"].path == projector


def test_declining_that_question_leaves_the_projector_to_download(monkeypatch, tmp_path):
    source = their_file(tmp_path, name="MyMerge-Q5_K_M.gguf")
    (source.parent / "MyMerge-mmproj-f16.gguf").write_bytes(b"a stand-in for the projector")
    monkeypatch.setattr(setup_cli, "accept_file",
                        lambda component, path: ("that is not the pinned projector", None))
    answers(monkeypatch, "Y", str(source), "Y", "Y", "N")

    _quant, sources = setup_cli.ask_local_model("Q6_K_P", setup_cli.Steps(4))
    assert set(sources) == {"model-Q6_K_P"}
