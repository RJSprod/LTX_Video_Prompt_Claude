"""Tests for the parts the source distribution adds.

The prompt engine is covered by ``test_upstream_parity.py`` and must not change.
What is new here is everything around it: where an install lives now that there
is no frozen executable to anchor to, which artifacts a given GPU resolves to,
and the console setup that the one-click installer drives.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest

from prompt_master.core.models import GpuInfo
from prompt_master.core.paths import DEFAULT_SUBDIR, ROOT_ENV, AppPaths
from prompt_master.inference.device_detection import (PINNED, QUANTIZATIONS,
    recommended_quantization, runtime_component_id, vram_shortfall_mb)
from prompt_master.provisioning import installer
from prompt_master import setup_cli


def gpu(name="NVIDIA GeForce RTX 4090", total=24564, compute=None, index=0):
    return GpuInfo(index, f"GPU-{index}", name, total, total - 2000, "560.94", compute)


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


# ── console setup ────────────────────────────────────────────────────────────

def test_setup_flags_cover_every_question():
    """An unattended reinstall must not need a human, or the installer cannot
    be scripted."""
    options = setup_cli.parse_args(["--dir", "D:/PM", "--gpu", "1", "--quant", "Q8_K_P", "--yes"])
    assert (options.directory, options.gpu, options.quant, options.yes) == ("D:/PM", 1, "Q8_K_P", True)
    assert options.context_size == installer.DEFAULT_CONTEXT_SIZE
    assert options.gpu_layers == installer.FULL_OFFLOAD


def test_preselected_gpu_index_must_exist(monkeypatch):
    monkeypatch.setattr(setup_cli, "detect_gpus", lambda: [gpu(index=0)])
    assert setup_cli.ask_gpu(0).physical_index == 0
    with pytest.raises(SystemExit, match="No GPU with index 3"):
        setup_cli.ask_gpu(3)


def test_missing_driver_is_reported_not_raised(monkeypatch):
    def explode():
        raise RuntimeError("nvidia-smi is not available.")

    monkeypatch.setattr(setup_cli, "detect_gpus", explode)
    with pytest.raises(SystemExit, match="nvidia-smi"):
        setup_cli.ask_gpu(None)


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


def test_single_gpu_is_not_a_question(monkeypatch, capsys):
    monkeypatch.setattr(setup_cli, "detect_gpus", lambda: [gpu(index=2)])
    monkeypatch.setattr(setup_cli, "ask", lambda *_a, **_k: pytest.fail("should not ask"))
    assert setup_cli.ask_gpu(None).physical_index == 2


def test_no_gpu_is_a_clean_exit(monkeypatch):
    monkeypatch.setattr(setup_cli, "detect_gpus", lambda: [])
    with pytest.raises(SystemExit, match="no CUDA GPU"):
        setup_cli.ask_gpu(None)


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
