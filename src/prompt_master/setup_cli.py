"""Console setup — the same questions the Qt wizard asks.

This is what the one-click installer runs after it has built the Python
environment, and what ``python app.py --setup`` re-runs later. It asks where to
install, which device to run on, which quantization to download, and whether the
model is already on this machine, then hands off to ``provisioning.installer`` —
the same pipeline the Qt wizard uses, so answering here and answering there
produce the same install.

Every question can also be supplied as a flag, which is what makes an unattended
reinstall possible:

    python app.py --setup --dir D:/PromptMaster --gpu 0 --quant Q6_K_P --yes
    python app.py --setup --cpu --quant Q4_K_M --yes
    python app.py --setup --model-file D:/models/Gemma4-...-Q6_K_P.gguf --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from prompt_master.core.models import GpuInfo
from prompt_master.core.paths import AppPaths
from prompt_master.inference.device_detection import (QUANTIZATIONS, detect_cpu, detect_devices,
    recommended_quantization, runtime_component_id, vram_shortfall_mb)
from prompt_master.provisioning import importer, installer, verifier

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── console helpers ──────────────────────────────────────────────────────────

def banner(text: str) -> None:
    width = min(shutil.get_terminal_size((80, 24)).columns, 78)
    print(f"\n{'*' * width}\n* {text}\n{'*' * width}\n")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        # Piped stdin with nothing left to read. Taking the default is right for
        # a defaulted question and impossible for one without a default, so the
        # latter must fail loudly rather than guess.
        if not default:
            raise SystemExit("Setup needs an answer but stdin is closed. Re-run interactively, or pass --dir/--gpu/--quant.")
        print(f"{prompt}{suffix}: {default}")
        return default
    return answer or default


def choose(prompt: str, options: list[str], default_index: int = 0) -> int:
    """Lettered menu, oobabooga style. Returns the chosen index."""
    for number, option in enumerate(options):
        marker = " (recommended)" if number == default_index else ""
        print(f"  {LETTERS[number]}) {option}{marker}")
    print()
    valid = LETTERS[:len(options)]
    while True:
        answer = ask(prompt, LETTERS[default_index]).strip().upper()
        if len(answer) == 1 and answer in valid:
            return valid.index(answer)
        if answer.isdigit() and 0 <= int(answer) < len(options):
            return int(answer)
        print(f"Enter one of: {', '.join(valid)}")


def confirm(prompt: str, default: bool = True) -> bool:
    answer = ask(prompt, "Y" if default else "N").strip().upper()
    return default if not answer else answer.startswith("Y")


class ConsoleProgress:
    """One redrawn progress line. Throttled so a fast disk cache does not
    produce thousands of writes to a slow Windows console."""

    def __init__(self, interval: float = 0.2):
        self.interval, self.last, self.message = interval, 0.0, ""

    def status(self, message: str) -> None:
        self.message = message
        self._draw(force=True)

    def progress(self, fraction: float) -> None:
        self._fraction = max(0.0, min(1.0, fraction))
        self._draw()

    def _draw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        fraction = getattr(self, "_fraction", 0.0)
        filled = int(fraction * 30)
        width = min(shutil.get_terminal_size((80, 24)).columns, 100) - 1
        line = f"[{'#' * filled}{'.' * (30 - filled)}] {fraction * 100:5.1f}%  {self.message}"
        sys.stdout.write("\r" + line[:width].ljust(width))
        sys.stdout.flush()

    def done(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ── the questions ────────────────────────────────────────────────────────────

class Steps:
    """Numbers the questions that are actually going to be asked.

    Answering one on the command line removes it, so the count has to be worked
    out from the flags rather than hard-coded — otherwise ``--dir`` produces a
    run that opens on "Question 2 of 3".
    """

    def __init__(self, total: int):
        self.total, self.asked = total, 0

    def ask(self, text: str) -> None:
        self.asked += 1
        banner(f"Question {self.asked} of {self.total} — {text}")


def ask_directory(default: Path, steps: Steps) -> AppPaths:
    steps.ask("where should models and runtime be installed?")
    print("The llama.cpp runtime, the GGUF model, the vision projector, the download")
    print("cache, the logs and the setup state all live under this directory.")
    print("The model alone is 16-27 GiB, so choose a drive with room.\n")
    while True:
        root = Path(ask("Installation directory", str(default))).expanduser().resolve()
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".write-test"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            print(f"\nCannot write there: {exc}\n")
            continue
        free = shutil.disk_usage(root).free
        print(f"\nUsing {root}  ({free / 2**30:.1f} GiB free)")
        return AppPaths(root)


def device_label(device: GpuInfo) -> str:
    """One line describing a device in the menu."""
    if device.is_cpu:
        return f"{device.name} — {device.memory_total_mb} MiB of system RAM — no GPU used"
    return f"{device.name} — {device.memory_total_mb} MiB — driver {device.driver_version}"


def ask_device(preselected: int | None = None, cpu: bool = False,
               steps: Steps | None = None) -> GpuInfo:
    """Which device runs the model: a CUDA GPU, or the processor.

    The processor is always the last option, so a machine with no NVIDIA driver
    has one answer rather than none, and a machine with a card can still choose
    to keep it free. A detected GPU stays the default — a run that took the
    first option before this question learned about the CPU still takes it.
    """
    if cpu:
        device = detect_cpu()
        print(f"Using the processor: {device.name} ({device.memory_total_mb} MiB of system RAM)")
        return device
    if preselected is None and steps is not None:
        steps.ask("what should run the model?")
    devices = detect_devices()
    gpus = [device for device in devices if not device.is_cpu]
    if preselected is not None:
        for gpu in gpus:
            if gpu.physical_index == preselected:
                print(f"Using GPU {preselected}: {gpu.name}")
                return gpu
        detected = ", ".join(f"{gpu.physical_index}={gpu.name}" for gpu in gpus) or "no CUDA GPU"
        raise SystemExit(f"No GPU with index {preselected}. Detected: {detected}\n"
                         "Pass --cpu to run on the processor and system RAM instead.")
    if not gpus:
        device = devices[0]
        print("No CUDA GPU was detected, so this install will run on the processor:")
        print(f"  {device.name} ({device.memory_total_mb} MiB of system RAM)")
        return device
    print(f"Found {len(gpus)} CUDA GPU(s). The processor is offered too, for a machine")
    print("whose card should stay free or whose driver is not installed.\n")
    return devices[choose("What should run the model?", [device_label(d) for d in devices])]


def read_path(prompt: str) -> Path | None:
    """A path typed or pasted at the console. Windows "Copy as path" quotes it."""
    answer = ask(prompt).strip().strip('"').strip("'")
    return Path(answer).expanduser() if answer else None


def check_file(component, path: Path) -> str:
    """The SHA-256 of ``path``, with a progress line — it is a long read."""
    print(f"\nChecking {path.name} against the pinned SHA-256 "
          f"({importer.human(path.stat().st_size)} to read)…")
    reporter = ConsoleProgress()
    reporter.status(component.component_id)
    try:
        digest = verifier.digest_of(path, lambda done, total: reporter.progress(done / total))
    finally:
        reporter.done()
    return digest


def accept_file(component, path: Path) -> tuple[str | None, str | None]:
    """Vet a supplied file: ``(refusal, what it actually is)``.

    Naming what the file is whenever the manifest can say — "that is the Q4_K_M
    build" — ends an investigation that "hash mismatch" only starts.
    """
    problem = importer.size_problem(component, path)
    if problem:
        return problem, None
    digest = check_file(component, path)
    if digest.casefold() == component.sha256.casefold():
        print("Verified: byte-for-byte the pinned build.\n")
        return None, component.component_id
    identified = installer.identify(digest)
    named = f"it is {identified}" if identified else "it is not a file this build pins"
    return f"{path.name} does not match {component.component_id} — {named}", identified


def ask_local_model(quant: str, steps: Steps, *, move: bool = True) -> tuple[str, dict]:
    """Question 4: a model already on this machine, instead of downloading it.

    Returns the quantization to install and the files to install it from — the
    quantization can change here, because a file that turns out to be a
    different build is better installed as what it is than as what was asked for.
    """
    steps.ask("do you already have the model file?")
    components = installer.load_components()
    model = components[f"model-{quant}"]
    print("The model is one large file. If you already have it — downloaded by hand,")
    print("copied from another install, or fetched with a download manager — setup can")
    print("take it from disk instead of downloading it again.\n")
    print(f"For {quant} that file is")
    print(f"  {Path(model.destination).name}  ({importer.human(model.size)})\n")
    print("It is MOVED into the installation directory, not copied, so you are not left")
    print("with two of them. Everything else is still downloaded normally.\n")
    if not confirm("Use a model file you already have?", default=False):
        return quant, {}

    while True:
        path = read_path("Full path to the .gguf file (blank to download instead)")
        if path is None:
            return quant, {}
        refusal, identified = accept_file(model, path)
        if refusal is None:
            break
        print(f"\n{refusal}")
        switch = (identified or "").removeprefix("model-")
        if switch in QUANTIZATIONS and confirm(f"\nInstall {switch} instead, since that is what you have?"):
            quant, model = switch, components[identified]
            break
        if confirm("\nUse it anyway? The prompts are tuned for the pinned build", default=False):
            break
        print()

    sources = {f"model-{quant}": importer.LocalSource(path, move=move, checked=True)}
    projector = _projector_beside(components["mmproj"], path)
    if projector is not None and confirm(f"\n{projector.name} is beside it — use that too?"):
        if accept_file(components["mmproj"], projector)[0] is None:
            sources["mmproj"] = importer.LocalSource(projector, move=move, checked=True)
        else:
            print("Leaving the projector to download.\n")
    return quant, sources


def _projector_beside(component, model_path: Path) -> Path | None:
    """The vision projector, if it sits beside the model — it usually does, both
    being files from the same repository."""
    candidate = model_path.parent / Path(component.destination).name
    return candidate if candidate.is_file() and candidate.stat().st_size == component.size else None


def ask_quantization(gpu: GpuInfo, preselected: str | None = None, steps: Steps | None = None) -> str:
    if preselected is not None:
        if preselected not in QUANTIZATIONS:
            raise SystemExit(f"Unknown quantization {preselected}. Choose one of: {', '.join(QUANTIZATIONS)}")
        print(f"Using {preselected}")
        return preselected
    if steps is not None:
        steps.ask("which model quality?")
    recommended = recommended_quantization(gpu)
    if gpu.is_cpu:
        # Sized against nothing, so measured against nothing: the download is
        # what differs between these three here, and all three are offered.
        print(f"{gpu.name} — {gpu.memory_total_mb} MiB of system RAM.\n")
        labels = [f"{quant} — {installer.format_download_size(gpu, quant)} to download"
                  for quant in QUANTIZATIONS]
        quant = QUANTIZATIONS[choose("Which quantization?", labels,
                                     list(QUANTIZATIONS).index(recommended))]
        print("\nThis install runs on the processor and system RAM. No NVIDIA GPU or driver is")
        print("used, and none is required. You can change the quantization at any time with")
        print("`python app.py --setup`.")
        return quant
    labels = []
    for quant in QUANTIZATIONS:
        shortfall = vram_shortfall_mb(gpu, quant)
        fit = f"needs ~{shortfall} MiB more VRAM than you have" if shortfall else "fits in VRAM"
        labels.append(f"{quant} — {installer.format_download_size(gpu, quant)} to download — {fit}")
    default_index = list(QUANTIZATIONS).index(recommended)
    print(f"{gpu.name} reports {gpu.memory_total_mb} MiB of VRAM.\n")
    quant = QUANTIZATIONS[choose("Which quantization?", labels, default_index)]
    if vram_shortfall_mb(gpu, quant):
        print("\nThat quantization is larger than this card comfortably holds. It will still")
        print("install, but llama.cpp will spill layers into system RAM and generation will")
        print("be slow. You can lower it later with `python app.py --setup`.")
    return quant


# ── entry point ──────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="app.py --setup", add_help=True,
                                     description="Configure the local model and runtime.")
    parser.add_argument("--dir", dest="directory", help="Installation directory for models and runtime")
    parser.add_argument("--gpu", type=int, help="Physical GPU index, as reported by nvidia-smi")
    parser.add_argument("--cpu", action="store_true",
                        help="Run on the processor and system RAM instead of a GPU")
    parser.add_argument("--quant", choices=list(QUANTIZATIONS), help="GGUF quantization to download")
    parser.add_argument("--context-size", type=int, default=installer.DEFAULT_CONTEXT_SIZE,
                        help=f"llama.cpp context size (default {installer.DEFAULT_CONTEXT_SIZE})")
    parser.add_argument("--gpu-layers", default=installer.FULL_OFFLOAD,
                        help="llama.cpp --n-gpu-layers; lower it to spill layers to system RAM on a "
                             "small card. Not used with --cpu, which offloads nothing by definition")
    parser.add_argument("--model-file", help="Install this .gguf instead of downloading the model")
    parser.add_argument("--mmproj-file", help="Install this .gguf instead of downloading the vision projector")
    parser.add_argument("--keep-source", action="store_true",
                        help="Copy --model-file/--mmproj-file into place instead of moving them")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before downloading")
    return parser.parse_args(argv)


def supplied_files(options: argparse.Namespace, quant: str) -> dict:
    """The files named on the command line, each vetted before setup starts.

    An unattended run gets no "use it anyway?" question, so a file that is not
    the pinned artifact stops here rather than becoming an install that claims
    to be something it is not.
    """
    components = installer.load_components()
    wanted = {f"model-{quant}": options.model_file, "mmproj": options.mmproj_file}
    sources = {}
    for key, given in wanted.items():
        if not given:
            continue
        path = Path(given).expanduser()
        refusal, _ = accept_file(components[key], path)
        if refusal is not None:
            raise SystemExit(f"{refusal}\nSupply the pinned file, or drop the flag to download it.")
        sources[key] = importer.LocalSource(path, move=not options.keep_source, checked=True)
    return sources


def run(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.cpu and options.gpu is not None:
        raise SystemExit("--cpu and --gpu name different devices. Pass one of them.")
    banner("Prompt Master — model and hardware setup")

    # The local-file question is skipped when a file was named on the command
    # line, and when --yes says nobody is watching.
    asks_local = not (options.model_file or options.yes)
    answered = (options.directory is not None, options.gpu is not None or options.cpu,
                options.quant is not None)
    steps = Steps(sum(not given for given in answered) + asks_local)

    if options.directory:
        paths = AppPaths(Path(options.directory).expanduser().resolve())
    else:
        paths = ask_directory(AppPaths.discover().root, steps)
    paths.create_managed_dirs()

    gpu = ask_device(options.gpu, options.cpu, steps)
    quant = ask_quantization(gpu, options.quant, steps)
    if asks_local:
        quant, sources = ask_local_model(quant, steps, move=not options.keep_source)
    else:
        sources = supplied_files(options, quant)

    banner("Downloading and verifying")
    print(f"Device     : {gpu.name}" + ("" if gpu.is_cpu else f" (index {gpu.physical_index})"))
    print(f"Runtime    : {runtime_component_id(gpu)}")
    print(f"Model      : {quant}")
    print(f"Directory  : {paths.root}")
    for key, source in sources.items():
        print(f"From disk  : {key} ← {source.path}  ({'moved' if source.move else 'copied'})")
    print(f"Download   : {installer.format_download_size(gpu, quant, sources)}")
    print("\nEvery artifact is pinned by SHA-256 and verified after download. A dropped")
    print("connection is retried from where it stopped, and interrupted downloads resume,")
    print("so re-running setup does not start over.\n")
    if not options.yes and not confirm("Continue?"):
        print("Setup cancelled. Nothing was downloaded.")
        return 1

    reporter = ConsoleProgress()
    try:
        installer.provision(paths, gpu, quant,
                            sources=sources,
                            context_size=options.context_size,
                            gpu_layers=options.gpu_layers,
                            on_status=reporter.status,
                            on_progress=reporter.progress)
    except KeyboardInterrupt:
        reporter.done()
        print("\nSetup interrupted. Partial downloads are kept and will resume next time.")
        return 130
    except Exception as exc:
        reporter.done()
        print(f"\nSetup failed: {exc}\n")
        print(f"The llama-server log, if the runtime got far enough to write one, is at:\n  {paths.logs / 'llama-server.log'}")
        return 1
    reporter.done()

    banner("Setup complete")
    print(f"Models and runtime : {paths.root}")
    print(f"Launch             : python app.py")
    print(f"Reconfigure        : python app.py --setup\n")
    return 0


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
