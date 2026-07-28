"""Console setup — the same three questions the Qt wizard asks.

This is what the one-click installer runs after it has built the Python
environment, and what ``python app.py --setup`` re-runs later. It asks where to
install, which GPU to use and which quantization to download, then hands off to
``provisioning.installer`` — the same pipeline the Qt wizard uses, so answering
here and answering there produce the same install.

Every question can also be supplied as a flag, which is what makes an unattended
reinstall possible:

    python app.py --setup --dir D:/PromptMaster --gpu 0 --quant Q6_K_P --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from prompt_master.core.models import GpuInfo
from prompt_master.core.paths import AppPaths
from prompt_master.inference.device_detection import (QUANTIZATIONS, detect_gpus,
    recommended_quantization, runtime_component_id, vram_shortfall_mb)
from prompt_master.provisioning import installer

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


def ask_gpu(preselected: int | None = None, steps: Steps | None = None) -> GpuInfo:
    if preselected is None and steps is not None:
        steps.ask("which GPU should run the model?")
    try:
        gpus = detect_gpus()
    except RuntimeError as exc:
        # A missing driver is the single most likely first-run failure, and a
        # traceback is a poor way to say "install the NVIDIA driver".
        raise SystemExit(f"{exc}\n\nThis application runs the model on an NVIDIA GPU through llama.cpp.")
    if not gpus:
        raise SystemExit(
            "nvidia-smi reported no CUDA GPU.\n"
            "This application runs the model on an NVIDIA GPU through llama.cpp; there is no CPU path."
        )
    if preselected is not None:
        for gpu in gpus:
            if gpu.physical_index == preselected:
                print(f"Using GPU {preselected}: {gpu.name}")
                return gpu
        raise SystemExit(f"No GPU with index {preselected}. Detected: " +
                         ", ".join(f"{g.physical_index}={g.name}" for g in gpus))
    if len(gpus) == 1:
        gpu = gpus[0]
        print(f"Found one CUDA GPU: {gpu.name} ({gpu.memory_total_mb} MiB, driver {gpu.driver_version})")
        return gpu
    labels = [f"{gpu.name} — {gpu.memory_total_mb} MiB — driver {gpu.driver_version}" for gpu in gpus]
    return gpus[choose("What is your GPU?", labels)]


def ask_quantization(gpu: GpuInfo, preselected: str | None = None, steps: Steps | None = None) -> str:
    if preselected is not None:
        if preselected not in QUANTIZATIONS:
            raise SystemExit(f"Unknown quantization {preselected}. Choose one of: {', '.join(QUANTIZATIONS)}")
        print(f"Using {preselected}")
        return preselected
    if steps is not None:
        steps.ask("which model quality?")
    recommended = recommended_quantization(gpu)
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
    parser.add_argument("--quant", choices=list(QUANTIZATIONS), help="GGUF quantization to download")
    parser.add_argument("--context-size", type=int, default=installer.DEFAULT_CONTEXT_SIZE,
                        help=f"llama.cpp context size (default {installer.DEFAULT_CONTEXT_SIZE})")
    parser.add_argument("--gpu-layers", default=installer.FULL_OFFLOAD,
                        help="llama.cpp --n-gpu-layers; lower it to spill layers to system RAM on a small card")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before downloading")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    banner("Prompt Master — model and hardware setup")

    steps = Steps(sum(answer is None for answer in (options.directory, options.gpu, options.quant)))

    if options.directory:
        paths = AppPaths(Path(options.directory).expanduser().resolve())
    else:
        paths = ask_directory(AppPaths.discover().root, steps)
    paths.create_managed_dirs()

    gpu = ask_gpu(options.gpu, steps)
    quant = ask_quantization(gpu, options.quant, steps)

    banner("Downloading and verifying")
    print(f"GPU        : {gpu.name} (index {gpu.physical_index})")
    print(f"Runtime    : {runtime_component_id(gpu)}")
    print(f"Model      : {quant}")
    print(f"Directory  : {paths.root}")
    print(f"Download   : {installer.format_download_size(gpu, quant)}")
    print("\nEvery artifact is pinned by SHA-256 and verified after download. Interrupted")
    print("downloads resume, so re-running setup does not start over.\n")
    if not options.yes and not confirm("Continue?"):
        print("Setup cancelled. Nothing was downloaded.")
        return 1

    reporter = ConsoleProgress()
    try:
        installer.provision(paths, gpu, quant,
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
