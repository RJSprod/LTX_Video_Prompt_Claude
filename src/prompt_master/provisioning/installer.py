"""The provisioning pipeline, shared by both front ends.

Setup runs in two places — the console installer (``prompt_master.setup_cli``,
driven by the one-click installer) and the Qt "Models and Hardware" wizard — and
they must provision identically. Everything either one does beyond drawing its
own widgets lives here, so there is one download list, one verification step,
one state file and one validation pass rather than two that can drift.

The order below is load-bearing and is upstream's:

* nothing is written to ``setup-state.json`` until every artifact has been
  downloaded, hash-verified and extracted, so a half-finished run never leaves
  behind state the app would try to launch from;
* the state file is validated by actually generating — one text request and one
  image request — and is deleted again if either fails, because a runtime that
  starts but cannot answer is not a working install.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from prompt_master.core.config import atomic_write_json
from prompt_master.core.models import GpuInfo, PromptRequest
from prompt_master.core.paths import AppPaths
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.inference.device_detection import (CPU_DEVICE, NO_OFFLOAD,
    list_llama_devices, runtime_component_id)
from prompt_master.provisioning.downloader import download
from prompt_master.provisioning.extractor import extract_zips_atomic
from prompt_master.provisioning.importer import LocalSource, adopt
from prompt_master.provisioning.manifest import Component, load_manifest

# Upstream's context size. Exposed here because it is the one setup value that
# trades VRAM against how long a brief may be.
DEFAULT_CONTEXT_SIZE = 16384

# llama.cpp's own token for "put every layer on the GPU".
FULL_OFFLOAD = "all"

StatusFn = Callable[[str], None]
ProgressFn = Callable[[float], None]


def _ignore_status(_message: str) -> None: ...
def _ignore_progress(_fraction: float) -> None: ...


def manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "release-manifest.json"


def load_components() -> dict[str, Component]:
    """Every pinned component, each already validated for HTTPS and SHA-256."""
    return load_manifest(manifest_path())


def component_ids(gpu: GpuInfo, quantization: str) -> tuple[str, ...]:
    """The artifacts one install needs, in download order.

    Four for a GPU: the llama.cpp program archive and its CUDA runtime DLLs are
    separate releases upstream and are pinned separately here, and they are
    combined into a single runtime directory during installation. Three for the
    processor, whose archive carries everything it needs and has no CUDA DLLs to
    be paired with.
    """
    runtime = runtime_component_id(gpu)
    model = (f"model-{quantization}", "mmproj")
    if gpu.is_cpu:
        return (runtime, *model)
    return (runtime, f"{runtime}-cudart", *model)


def resolve(gpu: GpuInfo, quantization: str) -> list[Component]:
    """Look each component up, failing before any network access."""
    components = load_components()
    ids = component_ids(gpu, quantization)
    missing = [key for key in ids if key not in components]
    if missing:
        raise RuntimeError(
            f"Release manifest has no complete {quantization} component set "
            f"(missing {', '.join(missing)})"
        )
    return [components[key] for key in ids]


def identify(sha256: str) -> str | None:
    """The component a file's SHA-256 belongs to, if the manifest pins one.

    What turns "that file is not the Q6_K_P build" into "that file is the
    Q4_K_M build", which is the difference between a dead end and an answer.
    """
    for key, component in load_components().items():
        if component.sha256.casefold() == sha256.casefold(): return key
    return None


def download_estimate(gpu: GpuInfo, quantization: str, skip: Collection[str] = ()) -> tuple[int, bool]:
    """``(bytes, exact)`` for one install's downloads, ignoring ``skip``.

    Components supplied from a local file are skipped: they are installed from
    disk, so counting them would quote a download that is not going to happen.

    ``size`` is optional in the manifest because some publishers do not report
    one — the llama.cpp release archives are exactly that case, so a total that
    insisted on every size would never be available to show. The known sizes are
    summed instead and ``exact`` says whether anything was missing, which lets
    the caller say "about 17 GiB" versus "at least 17 GiB" rather than nothing
    at all. The SHA-256 is mandatory either way, so a missing size costs
    precision in a progress message and nothing more.
    """
    sizes = [component.size for key, component in zip(component_ids(gpu, quantization),
                                                      resolve(gpu, quantization)) if key not in skip]
    return sum(size for size in sizes if size is not None), all(size is not None for size in sizes)


def format_download_size(gpu: GpuInfo, quantization: str, skip: Collection[str] = ()) -> str:
    known, exact = download_estimate(gpu, quantization, skip)
    if not known: return "nothing" if exact else "the runtime archives only"
    return f"{'about' if exact else 'at least'} {known / 2 ** 30:.1f} GiB"


@dataclass(frozen=True)
class Installed:
    """Where setup put things, relative to the install root."""

    runtime: str
    model: str
    mmproj: str


def fetch(paths: AppPaths, gpu: GpuInfo, quantization: str, *,
          sources: Mapping[str, LocalSource] | None = None,
          on_status: StatusFn = _ignore_status,
          on_progress: ProgressFn = _ignore_progress) -> Installed:
    """Download, verify and extract everything. No state is written here.

    A component named in ``sources`` is taken from the file the caller supplies
    instead of downloaded — same verification, same destination, same result.
    """
    components = resolve(gpu, quantization)
    ids = component_ids(gpu, quantization)
    supplied = dict(sources or {})
    unused = set(supplied) - set(ids)
    if unused:
        # A file supplied for a component this install does not use — a Q4_K_M
        # model while installing Q6_K_P — would otherwise be silently ignored
        # and quietly downloaded instead.
        raise RuntimeError(f"This install uses no {', '.join(sorted(unused))}; "
                           f"it installs {', '.join(ids)}")
    paths.create_managed_dirs()

    runtime_archives: list[Path] = []
    model = mmproj = ""
    share = 1.0 / (len(ids) + 1)  # the extract step is the final share
    for number, (key, component) in enumerate(zip(ids, components)):
        target = paths.contained(component.destination)
        report = (lambda done, total, n=number: on_progress(share * (n + done / max(total, 1))))
        source = supplied.get(key)
        if source is not None:
            on_status(f"Installing {key} from {source.path.name}…")
            artifact = adopt(component, target, source, report)
        else:
            on_status(f"Downloading {key}…")
            artifact = download(component, target, report,
                                lambda text, k=key: on_status(f"{k}: {text}"))
        if key.startswith("llama-runtime-"):
            runtime_archives.append(artifact)
        elif key.startswith("model-"):
            model = artifact.relative_to(paths.root).as_posix()
        else:
            mmproj = artifact.relative_to(paths.root).as_posix()

    on_status("Extracting runtime…")
    on_progress(share * len(ids))
    runtime_dir = paths.root / "runtime"
    extract_zips_atomic(runtime_archives, runtime_dir)
    executable = _find_server(runtime_dir)
    on_progress(1.0)
    return Installed(executable.relative_to(paths.root).as_posix(), model, mmproj)


def _find_server(runtime_dir: Path) -> Path:
    for name in ("llama-server.exe", "llama-server"):
        matches = sorted(runtime_dir.rglob(name))
        if matches:
            return matches[0]
    raise RuntimeError("Combined runtime archives contain no llama-server executable")


def write_state(paths: AppPaths, gpu: GpuInfo, quantization: str, installed: Installed, *,
                context_size: int = DEFAULT_CONTEXT_SIZE,
                gpu_layers: str = FULL_OFFLOAD) -> dict:
    """Record the validated-so-far install, atomically, via a pending file.

    ``gpu_layers`` is the caller's only in GPU mode. The other two modes are
    defined by there being no resident layers at all, so recording a count
    beside them would be fiction and it is replaced here.
    """
    if gpu.is_cpu:
        # Nothing to ask llama.cpp about either: --device none is the whole
        # answer, and there is no CUDA device for the probe to name.
        device, device_name, gpu_layers = CPU_DEVICE, gpu.name, NO_OFFLOAD
    else:
        device, device_name = list_llama_devices(paths.contained(installed.runtime),
                                                 gpu.physical_index)
        if gpu.is_mixed:
            gpu_layers = NO_OFFLOAD
    state = {
        "runtime": installed.runtime,
        "model": installed.model,
        "mmproj": installed.mmproj,
        "mode": gpu.mode,
        "gpu_index": gpu.physical_index,
        "gpu_uuid": gpu.uuid,
        "gpu_name": gpu.name,
        "gpu_device": device,
        "gpu_device_name": device_name,
        "quantization": quantization,
        "context_size": int(context_size),
        "gpu_layers": str(gpu_layers),
    }
    pending = paths.data / "setup-state.pending.json"
    atomic_write_json(pending, state)
    shutil.copy2(pending, paths.state_file)
    pending.unlink(missing_ok=True)
    return state


def validate(paths: AppPaths, *, on_status: StatusFn = _ignore_status) -> None:
    """Prove the install generates, over both modalities, or remove the state.

    Both probes are required. A text-only pass would let an install whose vision
    projector is broken reach the main window and fail on the first attached
    image, which is the whole feature the projector is downloaded for.
    """
    from PIL import Image

    from prompt_master.inference.service import InferenceService
    from prompt_master.prompt_engine.adapter import PromptEngine

    service = InferenceService(paths)
    try:
        on_status("Starting llama-server…")
        client = service.client()
        engine = PromptEngine()
        intent = "A red ball rolls across a wooden table"

        on_status("Validating text generation…")
        text_probe = PromptRequest(intent, video_mode="t2v", smart_negative=False)
        if not client.stream_chat(engine.build(text_probe).messages, 64, 1, lambda _: None).strip():
            raise RuntimeError("Text validation returned no content")

        on_status("Validating image generation…")
        probe = paths.cache / "temp-images" / "setup-probe.jpg"
        probe.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (220, 30, 30)).save(probe)
        image_probe = PromptRequest(intent, video_mode="i2v", image_data_url=image_data_url(probe),
                                    image_name=probe.name, smart_negative=False)
        if not client.stream_chat(engine.build(image_probe).messages, 64, 1, lambda _: None).strip():
            raise RuntimeError("Image validation returned no content")
    except Exception:
        paths.state_file.unlink(missing_ok=True)
        raise
    finally:
        service.stop()
        (paths.data / "setup-state.pending.json").unlink(missing_ok=True)


def provision(paths: AppPaths, gpu: GpuInfo, quantization: str, *,
              sources: Mapping[str, LocalSource] | None = None,
              context_size: int = DEFAULT_CONTEXT_SIZE,
              gpu_layers: str = FULL_OFFLOAD,
              on_status: StatusFn = _ignore_status,
              on_progress: ProgressFn = _ignore_progress) -> dict:
    """Full setup: fetch, record, validate, and point future launches here."""
    installed = fetch(paths, gpu, quantization, sources=sources,
                      on_status=on_status, on_progress=on_progress)
    state = write_state(paths, gpu, quantization, installed,
                        context_size=context_size, gpu_layers=gpu_layers)
    validate(paths, on_status=on_status)
    AppPaths.record(paths.root)
    on_status("Runtime, text inference, and image inference validated.")
    return state
