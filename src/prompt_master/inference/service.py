from __future__ import annotations

from pathlib import Path

from prompt_master.core.config import read_json
from prompt_master.core.paths import AppPaths
from .llama_client import LlamaClient
from .llama_process import LlamaProcess


class InferenceService:
    """Owns the single managed llama-server process for the application."""

    def __init__(self, paths: AppPaths):
        self.paths = paths
        self.process = LlamaProcess()
        self.signature: tuple | None = None

    def client(self, needs_vision: bool = False) -> LlamaClient:
        state = read_json(self.paths.state_file)
        required = ("runtime", "model", "mmproj", "gpu_index")
        missing = [key for key in required if key not in state]
        if missing:
            raise RuntimeError("Setup is incomplete (missing " + ", ".join(missing) + "). Run `python app.py --setup`, or open Models and Hardware setup.")
        runtime, model, mmproj = (self.paths.contained(state[key]) for key in required[:3])
        for label, path in (("llama-server", runtime), ("model", model), ("vision projector", mmproj)):
            if not path.is_file():
                raise RuntimeError(f"Configured {label} is missing: {path}")
        if needs_vision and not mmproj.is_file():
            raise RuntimeError("Image generation requires the configured vision projector; text-only fallback is disabled.")
        signature = (runtime, model, mmproj, int(state["gpu_index"]), state.get("gpu_device", "CUDA0"), int(state.get("context_size", 8192)), str(state.get("gpu_layers", "all")))
        if not self.process.running or signature != self.signature:
            self.process.start(runtime, model, mmproj, signature[3], signature[4], signature[5], self.paths.logs / "llama-server.log", gpu_layers=signature[6])
            self.process.wait_ready()
            self.signature = signature
        return LlamaClient(f"http://127.0.0.1:{self.process.port}", self.process.api_key)

    def stop(self) -> None:
        self.process.stop()
        self.signature = None
