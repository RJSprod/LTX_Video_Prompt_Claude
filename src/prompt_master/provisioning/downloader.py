from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os
import httpx

from .manifest import Component
from .verifier import verify


def download(component: Component, destination: Path, progress: Callable[[int, int], None] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True); partial = destination.with_name(destination.name + ".part")
    if destination.is_file():
        try:
            verify(destination, component.size, component.sha256)
            if progress: progress(component.size or destination.stat().st_size, component.size or destination.stat().st_size)
            return destination
        except (OSError, ValueError):
            destination.unlink(missing_ok=True)
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(600, connect=30)) as client:
        with client.stream("GET", component.url, headers=headers) as response:
            response.raise_for_status()
            append = existing > 0 and response.status_code == 206
            if existing and not append: existing = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as stream:
                done = existing
                for chunk in response.iter_bytes(1024 * 1024):
                    stream.write(chunk); done += len(chunk)
                    if progress: progress(done, component.size or max(done, 1))
                stream.flush(); os.fsync(stream.fileno())
    verify(partial, component.size, component.sha256); os.replace(partial, destination); return destination
