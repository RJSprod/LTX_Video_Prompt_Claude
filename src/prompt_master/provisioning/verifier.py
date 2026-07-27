import hashlib
from pathlib import Path


def verify(path: Path, size: int | None, sha256: str) -> None:
    if size is not None and path.stat().st_size != size: raise ValueError(f"Size mismatch for {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(block)
    if digest.hexdigest().casefold() != sha256.casefold(): raise ValueError(f"SHA-256 mismatch for {path.name}")
