import hashlib, io, json, zipfile
from pathlib import Path
import httpx
import pytest
from PIL import Image
from prompt_master.core.config import atomic_write_json, read_json
from prompt_master.core.models import PromptRequest
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.inference.sse import assistant_chunks
from prompt_master.inference.device_detection import runtime_component_id
from prompt_master.core.models import GpuInfo
from prompt_master.prompt_engine.adapter import PromptEngine
from prompt_master.provisioning import downloader
from prompt_master.provisioning.extractor import extract_zip_atomic, extract_zips_atomic
from prompt_master.provisioning.manifest import Component


def test_multimodal_and_negative_dedup(tmp_path):
    image=tmp_path/"x.png"; Image.new("RGB",(1000,500),"red").save(image); url=image_data_url(image)
    request=PromptRequest("A runner",image_data_url=url,video_mode="i2v",negative_extra="logo, custom")
    content=PromptEngine().build(request).messages[1]["content"]
    image_part=next(part for part in content if part["type"]=="image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # "logo" is already in upstream's core bank; upstream dedupe keeps one.
    negative=PromptEngine().base_negative(request); assert negative.lower().count("logo")==1 and "custom" in negative


def test_atomic_json(tmp_path):
    path=tmp_path/"data"/"settings.json"; atomic_write_json(path,{"unicode":"雪"}); assert read_json(path)=={"unicode":"雪"}


def test_sse_ignores_reasoning():
    lines=['data: {"choices":[{"delta":{"reasoning_content":"secret","content":"hello"}}]}','data: [DONE]']
    assert list(assistant_chunks(lines))==["hello"]


def test_zip_slip_rejected(tmp_path):
    archive=tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as z: z.writestr("../escape",b"bad")
    with pytest.raises(ValueError): extract_zip_atomic(archive,tmp_path/"out")


def test_related_runtime_archives_are_merged(tmp_path):
    first=tmp_path/"program.zip"; second=tmp_path/"dlls.zip"
    with zipfile.ZipFile(first,"w") as z: z.writestr("llama-server.exe",b"exe")
    with zipfile.ZipFile(second,"w") as z: z.writestr("cudart64.dll",b"dll")
    extract_zips_atomic([first,second],tmp_path/"runtime")
    assert (tmp_path/"runtime"/"llama-server.exe").read_bytes()==b"exe"
    assert (tmp_path/"runtime"/"cudart64.dll").read_bytes()==b"dll"


# ── downloads that survive a long transfer ───────────────────────────────────
#
# The model is one 16-27 GiB file. A connection that drops 0.8% in used to end
# setup with "peer closed connection without sending complete message body",
# which is what these cover.

PAYLOAD = b"GGUF" * 64


def part_of(target): return target.with_name(target.name + ".part")


def component(payload=PAYLOAD, size=True, sha256=None):
    return Component("model-Q6_K_P", "https://example.invalid/model.gguf", "models/model.gguf",
                     len(payload) if size else None,
                     sha256 or hashlib.sha256(payload).hexdigest(), "pinned")


class FakeCDN:
    """A range-aware server that can drop the connection mid-body."""

    def __init__(self, payload=PAYLOAD, drop_after=(), status=None, ignore_range=False):
        self.payload, self.drops, self.status = payload, list(drop_after), list(status or [])
        self.ignore_range, self.requests = ignore_range, []

    def transport(self): return httpx.MockTransport(self.handle)

    def client(self): return httpx.Client(transport=self.transport())

    def handle(self, request):
        self.requests.append(request)
        if self.status:
            code = self.status.pop(0)
            if code >= 400: return httpx.Response(code)
        start = 0
        if "Range" in request.headers and not self.ignore_range:
            start = int(request.headers["Range"].removeprefix("bytes=").split("-")[0])
        body, drop = self.payload[start:], self.drops.pop(0) if self.drops else None

        def stream():
            sent = 0
            for index in range(0, len(body), 8):
                if drop is not None and sent >= drop:
                    raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
                yield body[index:index + 8]; sent += 8

        return httpx.Response(206 if start else 200, content=stream())


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    """Bytes reach the disk a block at a time, so the block has to be smaller
    than these payloads for a drop to leave anything to resume from."""
    monkeypatch.setattr(downloader, "CHUNK_BYTES", 8)


@pytest.fixture
def instant_retries(monkeypatch):
    monkeypatch.setattr(downloader, "BACKOFF_SECONDS", (0,) * len(downloader.BACKOFF_SECONDS))


def serve(monkeypatch, cdn):
    monkeypatch.setattr(downloader, "_client", cdn.client)
    return cdn


def test_a_dropped_connection_resumes_where_it_stopped(tmp_path, monkeypatch, instant_retries):
    cdn = serve(monkeypatch, FakeCDN(drop_after=[40]))
    target = tmp_path / "models" / "model.gguf"
    notices = []

    assert downloader.download(component(), target, notice=notices.append) == target
    assert target.read_bytes() == PAYLOAD
    # Resumed with a range request rather than starting the 21 GiB over again.
    assert cdn.requests[1].headers["Range"] == "bytes=40-"
    assert not part_of(target).exists()
    # The console says why it paused, so a backoff does not look like a hang.
    assert notices and "retrying" in notices[0]


def test_retrying_stops_once_nothing_is_getting_through(tmp_path, monkeypatch, instant_retries):
    """A budget spent only on attempts that move no bytes: five of those and it
    gives up, rather than hammering a connection that is plainly down."""
    cdn = serve(monkeypatch, FakeCDN(drop_after=[0] * 20))
    with pytest.raises(httpx.RemoteProtocolError):
        downloader.download(component(), tmp_path / "models" / "model.gguf")
    assert len(cdn.requests) == downloader.RETRIES_WITHOUT_PROGRESS + 1


def test_every_attempt_that_moves_bytes_earns_a_fresh_budget(tmp_path, monkeypatch, instant_retries):
    """Eight drops is more than the budget; because each one gets further, the
    download still finishes."""
    cdn = serve(monkeypatch, FakeCDN(drop_after=[8, 16, 24, 32, 40, 48, 56, 64]))
    target = tmp_path / "models" / "model.gguf"
    assert downloader.download(component(), target) == target
    assert target.read_bytes() == PAYLOAD


def test_a_server_that_ignores_the_range_restarts_cleanly(tmp_path, monkeypatch, instant_retries):
    cdn = serve(monkeypatch, FakeCDN(drop_after=[40], ignore_range=True))
    target = tmp_path / "models" / "model.gguf"
    assert downloader.download(component(), target) == target
    assert target.read_bytes() == PAYLOAD          # not 40 stale bytes plus a whole body


def test_a_part_file_at_full_size_is_started_over_rather_than_resumed(tmp_path, monkeypatch):
    """Resuming past the end of a file is a 416, and a stale part file would
    make every future run repeat it."""
    target = tmp_path / "models" / "model.gguf"; target.parent.mkdir(parents=True)
    part_of(target).write_bytes(b"x" * len(PAYLOAD))
    cdn = serve(monkeypatch, FakeCDN())

    assert downloader.download(component(), target) == target
    assert target.read_bytes() == PAYLOAD
    assert "Range" not in cdn.requests[0].headers


def test_a_range_the_server_rejects_starts_the_file_over(tmp_path, monkeypatch, instant_retries):
    """The runtime archives are pinned without a size, so a part file longer
    than the file itself is only discovered as a 416 — which every later resume
    would repeat."""
    target = tmp_path / "cache" / "runtime.zip"; target.parent.mkdir(parents=True)
    part_of(target).write_bytes(b"x" * (len(PAYLOAD) * 2))
    cdn = serve(monkeypatch, FakeCDN(status=[416]))

    assert downloader.download(component(size=False), target) == target
    assert target.read_bytes() == PAYLOAD
    assert len(cdn.requests) == 2 and "Range" not in cdn.requests[1].headers


def test_bytes_that_fail_verification_are_dropped_not_kept(tmp_path, monkeypatch):
    serve(monkeypatch, FakeCDN())
    target = tmp_path / "models" / "model.gguf"
    with pytest.raises(ValueError):
        downloader.download(component(sha256="0" * 64), target)
    assert not part_of(target).exists()


def test_a_server_error_is_retried_and_a_missing_file_is_not(tmp_path, monkeypatch, instant_retries):
    cdn = serve(monkeypatch, FakeCDN(status=[503, 502]))
    target = tmp_path / "models" / "model.gguf"
    assert downloader.download(component(), target) == target
    assert len(cdn.requests) == 3

    cdn = serve(monkeypatch, FakeCDN(status=[404]))
    with pytest.raises(httpx.HTTPStatusError):
        downloader.download(component(), tmp_path / "models" / "other.gguf")
    assert len(cdn.requests) == 1


def test_an_already_downloaded_file_is_not_downloaded_again(tmp_path, monkeypatch):
    cdn = serve(monkeypatch, FakeCDN())
    target = tmp_path / "models" / "model.gguf"; target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    assert downloader.download(component(), target) == target
    assert cdn.requests == []


def test_gpu_runtime_mapping_is_generation_specific():
    gpu=lambda name: GpuInfo(0,"uuid",name,24576,20000,"1")
    assert runtime_component_id(gpu("NVIDIA GeForce RTX 3090")) == "llama-runtime-cuda12"
    assert runtime_component_id(gpu("NVIDIA GeForce RTX 5090")) == "llama-runtime-cuda13"
