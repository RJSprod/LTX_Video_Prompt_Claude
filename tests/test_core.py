import io, json, zipfile
from pathlib import Path
import pytest
from PIL import Image
from prompt_master.core.config import atomic_write_json, read_json
from prompt_master.core.models import PromptRequest
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.inference.sse import assistant_chunks
from prompt_master.inference.device_detection import runtime_component_id
from prompt_master.core.models import GpuInfo
from prompt_master.prompt_engine.adapter import PromptEngine
from prompt_master.provisioning.extractor import extract_zip_atomic, extract_zips_atomic


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


def test_gpu_runtime_mapping_is_generation_specific():
    gpu=lambda name: GpuInfo(0,"uuid",name,24576,20000,"1")
    assert runtime_component_id(gpu("NVIDIA GeForce RTX 3090")) == "llama-runtime-cuda12"
    assert runtime_component_id(gpu("NVIDIA GeForce RTX 5090")) == "llama-runtime-cuda13"
