# Prompt Master Standalone — Python edition

Turn a one-line idea into a full LTX-Video 2.3 prompt, positive and negative,
generated locally by a Gemma model running in `llama.cpp`. Attach a reference
image and the same model looks at it while it writes.

This is [`RJSprod/LTX_Video_Prompts`](https://github.com/RJSprod/LTX_Video_Prompts)
distributed as Python source with a one-click installer, instead of as a
compiled `.exe` behind an Inno Setup wizard. The application, the prompt engine
and the prompts it produces are unchanged — see [Relationship to the packaged
build](#relationship-to-the-packaged-build).

## Install

Download or clone this repository, then run:

```
start_windows.bat
```

That is the whole install. It will:

1. find a Python 3.12+ interpreter, or install a private one under
   `installer_files\conda` if the machine has none — verified against a pinned
   SHA-256, and without touching your PATH, registry or system Python;
2. build an isolated environment in `installer_files\env` and install the five
   dependencies;
3. ask you three questions (below) and download the model;
4. launch the application.

Re-running it is always safe — every step it has already done is skipped.

### The three questions

| Question | What it decides |
| --- | --- |
| **Installation directory** | Where the runtime, model, projector, cache and logs live. The model alone is 16–27 GiB, so this can be a different drive from the checkout. |
| **Which GPU** | Which card runs the model, and which pinned `llama.cpp` build gets downloaded — CUDA 12 or CUDA 13, chosen by compute capability. Skipped if you only have one. |
| **Model quality** | `Q4_K_M`, `Q6_K_P` or `Q8_K_P`. The default is sized from your VRAM, and each option shows its download size and whether it fits. |

Nothing is downloaded before you confirm. Every artifact is pinned by URL and
verified by SHA-256 after download; interrupted downloads resume rather than
restart. Setup only writes its state after it has generated one text prompt and
one image prompt successfully, so a half-finished install cannot be launched.

## Launch

```
python app.py
```

That is the command-line launch this repository exists to provide. It works from
any command prompt, from any directory, with whatever `python` is on your PATH —
`app.py` re-executes itself under the installed environment when it needs to, so
there is nothing to activate first.

| Command | Effect |
| --- | --- |
| `python app.py` | Open the application |
| `python app.py --setup` | Re-run the three questions — change GPU, model or directory |
| `python app.py --version` | Print the version |
| `cmd_windows.bat` | Open a shell with the environment active, for `pytest` |
| `update_windows.bat` | Reinstall dependencies after pulling a new version |

For an unattended reinstall, every question has a flag:

```
python app.py --setup --dir D:\PromptMaster --gpu 0 --quant Q6_K_P --yes
```

`--gpu-layers` is there too, for a card that cannot hold its quantization
entirely in VRAM.

## Using it

Type what you want in **Intent**, set whichever controls matter, and press
**Generate**. The positive prompt streams in as the model writes it; the
negative prompt is assembled from the engine's gated banks and, if **Smart
negative** is on, a second pass over the finished script.

**Attach an image** with *Browse image…* and the mode switches to image-to-video
automatically. The image is EXIF-normalised, converted to RGB, resized to 768px,
JPEG-encoded and sent in the same multimodal request as the text — the model
sees the still it is writing the first frame from. Image-to-video without an
image is refused rather than silently downgraded, because the two produce
different prompts.

Every control from the packaged build is present: 47 accents with 3 strengths,
35 music genres plus accent-matched auto, 20 visual styles in 4 groups, 10
cameras, 10 transitions, 3 output formats, POV, wardrobe and undress, a lexicon,
dialogue percentage, duration, FPS, dimensions, seed and extra negative terms.
Results copy to the clipboard or save to `.txt`.

## Requirements

- Windows x64
- An NVIDIA GPU with a current driver. The 3090 and 5090 are the pinned
  configurations; any other CUDA card is sized from its own VRAM and compute
  capability. There is no CPU path — inference runs on the GPU through
  `llama.cpp`.
- Roughly 20–30 GiB of disk for the model, plus ~1 GiB for the environment.
- Python 3.12+ is used if present, and installed privately if not.

## Relationship to the packaged build

The prompt engine is a vendored copy of
[Prompt Master LD](https://github.com/Brojakhoeman/Prompt-Master-LD), and this
repository does not touch it: `src/prompt_master/prompt_engine/` is
byte-identical to the packaged build, which is itself byte-identical to upstream
in 14 of 15 modules. The one approved difference is recorded in
[`UPSTREAM_DIFF_NOTES.md`](UPSTREAM_DIFF_NOTES.md) and the numbers are in
[`PARITY_REPORT.md`](PARITY_REPORT.md).

`prompt_engine/options.py` still builds every dropdown from the upstream
constants at import time, so no control can drift from the engine consuming its
value.

What changed, and only this:

| Area | Packaged build | Here |
| --- | --- | --- |
| Distribution | Nuitka `.exe` + Inno Setup | Python source + `start_windows.bat` |
| Launch | `PromptMaster.exe` | `python app.py` |
| Setup questions | Qt wizard only | Console at install, Qt wizard still in Settings — both share one pipeline |
| Install root | Beside the frozen `.exe` | `install.json` beside `app.py`, so models can live on another drive |
| GPU support | RTX 3090 / 5090 only, others refused | Any NVIDIA card; 3090 and 5090 keep their exact pinned runtime and quantization |

The GPU widening is the one behavioural change. It is pinned so that it cannot
affect the two supported cards: `device_detection.PINNED` maps them to their
original runtime and quantization before any heuristic runs, and
`tests/test_install_flow.py` asserts that.

## Tests

```
python -m pytest tests/        # 510 passed
```

- `test_upstream_parity.py` (434) — the upstream self-test, ported
- `test_prompt_engine.py` (26) — the adapter seam and the UI option sources
- `test_core.py` (6) — multimodal requests, atomic JSON, SSE, zip-slip
- `test_install_flow.py` (44) — install-root discovery, GPU sizing, manifest
  resolution, console setup

Output-level parity against an upstream checkout is checked separately:

```
python tools/compare_upstream_engine.py --upstream <checkout>   # 0 mismatches
python tools/check_upstream_sync.py     --upstream <checkout>   # 1 approved diff
```

## Layout

```
start_windows.bat          One-click installer and launcher
one_click.py               Installer logic: interpreter, environment, setup, launch
app.py                     Command-line entry point
requirements.txt           The five runtime dependencies
src/prompt_master/
  app.py                   Argument handling and window startup
  setup_cli.py             Console setup — the three questions
  prompt_engine/           Vendored upstream engine, untouched
  provisioning/installer.py  Download, verify, extract, validate — one pipeline
  inference/               llama-server process, streaming client, GPU detection
  ui/                      Main window and the Qt setup wizard
installer_files/           Created by the installer (gitignored)
user_data/                 Default install root (gitignored)
```
