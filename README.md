# Prompt Master Standalone — Python edition

Turn a one-line idea into a full LTX-Video 2.3 prompt, positive and negative,
generated locally by a Gemma model running in `llama.cpp`. Attach a reference
image and the same model looks at it while it writes.

This is [`RJSprod/LTX_Video_Prompts`](https://github.com/RJSprod/LTX_Video_Prompts)
distributed as Python source with a one-click installer, instead of as a
compiled `.exe` behind an Inno Setup wizard. The application, the prompt engine
and the prompts it produces are unchanged — every setting added here opens on
the behaviour the packaged build had, and says so where it does not. See
[Relationship to the packaged build](#relationship-to-the-packaged-build).

## Install

Download or clone this repository, then run:

```
start_windows.bat
```

That is the whole install. It will:

1. find a Python 3.12 or 3.13 interpreter, or install a private one under
   `installer_files\conda` if the machine has none — verified against a pinned
   SHA-256, and without touching your PATH, registry or system Python;
2. build an isolated environment in `installer_files\env` and install the five
   dependencies;
3. ask you four questions (below) and download the model;
4. launch the application.

Re-running it is always safe — every step it has already done is skipped.

### The four questions

| Question | What it decides |
| --- | --- |
| **Installation directory** | Where the runtime, model, projector, cache and logs live. The model alone is 16–27 GiB, so this can be a different drive from the checkout. |
| **What runs the model** | Every CUDA card on the machine, and your processor. A card decides which pinned `llama.cpp` build gets downloaded — CUDA 12 or CUDA 13, chosen by compute capability; the processor gets the CPU build. |
| **Model quality** | `Q4_K_M`, `Q6_K_P` or `Q8_K_P`. On a card the default is sized from your VRAM and each option shows its download size and whether it fits; on the processor the default is `Q4_K_M` and each option shows its download size. |
| **A model you already have** | Optional. Point setup at a `.gguf` already on disk and it is installed from there instead of downloaded — see below. Answer no and everything downloads as before. |

### Using a model you already have

The model is one 16–27 GiB file, and a connection that cannot carry it is not
something setup can fix. If you already have that file — downloaded by hand,
copied from another install, or fetched with a download manager — hand it over:

```
python app.py --setup --model-file D:\models\Gemma4-...-Q6_K_P.gguf
```

or answer the fourth question and paste the path. Either way it is checked
against the same pinned SHA-256 as a download, then **moved** into the
installation directory rather than copied, so you are not left with two copies
of a 21 GiB file. `--keep-source` copies instead. If the vision projector sits
beside it — it usually does, both being files from the same repository — setup
offers to take that too, and anything not supplied is still downloaded normally.

A file that is not the pinned build is named rather than just rejected: "that is
the `Q4_K_M` build" is an answer, and setup offers to install it as what it is.
The same picker is on the Qt wizard's model page, for re-running setup later.

Nothing is downloaded before you confirm. Every artifact is pinned by URL and
verified by SHA-256 after download; interrupted downloads resume rather than
restart. A connection dropped mid-transfer — which a 16–27 GiB single file
invites — is retried from where it stopped rather than ending setup, and only
attempts that move no bytes at all count against the retry budget. Setup only
writes its state after it has generated one text prompt and one image prompt
successfully, so a half-finished install cannot be launched.

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
| `python app.py --setup` | Re-run the questions — change device, model or directory |
| `python app.py --version` | Print the version |
| `cmd_windows.bat` | Open a shell with the environment active, for `pytest` |
| `update_windows.bat` | Reinstall dependencies after pulling a new version |

For an unattended reinstall, every question has a flag:

```
python app.py --setup --dir D:\PromptMaster --gpu 0 --quant Q6_K_P --yes
python app.py --setup --cpu --quant Q4_K_M --yes
```

`--model-file` and `--mmproj-file` install those from disk instead of
downloading them; a file that is not the pinned artifact stops an unattended run
rather than becoming an install that claims to be something it is not.
`--gpu-layers` is there too, for a card that cannot hold its quantization
entirely in VRAM. `--cpu` answers the device question with the processor
instead of a card, and does not need `nvidia-smi` to do it.

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

**Seed** takes `-1` for "a different one every generation" — step below zero to
reach it. A drawn seed is fixed before the brief is built, because the engine
seeds its casting and wardrobe with the same number llama.cpp seeds its sampler
with, and the seed that was actually used is named in the status line when the
generation finishes, so a shot worth keeping can be reproduced.

**Motion** is three settings for how the shot moves:

| Setting | What it writes |
| --- | --- |
| **Default** | Exactly what the engine wrote before this control existed — no directive, no added terms, upstream's own sampling temperatures. |
| **Inertia** | Weight and abruptness: hard starts and stops, angles that cut rather than ease, momentum that reads as impact. Runs hotter and narrower, and puts floaty drift in the negative. |
| **Flow** | Continuity and detail: one unbroken camera gesture, every action carrying from beat to beat, the words spent on how motion travels rather than on more events. Runs cooler and steadier, and puts stutter and snap cuts in the negative. |

**Extra speech** is a slider from 1× to 10×. At the far left — the default —
the intent goes to the engine exactly as you typed it. Above that, a pass runs
over the intent *before* the brief is built: it writes extra lines in the voice
the intent already quotes and mixes them back in, so what the engine receives is
a director's request that simply had more speech in it. 10× means ten times the
lines the intent quoted, up to a ceiling of 30.

Two things worth knowing about it:

- The **dialogue budget moves with the slider.** Upstream tells the model how
  many spoken lines a shot of this length should carry — at 20% that is about
  two — so twenty extra lines against that budget would be eighteen lines the
  model is told to drop. The slider lifts the budget from your own setting
  toward 100%, never below it, and only for that generation; the dial itself is
  untouched. The line under the slider names the figure it will use.
- An intent with **no quoted speech** gets lines written for the scene instead,
  from the people and mood the brief describes.

The pass never costs you a generation: if it fails or comes back empty, the shot
is written from your intent as typed and the status says which happened.

Each motion preset pulls three levers and only three: a directive appended *after* the
engine's finished system prompt, terms merged into the extra negatives the
engine already accepts as an input, and the sampling temperature of the writer
pass. The smart-negative pass keeps upstream's own numbers. Default appends
nothing and adds nothing, so a default generation is byte-for-byte the
generation this application produced before the presets existed — which
`tests/test_prompt_engine.py` asserts against `brain.build_system` directly.

### On a touch screen

The window is laid out for a finger. Intent and the finished prompts sit side by
side, so writing one and reading the other never scroll each other off the
screen; the controls between them are grouped — shot, look, voice and music,
wording — and scroll on their own; and the bar along the bottom holding the
status, Clear, Cancel and Generate never scrolls at all.

- **Drag to scroll.** The settings, both prompt boxes, the intent box and the
  long drop-downs all take a kinetic flick. Mouse drags still select text: only
  the touch gesture scrolls, so nothing a desktop user does changes. The scroll
  bars are still there, and are wide enough to be worth aiming at.
- **Nothing smaller than a fingertip.** Every button, drop-down, check box and
  drop-down row is at least 48 logical pixels tall, and spin boxes get a full
  height `−` and `+` either side of the value — hold either one to run it up —
  in place of Qt's two stacked arrows, which are half that.
- **View → Display size** switches between Comfortable, Large and Larger. It
  restyles the window live, applies to the setup wizard too, and is remembered.

The window opens on most of the screen rather than at a fixed size, and the
divider between the two panes can be dragged to give either one more room.

## Requirements

- Windows x64
- An NVIDIA GPU with a current driver, or an x64 processor. The 3090 and 5090
  are the pinned configurations; any other CUDA card is sized from its own VRAM
  and compute capability. Setup also offers your processor, which installs the
  CPU build of `llama.cpp` and keeps the model in system RAM — no NVIDIA card
  or driver is used, and none is required.
- Roughly 20–30 GiB of disk for the model, plus ~1 GiB for the environment.
- Python 3.12 or 3.13 is used if present, and installed privately if not. A
  newer Python on the machine is skipped rather than used: the pinned
  dependencies publish no wheels for it — PySide6 6.8.1 stops at 3.13 — so the
  installer builds its environment on a version they cover instead. An
  environment left behind by an earlier run on an unsupported Python is
  rebuilt automatically the next time `start_windows.bat` runs.

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
| Running without a card | Not possible | The processor and system RAM are an option in both front ends, on the pinned CPU build of `llama.cpp` |
| Where the model comes from | Downloaded, always | Downloaded, or installed from a `.gguf` you already have — against the same pinned SHA-256 |
| Window | One column of mouse-sized controls | Two panes, fingertip-sized targets, drag-to-scroll, a display-size setting |
| Motion | One way of writing it | Default is that way exactly; **Inertia** and **Flow** are opt-in |
| Seed | A number | A number, or `-1` for a fresh one each generation |
| Speech | Whatever the intent quotes | That, or up to 10× more written in the same voice — opt-in |

Four behavioural changes, all bounded, and the two that can reach a prompt are
both opt-in and both open on the old behaviour. The GPU widening is pinned so
that it cannot affect the two supported cards: `device_detection.PINNED` maps
them to their original runtime and quantization before any heuristic runs, and
`tests/test_install_flow.py` asserts that. The CPU option is the same kind of
widening — a device the setup questions now offer, reached only by choosing it,
resolving to its own pinned `llama.cpp` archive and leaving every card's
download list untouched. The motion presets are additive —
Default appends nothing, adds no terms and samples at upstream's own
temperatures, so an untouched control is an untouched prompt, and
`prompt_engine/motion.py` holds the whole of the difference the other two make.
Extra speech is a pass over the intent that does not run at 1×, and the engine
still only ever reads an intent: `prompt_engine/speech.py` writes the lines and
mixes them in before the brief is built, and changes nothing about how the brief
is then built from it. Supplying a model changes where the bytes come from and
nothing else — the file is checked against the same pinned hash a download is,
and lands at the same path, so everything downstream of setup cannot tell the
two apart.

## Tests

```
python -m pytest tests/        # 588 passed
```

- `test_upstream_parity.py` (434) — the upstream self-test, ported
- `test_prompt_engine.py` (44) — the adapter seam, the motion presets, speech
  expansion and the UI option sources
- `test_touch_ui.py` (16) — target sizes, drag-to-scroll, the sliders and the
  display size, measured on a real window built offscreen
- `test_core.py` (15) — multimodal requests, atomic JSON, SSE, zip-slip,
  download resume and retry
- `test_install_flow.py` (93) — install-root discovery, GPU sizing, the CPU
  device, manifest resolution, console setup, supplying a model from disk, and
  the installer's interpreter and environment checks

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
  setup_cli.py             Console setup — the four questions
  prompt_engine/           Vendored upstream engine, untouched
  prompt_engine/motion.py  The three motion settings — ours, applied at the seams
  prompt_engine/speech.py  The extra-speech pass over the intent — ours, before the brief
  provisioning/installer.py  Download, verify, extract, validate — one pipeline
  provisioning/importer.py   Installing a model you already have, instead
  inference/               llama-server process, streaming client, GPU detection
  ui/                      Main window and the Qt setup wizard
  ui/touch.py              Fingertip sizing and drag-to-scroll, in one place
installer_files/           Created by the installer (gitignored)
user_data/                 Default install root (gitignored)
```
