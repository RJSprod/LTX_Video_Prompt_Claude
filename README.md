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
| **What runs the model** | Every CUDA card on the machine — twice, once each way (see below) — and your processor. A card decides which pinned `llama.cpp` build gets downloaded, CUDA 12 or CUDA 13 by compute capability; the processor gets the CPU build. |
| **Model quality** | `Q4_K_M`, `Q6_K_P` or `Q8_K_P`. On a card holding the model the default is sized from your VRAM and each option shows its download size and whether it fits; when the model goes to system RAM the default is `Q4_K_M` and each option shows its download size. |
| **A model you already have** | Optional. Point setup at a `.gguf` already on disk and it is installed from there instead of downloaded — see below. Answer no and everything downloads as before. |

### The three ways to run it

Every card is offered twice, and the processor once:

| Choice | Where the model sits | What the card does | VRAM held |
| --- | --- | --- | --- |
| **A card** | In its VRAM | Everything | The whole model, 16–27 GiB |
| **The same card, mixed** | System RAM | The work `llama.cpp` can hand it — large-batch matrix multiplies during prompt processing, and the vision projector | A small working set, roughly 1–2 GiB |
| **Your processor** | System RAM | Nothing; no NVIDIA driver is needed | None |

Mixed mode is the answer to a card that cannot hold its quantization: nothing
is resident on it (`--n-gpu-layers 0`), but it is still the device `llama.cpp`
is pointed at, so it keeps taking the work it can. Token-by-token generation
runs on the processor in both system-RAM modes — what the card adds is prompt
processing and image encoding.

The choice is not permanent. `python app.py --setup` re-asks it, and only the
runtime is re-downloaded if the answer moves between a card and the processor —
the model and projector are already on disk.

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
of a 21 GiB file. `--keep-source` copies instead.

**It keeps its own file name.** A supplied file goes into the folder the
manifest names, not under the file name the manifest names: a models folder that
renamed `MyMerge-Q5_K_M.gguf` to the pinned build's name would claim to hold
something it does not, and would leave you looking for a file that no longer
exists under any name you chose. Nothing downstream needs a particular name —
what was installed is recorded in the state file as the path it actually went
to, and that is the only thing that ever reads it. What is *downloaded* still
lands on the pinned name, which is what the resume and the download cache are
keyed on.

If the vision projector sits beside it — it usually does, both being files from
the same repository — setup offers to take that too. It is found by the pinned
name first and then by any name a projector goes by, so the projector beside a
model you supplied is offered even though its naming is its publisher's. Setup
asks before taking one that does not match the pinned hash, the same way it asks
about the model. Anything not supplied is still downloaded normally.

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
python app.py --setup --gpu 0 --mixed --quant Q4_K_M --yes
python app.py --setup --cpu --quant Q4_K_M --yes
```

`--model-file` and `--mmproj-file` install those from disk instead of
downloading them; a file that is not the pinned artifact stops an unattended run
rather than becoming an install that claims to be something it is not.
`--gpu-layers` is there too, for a card that cannot hold its quantization
entirely in VRAM. `--cpu` answers the device question with the processor
instead of a card, and does not need `nvidia-smi` to do it; `--mixed` answers it
with a card in mixed mode — with `--gpu`, or the first card without it.

## Using it

**Settings → Mode** chooses what the application is doing:

| Mode | What it is |
| --- | --- |
| **Prompt mode** | Everything below — an intent in, an LTX-Video 2.3 positive and negative prompt out. This is what the packaged build does, unchanged. |
| **Conversation mode** | A chat with a character you wrote, on the same local model. See [Conversation mode](#conversation-mode). |
| **Image mode** | A prompt for a text-to-image model, or an instruction for one that edits. Same local model writes it. See [Image mode](#image-mode). |

The three share the window, the display size and the one `llama-server` the
application runs, and nothing else: neither a conversation nor an image prompt
can reach the prompt engine, which is what keeps that engine byte-identical to
upstream. The mode is remembered, so the application reopens on the one you were
using.

### Prompt mode

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
- **View → Display size** switches between Smaller, Small, Comfortable, Large
  and Larger. It restyles the window live, applies to the setup wizard too, and
  is remembered.

  Comfortable and up keep every target at a fingertip or more. **Small** and
  **Smaller** deliberately go under that floor — 41 and 35 logical pixels
  against the 48 a fingertip wants — because prompt mode has a great many
  controls and a large monitor with a mouse on it is a real way to use this.
  Everything scales together, so nothing overlaps or clips; it only gets small,
  and a finger will start to miss things at Smaller. Neither is ever a default.

The window opens on most of the screen rather than at a fixed size, and the
divider between the two panes can be dragged to give either one more room.

### Changing what runs the model, and giving the memory back

**Settings → What runs the model** lists the same devices setup offers — every
CUDA card twice, once holding the model and once in mixed mode, and your
processor — with the one in use ticked. See
[the three ways to run it](#the-three-ways-to-run-it). Picking another one says
what it will do and what it costs before it does anything:

- **the model is not downloaded again.** It is 16–27 GiB already on disk and it
  is the same file whichever device runs it. Only the llama.cpp build changes,
  and only when the new device needs a different one — a card and the same card
  in mixed mode share theirs, so that switch touches nothing on disk. When a
  build *is* needed it is fetched against the same pinned SHA-256 setup uses,
  or unpacked from the download cache if it has been fetched before;
- **the model is unloaded now and loads again on your next generation**, which
  takes a little while at that size. Nothing reloads until you ask for
  something;
- if the installed quantization wants more VRAM than the card you picked
  reports, it says so and by how much — the same warning setup gives, with the
  same answer: mixed mode.

A device cannot be changed while a generation or a reply is running. It says so
rather than interrupting one; wait for it, or press Stop.

Downloading a different quantization stays in **Models and Hardware**, because
that means fetching another 16–27 GiB file and belongs in the wizard that shows
the size and the progress.

### Running a model you already have

**Settings → Which model runs** is the other half of that menu: that one changes
the hardware and keeps the weights, this one keeps the hardware and changes the
weights. Give it the path to any `.gguf` on the machine and that is what runs
from your next generation.

It is deliberately the small half of setup. Nothing is downloaded, verified,
copied or moved — the file is read where it is, so a model on another drive
stays on that drive — and the llama.cpp runtime, the device and the context size
are all left exactly as they were. A file the release manifest does not pin
cannot be checked against a hash it has no entry for, so this does not pretend
to: it is the escape hatch for running something else, and the pinned download
remains what setup installs.

**The vision projector is optional, and asked for separately.** The pinned model
ships beside its `mmproj` file; an arbitrary GGUF may have none. So there is a
second box for the projector's path, with **None** beside it:

- **give one** and images work as they always have — image-to-video prompts,
  and pictures attached in a chat. When a file that looks like a projector is
  sitting beside the model you picked, which is where it usually is, it is
  offered into the box as a suggestion you can empty. It is never paired
  automatically: nothing in a file name proves a projector was made for the
  model beside it;
- **leave it empty** and the model still runs, answering text. `--mmproj` is
  left off the command line rather than passed something empty, and anything
  carrying an image is refused with a sentence saying why — before the
  generation starts, while the image is still attached to remove. The status bar
  reads `Model: Q6_K_P, no vision` so it is not a surprise later, and the chat
  says the same when you attach a picture.

Like a device change, the model cannot be changed mid-generation, and the server
is unloaded so the next generation loads the one you chose.

**Settings → Unload the model from memory** hands the VRAM or system RAM back
now instead of when the application closes — for giving a card to something
else without quitting. The next generation starts the server again by itself.

## Conversation mode

The second thing the drop-down offers: a chat with a character, on the model
already installed. It is [oobabooga](https://github.com/oobabooga/textgen)'s
chat tab and character editor, in the same file format, laid out for a finger.

### Characters

**Characters…** opens the editor: the list of characters down one side, and the
four fields that make one down the other.

| Field | What it is |
| --- | --- |
| **Name** | What the character is called, and the name on their replies. |
| **Picture** | Optional. Shown beside every reply, and in the character list. |
| **Context** | Who they are — personality, background, how they speak. This is what the model is told before the first line. |
| **Greeting** | The first thing they say. Every new chat opens on it. |

`{{char}}` and `{{user}}` in either text box are replaced with the character's
name and yours; `<BOT>` and `<USER>` work too, because characters written for
other front ends use them.

Characters are YAML files in `user_data/characters/`, with the picture beside
each one as `<name>.png` — oobabooga's own layout, so a folder copied from a
text-generation-webui install works here as it stands, and one written here
works there. **Import…** takes the three shapes a character comes in:

- a `.yaml` character;
- a `.json` character, TavernAI's fields included;
- a `.png` **character card** — the JSON embedded in the image, as TavernAI and
  SillyTavern write it. The card is also the character's picture.

Cards carry fields this editor has no box for — personality, scenario, example
dialogue. They are folded into the context under headings rather than dropped.

### Yourself

**You…** is optional, and empty by default: characters reply to an unnamed
"you", which is what a chat with no persona has always been. Fill in a name and
a description and both go into the system prompt, so the character knows what to
call you and can use what you told it. It is one persona for every character,
stored in `data/persona.json`.

### The chat

Type, and press **Send** or Ctrl+Enter. The reply streams in as it is written,
and **Stop** ends it early, keeping what had arrived.

The transcript is bubbles on their own sides — yours to the right, theirs to
the left with their picture beside the first reply of each run — and no names
over anything, because the side a message is on is what says who said it. A
bubble is as wide as its words and no wider, up to about three-quarters of the
transcript.

**It follows the newest message until you leave it.** Scroll up to read
something and it stops chasing what arrives; scroll back to the bottom and it
starts again. Sending a message, opening a chat and starting one all count as
going back to the end.

**View → Conversation** hides any of the three rows around the transcript —
the *Talking to* bar, the *Chat* bar and the quick actions — one at a time and
independently. On a small screen a chat that has been set up is mostly
transcript, and the two drop-downs only matter while they are being changed.
What you hide is remembered.

Under the transcript are the four things done between messages — **Regenerate**,
**Continue**, **Impersonate** (the model writes your next message into the box
for you to edit), and **Remove last**, which takes back the last exchange and
puts your message back where you typed it.

**Tap a message** to reveal a **⋯** on it — the touch equivalent of a
long-press, and where the rest of what can be done to that message lives.
Tapping it again puts it away, dragging still scrolls, and dragging across
words still selects them:

| Action | What it does |
| --- | --- |
| **Edit** | Rewrite anything either of you said, in place. |
| **Regenerate** | Write that reply again — see below. |
| **Continue** | Carry the last reply on from where it stopped. |
| **Send again from here** | Answer one of your messages again, dropping what followed. |
| **Branch from here** | Copy the chat up to that message into a new one. The chat it came from is untouched. |
| **Delete message** / **Delete from here** | One message, or that message and everything after it. |
| **Copy** | The message text, to the clipboard. |

**Regenerating keeps the reply it replaced.** A regenerated message grows a
`◀ 2/3 ▶` pager — the one thing shown without a tap, because it is what says an
earlier attempt is still there — and paging back to it is how a regenerate that
came back worse is undone. **Delete this version** drops just the one showing.

**Past chats** are the drop-down at the top: every conversation with that
character, newest first, named after the first thing you said in it and
renameable. Each is a JSON file under `user_data/chats/<character>/`.

**Attach…** sends a picture with your message — the same preprocessing prompt
mode uses, and the same vision projector. Older pictures drop out of the
context as the conversation grows, and the messages that carried them say so.

**Settings** folds out a panel of temperature, top-p, reply length, seed and a
custom system message that replaces the built one entirely. They are saved in
the character's own file, so each character keeps how it is talked to. A seed of
`-1` draws a fresh one per reply, which is what makes a regenerate come back
different.

Changing one of them is what saves it: the panel writes itself back to the
character a moment after the control stops moving, so holding a stepper down is
one write rather than forty, and letting go of it shows the change already
saved. **Save** at the bottom of the panel does it now and says which character
it went to — the button is not the only way to save, it is the way to be told
that it saved. The line above it says which state the panel is in, and a write
that fails says so there rather than going quiet.

The conversation is trimmed from the front to fit the context window
`llama-server` was started with. The character survives a long chat; the
beginning of the chat does not.

## Image mode

The third thing **Settings → Mode** offers. A video prompt describes change over
time — motion, beats, camera travel, speech, duration. An image prompt describes
one instant, and is judged on composition, light and material. Roughly half the
LTX engine's control surface has no image equivalent, so image mode is a second
engine rather than the same one with different labels: `image_engine/`, written
against the three profiles below, sharing no code with `prompt_engine/`.

### The two tasks

Chosen with **Task**, at the top of the page. They share the page and not the
control set.

| Task | What goes in | What comes out |
| --- | --- | --- |
| **Generate** | An intent — one line or many | A finished text-to-image prompt, and a negative prompt where the model has one |
| **Edit** | A terse request — "swap the wooden door for a steel one" | A full editing instruction: the change, the intended result, and what must survive it |

### The models, and why the page changes shape

Three profiles, and they disagree in ways that are not cosmetic:

| | FLUX.2 [klein] 9B | Krea 2 Turbo | Krea 2 RAW |
| --- | --- | --- | --- |
| Steps / CFG | 4 / 1.0 | 8 / 0.0 | 52 / 3.5 |
| **Negative prompt** | none | none | real |
| **Instruction editing** | native | experimental only | experimental only |
| Reference images | 4 | 2 | 2 |
| Hex colour binding | yes | no | no |
| JSON prompts | yes | no benefit | no benefit |
| Native edge | 1024 | 1536 | 1536 |

Five things follow the **Image model** drop-down, live, and all five hide rather
than grey out — a populated field the model discards gives a false sense of
control, which is worse than the field not being there at all:

- the **negative prompt** controls and pane, on the one profile with real
  classifier-free guidance, and on no editing path at all;
- the **hex swatch** option, which only FLUX binds to a named object;
- the **JSON prompt shape**, offered only where it is parsed;
- the **Edit task**, which moves to the model that edits natively rather than
  letting you wait for a server to start and then be refused;
- the number of **reference slots**, capped at what the model reads.

The line beside **Seed** says the rest: the frame, the steps, the guidance, the
word band and — in edit mode — the denoise the strength control resolves to.
Those are what the image model itself has to be set to, and they change with the
profile, so they are shown rather than left to be looked up.

### Auto, Choose for me, Random

Every drop-down carries the same three values above its options. They answer
three different questions:

| | Means | Consults the intent | States anything |
| --- | --- | --- | --- |
| **Auto** | "I don't care" | no | no — the facet is left out entirely |
| **Choose for me** | "You decide, and tell me" | **yes** | yes |
| **Random** | "Surprise me" | no | yes |

**Choose for me** goes to the local model in one batched call, is validated
against the list it was given, and falls back to keyword matching when the model
is unavailable or answers unusably. What it settled on appears under the
drop-down with where it came from — `model`, `heuristic`, `random` or nothing
fitted — because a choice you cannot see is no better than Auto, and the value
shown is what goes in the box next time.

**Random** is drawn per control from the seed rather than from one shared
sequence, so adding a control later does not shift every draw after it. A fixed
seed reproduces last week's image; `-1` draws a fresh one each generation.

### Editing

Three things an edit instruction carries that a generation prompt never does,
and all three are controls:

- **Target element.** These models have no mask, so the description of what to
  change does the work a mask would. "Remove the guy" fails when there are three
  people.
- **What must survive.** They change what they are told to change and drift on
  anything left unmentioned, so preservation is stated rather than assumed —
  this is the single biggest quality lever in edit mode. Identity, composition,
  lighting and style are ticked by default.
- **Physical consistency.** A new object needs the scene's light direction, its
  perspective and a contact shadow, or it reads as a sticker.

References are addressed by **number and role together** — "the background from
image 2" — so each attached image has a role beside it. A request that looks like
several changes at once is flagged rather than blocked: these models hold
identity better across a chain of small edits than across one compound
instruction, and the writer is told to write the most important one.

Krea 2 does not ship with editing at all; Krea's own technical report lists it as
future work, and the path that exists needs a third-party LoRA and a custom node
pack. It is reachable through **Allow the experimental editing path**, which
says what has to be installed before it will do anything.

### What it deliberately does not have

The LTX engine ships an `undress` control. It is **not** ported, and should not
be added later: a control whose function is removing clothing from a depicted
person is the core mechanic of nudify tooling, and the harm is the same whether
what comes out is an image or a prompt that reliably produces one. Edit mode
would make it worse still, because the input is a photograph of a real person.

Full wardrobe, costume and period-dress banks are present, and **Change
clothing** is one of the twenty edit operations, so ordinary garment changes work
normally. Both sanitisers strip undressing directives arriving through the
free-text boxes, and there are tests asserting both halves of that — the boundary
holds, and it is not enforced by over-blocking.

This is the one place where image mode deliberately lacks parity with video
mode.

### Reading further

[`docs/IMAGE_PROMPT_MODE_SPEC.md`](docs/IMAGE_PROMPT_MODE_SPEC.md) is the
engine's specification — the model facts and where they came from, the two
sanitisers and why they are not interchangeable, and the list of constants to
re-verify against current model cards. `python tools/image_engine_demo.py` walks
through every behaviour with a stub writer, so it needs no server.

## Requirements

- Windows x64
- An NVIDIA GPU with a current driver, or an x64 processor. The 3090 and 5090
  are the pinned configurations; any other CUDA card is sized from its own VRAM
  and compute capability. A card that cannot hold the model can run in mixed
  mode instead, keeping the weights in system RAM and doing the processing it
  can. Setup also offers your processor on its own, which installs the CPU
  build of `llama.cpp` — no NVIDIA card or driver is used, and none is
  required. See [the three ways to run it](#the-three-ways-to-run-it).
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
| Running with a card too small for the model | Not possible | Mixed mode: same CUDA install, weights in system RAM, nothing resident on the card, which keeps the work `llama.cpp` can give it |
| Where the model comes from | Downloaded, always | Downloaded, or installed from a `.gguf` you already have — against the same pinned SHA-256, and keeping its own file name |
| Window | One column of mouse-sized controls | Two panes, fingertip-sized targets, drag-to-scroll, five display sizes from Smaller to Larger |
| Changing device | Re-run setup | That, or Settings → What runs the model, which swaps the llama.cpp build and keeps the model that is installed |
| Changing model | Re-run setup, downloading another pinned quantization | That, or Settings → Which model runs — any `.gguf` on the machine, read where it is, with its vision projector optional |
| Freeing the memory | Close the application | That, or Settings → Unload the model from memory |
| What the window does | Writes prompts | That, or — on Settings → Mode — a chat with characters you write or import, or a prompt for a text-to-image model, both on the same local model |
| Motion | One way of writing it | Default is that way exactly; **Inertia** and **Flow** are opt-in |
| Seed | A number | A number, or `-1` for a fresh one each generation |
| Speech | Whatever the intent quotes | That, or up to 10× more written in the same voice — opt-in |

Four behavioural changes, all bounded, and the two that can reach a prompt are
both opt-in and both open on the old behaviour. The GPU widening is pinned so
that it cannot affect the two supported cards: `device_detection.PINNED` maps
them to their original runtime and quantization before any heuristic runs, and
`tests/test_install_flow.py` asserts that. The CPU and mixed options are the
same kind of widening — answers the setup questions now offer, reached only by
choosing them, and leaving what a card downloads and how it is launched exactly
as it was: CPU mode resolves to its own pinned `llama.cpp` archive, and mixed
mode changes one number in the command that starts the same CUDA runtime. The
motion presets are additive —
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

Conversation mode and image mode add no fifth change, because neither is on the
path a prompt takes. Both import nothing from `prompt_engine` — image mode has
its own engine, in `image_engine/`, and two tests walk the AST of every module
in it to assert that no import ever crosses — and both reach the model through
the same `InferenceService` and `LlamaClient` prompt mode uses. A prompt
generated with the mode set any of the three ways is the same prompt.

## Tests

```
python -m pytest tests/        # 938 passed
```

- `test_upstream_parity.py` (434) — the upstream self-test, ported
- `test_install_flow.py` (139) — install-root discovery, GPU sizing, the CPU
  and mixed devices, manifest resolution, console setup, supplying a model from
  disk, changing device without re-downloading the model, and the installer's
  interpreter and environment checks
- `test_image_engine.py` + `test_selection_quality.py` (157) — the image engine
  as delivered: the two import-wall tests that guard LTX parity, profile
  invariants, seeded randomness, Choose-for-me and its local fallback, geometry
  across every profile and ratio, both system prompts, edit mode, the two
  sanitisers, negative gating, and a standing 28-case benchmark for the
  fallback's quality. Stdlib only — no Qt, no server, no network
- `test_touch_ui.py` (72) — target sizes, drag-to-scroll, the sliders and the
  five display sizes, the menu-bar mode switch and the View toggles, the
  runtime device menu and unloading, the transcript's layout and its sticky
  bottom, and conversation mode driven end to end against a scripted server,
  measured on a real window built offscreen
- `test_prompt_engine.py` (44) — the adapter seam, the motion presets, speech
  expansion and the UI option sources
- `test_chat.py` (36) — the character format and its three imports, chat
  history and branching, and what a chat turn puts on the wire
- `test_image_mode.py` (41) — the seam image mode adds: the client adapter's
  message shape and its push-to-pull bridge, the controls being the engine's
  own banks, the five things that follow the model drop-down, and both tasks
  driven end to end against a scripted server
- `test_core.py` (15) — multimodal requests, atomic JSON, SSE, zip-slip,
  download resume and retry

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
  image_engine/            Image mode's engine — its own, sharing nothing with the above
  image_engine/profiles.py Everything that differs between one image model and another
  image_engine/options.py  Every image option bank, and the three sentinels
  image_engine/edit.py     Instruction editing: the target, what survives, references
  image_engine/sanitize.py The two sanitisers — generation and edit, not interchangeable
  provisioning/installer.py  Download, verify, extract, validate — one pipeline
  provisioning/importer.py   Installing a model you already have, instead
  inference/               llama-server process, streaming client, GPU detection
  inference/image_client.py  That client in the shape the image engine wants
  chat/                    Conversation mode, below the window
  chat/characters.py       Characters in oobabooga's format, and the persona
  chat/history.py          Messages, their versions, branching, saved chats
  chat/prompt.py           What one chat turn puts on the wire
  chat/yamlish.py          The YAML subset a character file is written in
  ui/                      Main window, chat page, image page, character editor, setup wizard
  ui/touch.py              Fingertip sizing and drag-to-scroll, in one place
docs/                      The image engine's specification
tools/                     Upstream parity harnesses, and the image engine's demo
installer_files/           Created by the installer (gitignored)
user_data/                 Default install root (gitignored)
  characters/              Characters and their pictures — oobabooga's layout
  chats/                   Saved conversations, filed by character
```
