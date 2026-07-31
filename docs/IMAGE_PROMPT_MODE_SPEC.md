# Image Prompt mode — implementation specification

**For:** the developer integrating `image_engine` into `RJSprod/LTX_Video_Prompt_Claude`
**Engine version:** 2.0.0
**Status:** engine complete, 157 tests passing. UI and wiring to be built.

---

## 0. Where this landed

The document below is as received, and is still the reference for the engine.
This section records where each piece went and the four places the integration
answered a question the spec left open. Nothing else in the spec was edited.

| Delivered | Here |
| --- | --- |
| `image_engine/` | `src/prompt_master/image_engine/` — byte-identical |
| `tests/test_image_engine.py`, `tests/test_selection_quality.py` | `tests/`, with the import path changed to `prompt_master.image_engine`, and a note in each docstring saying that is the only edit |
| `demo.py` | `tools/image_engine_demo.py` — `python tools/image_engine_demo.py` |
| `docs/IMAGE_PROMPT_MODE_SPEC.md` | this file |
| — | `src/prompt_master/inference/image_client.py` — the adapter §12.1 asked for |
| — | `src/prompt_master/ui/image_page.py` — the page §9 describes |
| — | `tests/test_image_mode.py` — the adapter, the controls and both tasks end to end |

**The `ChatClient` signature differs, and the difference is a direction.** The
real `LlamaClient.stream_chat(messages, max_tokens, seed, on_text, cancel, …)`
pushes fragments to a callback and returns the whole reply; the protocol here
pulls. `ImageChatClient` bridges the two with a thread and a queue rather than
handing the UI the callback, because the engine makes three kinds of call on
one client and only the writer pass is yielded back out of `stream()`. Passing
the callback through would put the Choose-for-me JSON and the smart-negative
term list in the output pane. The engine was not edited.

**`PromptShape` cannot travel through Qt item data.** It is a `str` enum, so Qt
stores it as the plain `str` it also is, and `build_brief` tests the member by
identity — a shape round-tripped through a combo box would have compared equal
to JSON and still been written as prose. The page stores `.value` and rebuilds
the member. Worth knowing before another enum control is added.

**The generation reference roles are the one list the page holds.** §9.4 gives
edit references a role bank in `option_banks()`, but the three roles the
*generation* system prompt has guidance for — style, subject, composition —
exist in `system.py` as prose rather than as a bank. The page names them, and a
test asserts each one changes the system prompt, so the list cannot drift into
offering a role the writer is given nothing for. A bank would be better.

**§12 is not done.** Every step, CFG and denoise constant is still as delivered.
They are one-line edits in `profiles.py` and want checking against current model
cards before anyone relies on the numbers the page displays.

---

## 1. What this is

A third mode alongside **LTX Prompt** and **Conversation**, with two tasks:

- **Generate** — an intent (one line or many) becomes a full text-to-image prompt.
- **Edit** — a terse request ("swap the wooden door for a steel one") becomes a
  complete editing instruction with the target pinned down and the preservation
  contract stated.

Both are written by the same local Gemma model already running behind the other
two modes.

The engine is a package with **no dependencies outside the standard library**.
It imports nothing from the application, and its tests run with no Qt, no
llama-server and no network. You inject the streaming client; everything else it
does itself.

### Files delivered

```
image_engine/
  __init__.py      public API, design contract
  profiles.py      ModelProfile + the three model profiles
  options.py       every option bank, the three sentinels, selection cues
  selection.py     "Choose for me" — batched model pass + local fallback
  brief.py         generation controls, brief, sentinel resolution, geometry
  edit.py          edit controls, brief, reference handling
  system.py        system + user prompts for both tasks, per model
  negative.py      gated negative banks
  sanitize.py      two sanitisers — see §7, they are not interchangeable
  engine.py        orchestrator, ChatClient protocol, result types
tests/
  test_image_engine.py       132 tests
  test_selection_quality.py   25 tests — a standing benchmark for CHOOSE
demo.py            runnable with no server; shows every behaviour below
docs/
  IMAGE_PROMPT_MODE_SPEC.md  this file
```

Drop `image_engine/` at `src/prompt_master/image_engine/` and the tests at
`tests/`. Run `python demo.py` first — it exercises every path described here
without needing a model.

---

## 2. The non-negotiable constraint

**The image engine must never import from `prompt_master.prompt_engine`.**

The LTX engine is byte-identical to upstream and `tools/compare_upstream_engine.py`
verifies that. An import from image mode would place this code inside the parity
surface, and "0 mismatches" would stop meaning what `PARITY_REPORT.md` says.

Enforced mechanically across every module, not by convention:

```python
def test_no_import_from_prompt_engine() -> None:
    for path in _modules():
        # ...walks the AST of every module in the package
        bad = [m for m in imported if "prompt_engine" in m or "prompt_master" in m]
        assert not bad, f"{path.name} imports {bad}"
```

A companion test asserts imports are stdlib-only. **Wire both into CI and treat
a failure as a release blocker** — they guard the parity report, not just this
package.

---

## 3. Why this is a new engine, not a parameterisation of the old one

A video prompt describes **change over time**: motion, beats, camera travel,
speech, duration, transitions. Roughly half the LTX engine's control surface —
accents, music genres, dialogue percentage, transitions, FPS, duration, motion
presets, the speech expansion pass — has no image equivalent. An image prompt
describes **one instant**, judged on composition, light and material.

Sharing the engine would mean either dead controls or conditionals threaded
through code that is contractually frozen. Same *architecture* (seeded casting →
brief → system prompt → streamed writer pass → gated negative), entirely new
*content*.

---

## 4. The models disagree, and that drives the design

Read this before reading the code.

| | FLUX.2 [klein] 9B | Krea 2 Turbo | Krea 2 RAW |
|---|---|---|---|
| Text encoder | Qwen3 8B embedder | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct |
| Steps / CFG (generate) | 4 / ~1.0 | 8 / 0.0 | 52 / 3.5 |
| **Negative prompt** | **none** | **none** | **real (CFG)** |
| **Instruction editing** | **native** | **experimental only** | **experimental only** |
| Reference images | 4 | 2 (LoRA path) | 2 (LoRA path) |
| Masked inpainting | no | no | no |
| Attention weights | ignored | ignored | ignored |
| Hex colour binding | yes | no | no |
| Text rendering | strong | fair | fair |
| JSON prompts | parses reliably | no benefit | no benefit |
| Word band (generate) | 40–120 | 30–140 | 30–140 |
| Word band (edit) | 20–110 | 15–90 | 15–90 |
| Native edge | 1024 | 1536 | 1536 |

Three consequences that shape everything:

**A negative field must not exist for two of three profiles.** Both distilled
checkpoints have no negative-conditioning path. A populated field would be
silently discarded while giving the user a false sense of control — worse than
no field. `build_negative()` returns `""` for them unconditionally, and the
status line says `no negative` so it isn't a mystery.

**Edit mode is FLUX-only by default.** FLUX.2 [klein] edits natively. Krea 2
does not ship with editing at all — Krea's own technical report lists it as
future work — and instruction editing on those weights exists only through a
third-party LoRA plus custom ComfyUI nodes. `editing_profiles()` returns FLUX
alone unless you pass `include_experimental=True`, and `build_edit_brief()`
raises `ValueError` with the install requirements if a user reaches it anyway.

**The system prompts genuinely differ.** FLUX wants Subject → Action → Style →
Context with the subject in the first clause. Krea wants plain physical
description and specifically punishes the glossy vocabulary other models reward,
having been tuned against the "AI look". Feeding one the other's system prompt
measurably degrades output.

Everything model-specific lives in the frozen `ModelProfile`. **Adding a fourth
model should mean adding a profile, not editing a builder.** If you write
`if profile.family == "flux"` outside the two helpers in `system.py` that
already do it, the property you need is missing from the profile.

---

## 5. The three sentinels

Every dropdown carries three values above its real options. They answer
different questions, which is why all three exist.

| Sentinel | Means | Behaviour | Consults intent? | States anything? |
|---|---|---|---|---|
| `AUTO` | "I don't care" | Facet omitted from the brief entirely | no | no |
| `CHOOSE` | "You decide, and tell me" | Engine picks the best fit and states it | **yes** | yes |
| `RANDOM` | "Surprise me" | Seeded draw from the bank | no | yes |

`AUTO` and `RANDOM` differ in whether anything is stated. `CHOOSE` and `RANDOM`
differ in whether the intent is consulted.

### 5.1 RANDOM

Seeded per control:

```python
rng = random.Random(seed ^ crc32(control_name))
```

Two properties worth understanding:

- **crc32, not the builtin `hash`.** Python salts string hashing per process, so
  a shared seed would only reproduce within a single run of the application.
- **Per control, not one shared sequence.** With a shared sequential RNG, adding
  a new control shifts every draw after it, and last week's seed stops
  reproducing last week's image. There's a test for exactly this.

With `seed = -1` the seed is drawn fresh each generation, so RANDOM is fresh
each generation. With a fixed seed it reproduces. Both behaviours come from the
one mechanism.

### 5.2 CHOOSE

One batched call to the local model, then validation, then a local fallback.

Three things make it reliable on a 9–27B model:

1. **Batched and capped.** All CHOOSE controls go in one request, up to
   `MAX_CONTROLS_PER_CALL = 6`; more than that splits into further calls.
   Constrained-choice accuracy degrades as simultaneous fields grow. Six is
   conservative — raise it with measurements, not by feel.
2. **Options shuffled per control, exact-string output.** Language models have a
   measurable bias toward particular positions in an enumerated list. Shuffling
   (seeded, so still reproducible) turns systematic bias into noise, and
   requiring the exact option string rather than a letter removes the
   letter-token bias entirely.
3. **An explicit escape value.** A model forced to choose when nothing fits will
   invent something or take the first item. Given `"__none__"` it says so, and
   the control falls back to AUTO.

Everything returned is validated against the bank. Case and whitespace drift are
tolerated; nothing looser, because a fuzzy match is how invented options get
through. Anything invalid, missing or unparseable falls to the deterministic
scorer.

**Surface the result.** `Generation.chosen` and `.chosen_source` carry what was
picked and where it came from (`model`, `heuristic`, `random`, `none`). Showing
it is the entire point — a hidden choice is no better than AUTO. Suggested UI:
the resolved value greyed beside the dropdown, with a tooltip naming the source.

### 5.3 The fallback, and its known quality

`heuristic_pick()` scores each option by keyword overlap: curated cue phrases
(5.0), curated cue words (3.0), the option's own name (2.0), description tokens
(0.5), with ambiguous state words like "dark" and "old" demoted to 1.5. Below a
threshold it returns nothing, because a bad guess is worse than silence.

`tests/test_selection_quality.py` is a **standing benchmark**, currently 28
cases at 100%. It exists so cue edits can't regress silently. Two regressions it
already caught and now guards:

- Generic verbs beating specific nouns — "put her in a red coat" resolved to
  *Add an object* instead of *Change clothing*.
- Ambiguous state words beating specific ones — "after dark" beat "campfire",
  giving *Chiaroscuro* instead of *Firelight*.

When a case fails, fix the cue table, not the threshold. If a case becomes
genuinely ambiguous, delete it — one already was ("make it snow" is defensibly
weather or season).

---

## 6. Edit mode

### 6.1 What an edit instruction has to carry

Three things a generation prompt never does:

1. **Which element.** "Remove the guy" fails when there are three people. The
   target is pinned by position, colour, size or relationship. These models have
   no mask, so the target description does the work a mask would.
2. **What survives.** They change what they're told to change and drift on
   anything left unmentioned. Preservation is stated, not assumed — this is the
   single biggest quality lever in edit mode.
3. **Physical consistency.** A new object needs the scene's light direction, its
   perspective and a contact shadow, or it reads as a sticker.

The system prompt enforces the shape: **imperative change → intended result →
preservation clauses.**

### 6.2 References

FLUX.2 [klein] takes up to 4, addressed by **number and role together** — "the
subject from image 1", "the background from image 2". A reference mentioned
without its number may not resolve; a number without a role says nothing about
what to take. `ReferenceImage(index, role)` carries both, indices are
renumbered from 1 on build, and the list is capped at
`profile.max_reference_images` so the prompt never promises more than the model
accepts.

### 6.3 Controls

| Control | Notes |
|---|---|
| `operation` | 20 operations. Defaults to `CHOOSE` — a request usually says plainly enough |
| `target_element` | Free text; empty means the writer infers it, less reliably |
| `localization` | 13 positions |
| `preservation` | Multi-select, 11 targets. Defaults to Identity + Composition + Lighting + Style |
| `edit_strength` | Preserve / Balanced / Transform → maps to a denoise hint |
| `match_lighting`, `match_perspective`, `seamless_blend` | Physical consistency, all default on |
| `references` | Up to the profile limit |
| `allow_experimental_edit` | Off. Required for the Krea path |

**Compound requests are flagged, not blocked.** These models degrade over
compound instructions and hold identity better across a chain of small edits.
`_looks_compound()` detects several changes in one request; the writer is told
to write only the most important one, and the user gets a note suggesting the
split.

---

## 7. The two sanitisers are not interchangeable

**This is the most destructive possible bug in the package, so it gets its own
section.**

In a *generation* prompt, negation is harmful — these models cannot subtract, so
naming a thing puts it in frame. `sanitize_positive()` strips it.

In an *edit instruction*, negation is load-bearing. Not mainly because of removal
verbs — "Remove the bins" survives either sanitiser, since "remove" isn't a
negation word. The real casualty is the **preservation clause**, which is very
often phrased negatively:

```
in : Add a bench beside the tree, without altering the background.
gen: Add a bench beside the tree.                        <- clause gone
ed : Add a bench beside the tree, without altering the background.
```

The generation sanitiser deletes exactly the part holding the image together.
`sanitize_edit_instruction()` leaves negation alone entirely and instead warns
when a preservation clause is *missing*.

I originally documented this as being about removal verbs, and wrote two tests
that passed vacuously as a result — neither input was ever altered by either
sanitiser. They're now parametrised over four genuinely divergent cases and
assert the generation sanitiser *does* shorten the text, so the test can't go
hollow again.

Both sanitisers also strip preambles and fences, remove attention weights while
keeping the word, drop quality slop (`UNIVERSAL_AVOID` plus the per-profile
list), normalise the punctuation removals leave behind, enforce word ceilings on
sentence boundaries, and restore capitalisation.

Bugs found and fixed during testing, worth knowing because they reappear under
edit:

- The code-fence pattern's language-tag matcher `` ```[a-zA-Z]* `` silently ate
  the first word of `` ```A weathered fisherman ``. Now lowercase-only and must
  end at whitespace. Regression test.
- Word removals left `", ,"` and `"fog,."` debris, which the model reads as
  structure. Punctuation normalisation pass.
- Stripping a clause after `;` left `"overcast;."`. Regression test.
- The smart-negative pass could echo the positive back and negate the image
  being asked for. Refine terms appearing in the positive are dropped.

---

## 8. Public API

```python
from prompt_master.image_engine import (
    ImagePromptEngine, ImageControls, EditControls, ReferenceImage,
    option_banks, profile_for, editing_profiles, CHOOSE, RANDOM, AUTO,
)

engine = ImagePromptEngine(llama_client)     # any object matching ChatClient

# --- generate ---
brief = engine.plan(intent, controls)        # resolve without touching model:
                                             # seed, dimensions, CHOOSE results
for chunk in engine.stream(intent, controls):
    append_to_output_pane(chunk)
result = engine.finish()                     # -> Generation

# --- edit ---
brief = engine.plan_edit(request, edit_controls)   # raises if model can't edit
for chunk in engine.stream_edit(request, edit_controls):
    append_to_output_pane(chunk)
result = engine.finish_edit()                # -> EditGeneration
```

Non-streaming: `engine.generate(...)` / `engine.edit(...)`.

`Generation` carries `.positive`, `.negative`, `.seed`, `.width`, `.height`,
`.steps`, `.cfg`, `.word_count`, `.chosen`, `.chosen_source`, `.notes`, plus
`.status_line()` and `.as_dict()`.

`EditGeneration` carries `.instruction`, `.operation`, `.denoise`,
`.references`, `.warnings` and the same reporting helpers.

### The client contract

```python
class ChatClient(Protocol):
    def stream_chat(self, *, system: str, user: str, temperature: float,
                    top_p: float, seed: int,
                    image_jpeg_b64: str | None = None) -> Iterator[str]: ...
```

Structural, so the engine imports nothing from `inference/`. **Confirm the real
`LlamaClient` matches.** If not, write a thin adapter rather than editing the
engine — client-agnosticism is what makes it testable against a scripted double.

---

## 9. What to build

### 9.1 Mode registration

Follow Conversation mode; it's the working precedent. Add the mode to the
settings enum, persist the selection, and give the page a **task selector
(Generate / Edit)** at the top — the two tasks share a page but not a control
set.

The three modes share the window, the display size and the one `llama-server`.
They share nothing else.

### 9.2 Controls

Build every dropdown from `option_banks()`. Never hold a local list — that's the
rule `prompt_engine/options.py` already enforces for LTX mode and the reason no
control there can drift from the engine consuming it.

Each dropdown gets Auto / Choose for me / Random above its real options.

### 9.3 Reactive UI — where the bugs will be

Five things must respond live to the profile dropdown:

1. **Negative controls hidden** (not disabled) unless `profile.negative_supported`.
   Krea 2 RAW only.
2. **Hex swatches** hidden unless `profile.supports_hex_colour`. FLUX only.
3. **JSON shape** offered only when `profile.supports_json_prompt`. FLUX only.
   The engine downgrades silently, but don't offer a choice you won't honour.
4. **Edit task disabled** unless `profile.can_edit`. With
   `allow_experimental_edit` on, show `profile.edit_requirements` prominently
   before letting the user proceed.
5. **Reference slots** capped at `profile.max_reference_images` — 4 for FLUX,
   2 on the Krea path.

Also surface `default_steps`, `default_cfg`, `edit_steps`, `edit_cfg`, the
resolved denoise and the dimensions. The user needs those to configure the
image model itself, and they change with the profile.

### 9.4 Reference images

Reuse the existing attachment path verbatim: EXIF-normalise → RGB → 768px →
JPEG → base64. Same vision-projector guard as prompt mode — if no `--mmproj` is
configured, refuse before generation starts, while the image is still attached
to remove.

For edit mode, each attached image needs a **role** dropdown beside it
(`REFERENCE_ROLES`). That's what lets the instruction say "the background from
image 2".

### 9.5 Touch and display size

Nothing special. `ui/touch.py` sizing, drag-to-scroll, 48px minimum targets at
Comfortable and above. Same as the other modes.

---

## 10. Flow

```
intent / request + controls
      ↓
seed fixed  (-1 → drawn now, before anything else)
      ↓
resolve_sentinels()      RANDOM drawn per control from the seed
                         CHOOSE batched → model → validated → local fallback
      ↓
build_brief() / build_edit_brief()
      ↓
build_system() / build_edit_system()      model- and task-specific
      ↓
client.stream_chat(temperature=profile.writer_temperature, seed=…)
      ↓
sanitize_positive() / sanitize_edit_instruction()      NOT interchangeable
      ↓
[Krea 2 RAW, generate only] smart-negative second pass
      ↓
build_negative()         gated banks, or "" when the model takes none
      ↓
Generation / EditGeneration
```

The seed is fixed **before** the brief and is the same number handed to the
sampler, so a reported seed reproduces both the engine's choices and the model's
sampling. Identical contract to LTX mode.

---

## 11. Tests

157 tests, all passing:

```
python -m pytest tests/ -q
```

Coverage: the import wall (2), profile invariants (8), RANDOM determinism and
independence (5), CHOOSE resolution, validation, batching, shuffling and
fallback (13), brief and seeding (5), geometry across every profile × ratio
(~30 parametrised), generation system prompts (5), **edit mode (17)**, the two
sanitisers (14), the content boundary (4), negative gating (6), engine
orchestration (8), option bank integrity (6), plus the 28-case selection
benchmark.

---

## 12. Verify before shipping

Listed honestly rather than buried.

1. **`LlamaClient.stream_chat` signature.** The `ChatClient` protocol is my best
   reconstruction of it. Adapt rather than edit the engine if it differs.
2. **Model facts.** Krea 2 postdates my training data; its figures (12.9B DiT,
   Qwen3-VL-4B encoder, Turbo 8 steps / CFG 0, RAW 52 / 3.5, editing absent from
   the shipped model) come from research rather than firsthand knowledge.
   FLUX.2 [klein]'s 4 steps, no-negative behaviour, 4-reference editing and
   number+role addressing are well-corroborated including by Black Forest Labs'
   own guidance. **Re-verify every step/CFG/denoise default against current
   model cards** — all are one-line edits in `profiles.py`.
3. **Edit denoise values** (0.35 / 0.5 / 0.7) are community-sourced rather than
   official, because the distilled checkpoints ignore the knob entirely.
   Validate on your own pipeline.
4. **The Krea edit path** depends on a third-party LoRA and node pack that
   neither Krea nor I control. If Krea ships official editing, promote the
   profile from `EXPERIMENTAL` to `NATIVE` and revisit `max_reference_images`.
5. **Licensing.** FLUX.2 [klein] 9B is non-commercial (4B is Apache-2.0); Krea 2
   has a revenue/seat threshold. Both surface as `licence_note` in
   `.notes`. Confirm current terms and consider showing them in the model
   picker.

---

## 13. Content boundary

The LTX engine ships an `undress` control. **It is deliberately not ported, and
should not be added later.**

A control whose function is removing clothing from a depicted person is the core
mechanic of nudify tooling, and the harm — non-consensual intimate imagery — is
the same whether the output is an image or a prompt that reliably produces one.
Prompt generation is exactly the step that makes it easy, and edit mode would
make it easier still, since the input is a real photograph of a real person.

What is present: full wardrobe, costume and period-dress banks, and a
`Change clothing` edit operation covering ordinary garment changes. There's a
test asserting that ordinary garment editing keeps working, so this boundary
can't be enforced by over-blocking.

Both sanitisers strip undressing directives arriving through free-text fields,
and `test_no_undress_control_exists` asserts no such term appears anywhere in
the option banks. Please leave those tests in place.

This is the one place where image mode deliberately lacks parity with video
mode. Everything else in this document is built for parity.

---

## 14. Suggested order of work

1. Drop in the package and tests; get CI green **including both wall tests**.
2. Run `python demo.py` to see every behaviour without a server.
3. Write the adapter if `LlamaClient` needs one. Confirm one real end-to-end
   generation before touching any UI.
4. Register the mode; bare page, generate task, default controls.
5. Build the controls from `option_banks()`, with all three sentinels.
6. Wire the five reactive behaviours in §9.3 — this is where bugs will be.
7. Add the edit task: task selector, edit controls, reference roles.
8. Surface `chosen` / `chosen_source` beside the dropdowns. CHOOSE is much less
   useful when you can't see what it picked.
9. Reference image attachment, touch sizing, copy/save.
10. Re-verify §12 against current model cards and correct the profile constants.
