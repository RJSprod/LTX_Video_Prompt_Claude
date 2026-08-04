# Prompt Master LD — parity report

The standalone prompt engine is now a direct port of Prompt-Master-LD, not a
reimplementation. This report records what was imported, what is preserved, and
what was verified.

## Upstream revision

| | |
| --- | --- |
| git commit | **unavailable** — see below |
| source archive | `69d6cd54-PromptMasterLDmain_1.zip` |
| archive SHA-256 | `66deb4f6cc41257bee0a166f0998adf04ee8aa94fb624b769ec93b423fb65313` |
| archive root | `Prompt-Master-LD-main/` |
| received | 2026-07-26 |

`github.com/Brojakhoeman/Prompt-Master-LD` was unreachable from the build
environment: anonymous GitHub access works there (public controls resolve), but
every repository under that account returns 404 or demands credentials. The
source was supplied as a GitHub "Download ZIP" of `main`, which carries no
`.git` directory and therefore no commit SHA.

Because there is no revision to pin, **the content is pinned instead**:
`src/prompt_master/prompt_engine/UPSTREAM_COMMIT.txt` records the SHA-256 of all
20 upstream files as received, and `tools/check_upstream_sync.py` verifies any
future checkout against those digests before it compares a line.

**This is the one item in the brief's definition-of-complete that is not
satisfied as written**, and it cannot be satisfied without access to the
repository. If the repo becomes reachable, confirm its `main` matches these
digests and replace the `git-commit: UNKNOWN` line with the real SHA; nothing
else needs to change.

## Modules imported

**15 modules.** 11 engine, 4 reference.

Engine (imported by the application): `brain.py`, `accents.py`, `music.py`,
`styles.py`, `cinematics.py`, `hands.py`, `identity.py`, `wardrobe.py`,
`negative.py`, `shotscript.py`, `imaging.py`

Reference (vendored, never imported): `node.py`, `routes.py`, `backend.py`,
`selftest.py`, plus upstream `README.md`

**14 of 15 are byte-identical to upstream.** The ten engine modules that carry
prompt text needed no edits whatsoever: upstream already used package-relative
imports, and none of those ten import ComfyUI, Torch or aiohttp.

## Options preserved

**305 across 13 tables**, every one asserted equal to upstream by the harness.

| Table | Count |
| --- | --- |
| Accents | 47 |
| Accent strengths | 3 |
| English varieties (vs. second-language accents) | 26 |
| Music genres | 35 |
| Accent → music mappings | 47 |
| Visual styles | 20 |
| Style groups | 4 |
| Cameras | 10 |
| Camera negatives | 10 |
| Transitions | 10 |
| Output formats | 3 |
| Identity regions | 11 |
| Accent → region mappings | 79 |

The UI takes all of these from `prompt_engine/options.py`, which is built from
the upstream constants at import time. No control carries its own list.

For comparison, the engine this replaced had 6 accents, 20 invented styles, 20
invented cameras and a single 60-word system prompt.

## Assertions run

**510 tests, all passing.**

| Suite | Tests | What it covers |
| --- | --- | --- |
| `tests/test_upstream_parity.py` | 434 | The upstream self-test, ported. 433 upstream assertions + 1 coverage guard. |
| `tests/test_prompt_engine.py` | 26 | The adapter seam and the UI option sources. |
| `tests/test_core.py` | 6 | Pre-existing app tests. |
| `tests/test_install_flow.py` | 44 | The source distribution: install-root discovery, GPU sizing, manifest resolution, console setup. |

The first three suites are unchanged from the packaged build and cover the
engine; the fourth covers only what replacing the compiled installer added, and
touches no prompt text. The engine's own count is therefore still 466.

The 433 figure is assertions that *fire*: 418 `check(...)` call sites, several
inside loops that sweep every accent, style or music key. The count is
deterministic and pinned, so an early abort fails rather than silently shrinking
coverage.

## Parity harness

`tools/compare_upstream_engine.py` loads an untouched checkout and the port into
one process and drives both with identical inputs.

```
configs compared  : 307
smart-neg cases   : 5
field comparisons : 3767
result            : passed — output-identical
```

Compared per config: system prompt, user prompt, base negative, frame count,
token budget, word budget, beat budget, spoken-line count, write-seconds,
talk percentage, output-format contract, and which conditional blocks fired.
Plus all 71 option/law constant tables, and smart-negative filtering over shared
raw model output. Normalization is limited to line endings.

**The harness was verified against deliberate sabotage.** Each forbidden change
from the porting contract was injected into a copy of upstream:

| Injected change | Detected |
| --- | --- |
| Paraphrase a prompt-law string | ✅ 24 mismatches |
| Drop 3 accents from the dict | ✅ 2 mismatches |
| Shorten the music genre list | ✅ 3 mismatches |
| Reorder two negative-bank terms | ✅ 308 mismatches |
| Change beat-budget arithmetic | ✅ 879 mismatches |
| Remove a conditional branch | ✅ 2 mismatches |
| Swap a camera negative | ✅ 2 mismatches |
| *(control: no change)* | passes |

The "drop 3 accents" case initially passed, because the config matrix is
generated *from* the reference engine and simply never drove the missing keys.
The constant-table comparison exists because of that.

## Approved differences

**One.** Fully documented in `UPSTREAM_DIFF_NOTES.md`.

`imaging.py` — removed `pil_to_tensor()` and `black()`, plus the `import torch`
they needed. Both are ComfyUI tensor plumbing whose only caller is `node.py`
(vendored as reference, never imported). The standalone app carries Pillow
images end to end. `numpy` is kept deliberately: `style_hint()` uses it for the
flat-cel detector that decides whether an I2V brief gets its hardened cartoon
medium law, which is prompt behavior.

`brain.py` does not import `imaging` at all, so no prompt text can be affected.

## The second engine

`prompt_engine/minimax_h3.py` writes prompts for MiniMax-H3 and is **not** part
of the port. It is this repository's own code, written to MiniMax's published
`VIDEO_PROMPT_WRITING_GUIDE_base_en.md`, and it exists as a separate module
rather than as a mode inside `brain.py` for the reason this whole report is
about: every string upstream carries is pinned, and a second model's laws living
next to them is a change that would not show up until a prompt came out
different.

The isolation is asserted rather than intended. `tests/test_minimax_h3.py`:

* parses `minimax_h3.py` and fails on any import naming `upstream` or `adapter`;
* scans every module under `prompt_engine/upstream/` and fails if one of them
  mentions MiniMax at all.

`tools/check_upstream_sync.py` and `tools/compare_upstream_engine.py` read only
`prompt_engine/upstream`, so neither their file manifests nor the 307-config
comparison is affected by the new module. The counts in this report are the LTX
engine's and remain what they were.

## Verification commands

```bash
python -m pytest tests/                                       # 510 passed
python tools/compare_upstream_engine.py --upstream <checkout> # 0 mismatches
python tools/check_upstream_sync.py     --upstream <checkout> # 1 approved diff
```

## Not verified

The brief's final item asks that prompts generated by the standalone app match
the original engine "using the same inputs, seed, and model settings". That is
untestable here — it needs a GPU, the pinned `llama.cpp` build and the Gemma
weights, none of which exist in this environment.

What *is* verified is everything upstream of the model: for identical inputs the
two engines emit byte-identical system prompts, user prompts, negatives and
token budgets. Given the same model, quantization and sampler settings, the
generations follow. Note that this holds only if the standalone app runs the
same model upstream ran — the repo pins Gemma via `llama.cpp`; if upstream's
ComfyUI node was pointed at different weights, the prompts still match exactly
but the generations will not.
