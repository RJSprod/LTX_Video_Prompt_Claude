# Upstream diff notes

Every difference between a vendored module under
`src/prompt_master/prompt_engine/upstream/` and the corresponding file in the
pinned Prompt-Master-LD source must be declared here. `tools/check_upstream_sync.py`
parses this file and fails on any undeclared change.

## Status

**15 modules vendored. 14 are byte-identical to upstream. One approved
difference, in `imaging.py`.**

| Module | Role | State |
| --- | --- | --- |
| `brain.py` | engine | byte-identical |
| `accents.py` | engine | byte-identical |
| `music.py` | engine | byte-identical |
| `styles.py` | engine | byte-identical |
| `cinematics.py` | engine | byte-identical |
| `hands.py` | engine | byte-identical |
| `identity.py` | engine | byte-identical |
| `wardrobe.py` | engine | byte-identical |
| `negative.py` | engine | byte-identical |
| `shotscript.py` | engine | byte-identical |
| `imaging.py` | engine | **1 approved difference** |
| `node.py` | reference | byte-identical |
| `routes.py` | reference | byte-identical |
| `backend.py` | reference | byte-identical |
| `selftest.py` | reference | byte-identical |

The ten engine modules that carry prompt text needed **no** edits at all.
Upstream already used package-relative imports (`from .accents import …`) and
none of those ten import ComfyUI, Torch or aiohttp, so the port is a copy.

## What does not need an entry

The sync checker auto-approves hunks in which every changed line is an import
statement or a blank/comment line. These are the packaging edits the porting
contract permits:

- rewriting an absolute import as a package-relative import,
- deleting a ComfyUI import,
- deleting a Torch import that only served tensor conversion,
- adding `from __future__ import annotations` or a typing-only import.

Everything else needs an entry, including any added type annotation, because an
annotation edit touches the same lines as behavior.

## Entry format

The parser expects a heading naming the module, followed by four fields. Line
numbers are 1-based and inclusive.

```
### accents.py
- Original lines: 12-14
- New lines: 12-16
- Reason: <what changed and why the porting contract permits it>
- Behavior unchanged: yes
```

`Behavior unchanged: yes` is a required confirmation, not a formality: it
asserts that `tools/compare_upstream_engine.py` produces identical system
prompts, user prompts, base negatives, budgets and frame counts across this
change. A hunk whose entry lacks the confirmation fails the check.

## Approved differences

### imaging.py

- Original lines: 50-58
- New lines: 49-49
- Reason: removed `pil_to_tensor()` and `black()`, the two ComfyUI tensor
  helpers. Both exist only to hand ComfyUI an `IMAGE` tensor — `pil_to_tensor`
  wraps a PIL image as a `torch.Tensor`, and `black` allocates a blank tensor
  for the T2V pack. Their only caller is `node.py`, which is vendored as a
  reference and never imported by the standalone application; the app carries
  Pillow images end to end and has no sampler to hand a tensor to. The porting
  contract permits exactly this ("Remove Torch tensor conversion where the
  standalone app uses Pillow", "Do not require Torch for normal image input").
  The accompanying `import torch` removal is auto-approved as an import change.
  `numpy` is deliberately kept: `style_hint()` uses it for the flat-cel
  detector that decides whether the I2V brief gets its hardened cartoon medium
  law, which is prompt behavior, not tensor plumbing.
- Behavior unchanged: yes

Verified: `tools/compare_upstream_engine.py` compares 307 configurations across
3,767 field comparisons with zero mismatches, and `tests/test_upstream_parity.py`
runs 433 upstream assertions green. Neither removed function appears anywhere in
the prompt-building path — `brain.py` does not import `imaging` at all.

## Additions

These files are new to the port. They are not modified upstream files, so they
carry no diff, but they are listed so nothing in the vendored tree is unaccounted
for.

### upstream/__init__.py

Upstream's own `__init__.py` registers ComfyUI nodes and imports `node.py`,
which pulls in ComfyUI and Torch. It is deliberately **not** vendored. The
replacement is a docstring only — it re-exports nothing, so the upstream
relative imports keep resolving exactly as they did in the ComfyUI package.

## Deliberate non-changes

Worth recording, because each looks like something a porter would "tidy":

- **`node.py`, `routes.py`, `backend.py` and `selftest.py` still import ComfyUI,
  aiohttp and Torch.** They are references, held byte-identical on purpose.
  Nothing in the application imports them. `routes.py` is additionally read as
  *data* by the ported self-test, which execs its `_prompt_estimate` function —
  so editing it would silently change a test.
- **`shotscript.py` keeps all three formats** (`flowing`, `bracket`,
  `shotscript`). The port brief lists two ("flowing prose" and "shot script");
  upstream has three, and the no-reduction rule wins over the brief's count.
- **`identity.py` keeps `_ACCENT_REGION` keys that no accent uses** (`pakistani`,
  `ukrainian`, `mexican`, and others). They are unreachable from
  `accents.ACCENT_KEYS`, but removing dead entries is still a dictionary edit
  and the fallback path (`region_for` → `("global", "")`) depends on which keys
  are absent.
- **`brain.py` keeps `max_tokens`'s unused `fmt` parameter.** Its docstring says
  it is retained for call-site compatibility and no longer changes the answer.
  The adapter passes it anyway.
