"""
image_engine.selection — resolving CHOOSE and RANDOM.

Two sentinels need resolving before a brief can be built.

RANDOM is a seeded draw. Each control gets its own RNG derived from the global
seed and a stable hash of the control's name, rather than sharing one sequential
stream. That matters: with a shared stream, adding a new control shifts every
draw that comes after it, so last week's seed stops reproducing last week's
image. Per-control derivation makes each draw independent of the others.

CHOOSE asks the local model to pick the best-fitting option for the user's
intent. Three things make that reliable on a 9-27B model:

  1. One batched call for several controls, not one call per control, and
     capped at a handful of controls per call. Accuracy on constrained choice
     degrades as the number of simultaneous fields grows.
  2. The option list is shuffled per call, seeded, and the model must return
     the exact option string rather than a letter or an index. Language models
     have a measurable bias toward particular positions in an enumerated list;
     shuffling turns a systematic bias into noise, and exact-string output
     removes the letter-token bias entirely.
  3. An explicit escape value. A model forced to choose when nothing fits will
     invent something or pick the first item. Given somewhere to say "nothing
     here fits", it says so, and the control falls back to AUTO.

Everything the model returns is validated against the bank. Anything invalid,
missing, or unparseable falls through to a deterministic keyword scorer that
needs no model at all — so CHOOSE still works with the server down, just less
well.
"""

from __future__ import annotations

import json
import random
import re
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .options import SELECTION_CUES

__all__ = [
    "NO_PREFERENCE",
    "SelectionOutcome",
    "seeded_rng_for",
    "random_pick",
    "heuristic_pick",
    "SelectionPass",
]

NO_PREFERENCE = "__none__"
"""The escape value. A model that must always choose will choose badly."""

MAX_CONTROLS_PER_CALL = 6
"""Constrained selection degrades as simultaneous fields grow. Six is
conservative; raise it only with measurements to back the change."""

_STOPWORDS = frozenset(
    """a an the of in on at to for with and or but from by as is are was were
    be been being it its this that these those there here shot lit palette
    style look mood into onto over under across through""".split()
)


# ---------------------------------------------------------------------------
# Seeded randomness
# ---------------------------------------------------------------------------


def seeded_rng_for(seed: int, control: str) -> random.Random:
    """An RNG for one control, derived from the global seed.

    Uses crc32 rather than the builtin ``hash`` because ``hash`` is salted per
    process for strings, which would make "reproducible" seeds reproduce only
    within a single run of the application.
    """
    salt = zlib.crc32(control.encode("utf-8")) & 0xFFFFFFFF
    return random.Random((seed ^ salt) & 0xFFFFFFFF)


def random_pick(seed: int, control: str, bank: Sequence[str]) -> str:
    """Draw one option for a control, reproducibly for a given seed."""
    if not bank:
        return ""
    rng = seeded_rng_for(seed, control)
    return bank[rng.randrange(len(bank))]


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z'-]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


# Evidence is not all equal. A curated cue is a deliberate signal that this
# option is what someone means when they say that word; a token mined from the
# option's own description is weak background noise. Weighting them the same
# lets a long description drown out a precise keyword — which is exactly how
# "please delete the bins" failed to resolve to a removal.
_W_CUE_PHRASE = 5.0
_W_CUE_WORD = 3.0
_W_CUE_WEAK = 1.5
_W_NAME = 2.0
_W_DESCRIPTION = 0.5
_SCORE_THRESHOLD = 2.0

_WEAK_CUE_WORDS = frozenset(
    """dark light bright dim old new same big small large tiny colour color
    warm cool soft hard""".split()
)
"""Cues that are real signals but ambiguous enough to lose to a specific noun.
"after dark" should not beat "campfire" for a firelit scene. Deliberately
excludes operation verbs like put, add and remove — those are the whole signal
for an edit operation and must keep full weight."""


def _score_option(
    intent_l: str, intent_tokens: set[str], option: str, description: str
) -> float:
    score = 0.0

    for cue in SELECTION_CUES.get(option, ()):
        cue_l = cue.lower()
        if " " in cue_l:
            if cue_l in intent_l:
                score += _W_CUE_PHRASE
        elif cue_l in intent_tokens:
            score += _W_CUE_WEAK if cue_l in _WEAK_CUE_WORDS else _W_CUE_WORD

    for token in _tokens(option):
        if token in intent_tokens:
            score += _W_CUE_WEAK if token in _WEAK_CUE_WORDS else _W_NAME

    for token in _tokens(description) - _tokens(option):
        if token in intent_tokens:
            score += _W_DESCRIPTION

    return score


def heuristic_pick(intent: str, bank: Mapping[str, str]) -> str:
    """Pick the best-fitting option using keyword overlap alone.

    No model, no network, fully deterministic. This is what runs when the
    selection call fails, returns something invalid, or was never made.

    Returns ``""`` when nothing clears the threshold — which the caller treats
    as AUTO, since a bad guess is worse than saying nothing. Ties break toward
    the earlier option in the bank, so the result is stable.
    """
    intent_l = intent.lower()
    intent_tokens = _tokens(intent)
    if not intent_tokens:
        return ""

    best, best_score = "", 0.0
    for option, description in bank.items():
        score = _score_option(intent_l, intent_tokens, option, description)
        if score > best_score:
            best, best_score = option, score

    return best if best_score >= _SCORE_THRESHOLD else ""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SelectionOutcome:
    """What the selection pass resolved, and how.

    ``source`` maps each control to "model", "heuristic", "random" or "none",
    so the UI can show the user where a value came from. Surfacing that is the
    point of CHOOSE — a hidden choice is no better than AUTO.
    """

    values: dict[str, str] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def resolved(self) -> dict[str, str]:
        return {k: v for k, v in self.values.items() if v}


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

_SELECT_SYSTEM = """\
You choose settings for an image prompt. You are given the user's intent and
several named settings, each with a fixed list of allowed values.

Return a single JSON object. Each key is a setting name. Each value is either
one of that setting's allowed values, copied EXACTLY character for character,
or the string "__none__" if no listed value genuinely fits the intent.

Rules:
- Copy values exactly. Do not paraphrase, reword, shorten or re-case them.
- Do not invent values that are not in the list.
- Choose on the merits of the intent, not on an option's position in the list.
- Prefer "__none__" over a weak fit. A wrong setting is worse than no setting.
- Output only the JSON object. No explanation, no markdown, no code fence.
"""


class SelectionPass:
    """Resolves CHOOSE controls by asking the model, with a local fallback."""

    def __init__(self, client=None) -> None:
        self._client = client

    def resolve(
        self,
        intent: str,
        requests: Mapping[str, Mapping[str, str]],
        *,
        seed: int,
        temperature: float = 0.25,
        use_model: bool = True,
    ) -> SelectionOutcome:
        """Choose a value for each requested control.

        ``requests`` maps control name to that control's bank. Returns a
        :class:`SelectionOutcome` whose ``values`` are guaranteed to be either
        a member of the corresponding bank or ``""``.
        """
        outcome = SelectionOutcome()
        if not requests:
            return outcome

        pending = dict(requests)

        if use_model and self._client is not None:
            for batch in self._batches(pending):
                try:
                    picked = self._ask(intent, batch, seed=seed, temperature=temperature)
                except Exception as exc:  # noqa: BLE001 - never lose a generation
                    outcome.notes.append(
                        f"Selection call failed ({type(exc).__name__}); "
                        "fell back to keyword matching."
                    )
                    continue
                for control, value in picked.items():
                    if value and value in requests[control]:
                        outcome.values[control] = value
                        outcome.source[control] = "model"
                    elif value == NO_PREFERENCE:
                        outcome.values[control] = ""
                        outcome.source[control] = "none"

        # Anything the model did not settle falls to the deterministic scorer.
        for control, bank in requests.items():
            if control in outcome.source:
                continue
            guess = heuristic_pick(intent, bank)
            outcome.values[control] = guess
            outcome.source[control] = "heuristic" if guess else "none"

        return outcome

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _batches(
        requests: Mapping[str, Mapping[str, str]]
    ) -> Iterable[dict[str, Mapping[str, str]]]:
        items = list(requests.items())
        for i in range(0, len(items), MAX_CONTROLS_PER_CALL):
            yield dict(items[i : i + MAX_CONTROLS_PER_CALL])

    def _ask(
        self,
        intent: str,
        batch: Mapping[str, Mapping[str, str]],
        *,
        seed: int,
        temperature: float,
    ) -> dict[str, str]:
        lines = [f"INTENT\n{intent.strip()}\n", "SETTINGS"]

        for control, bank in batch.items():
            options = list(bank)
            # Shuffle per control, seeded, to blunt position bias.
            seeded_rng_for(seed, f"select:{control}").shuffle(options)
            lines.append(f"\n{control}:")
            for option in options:
                hint = bank[option]
                lines.append(f"  - {option}" + (f"  ({hint})" if hint != option else ""))

        lines.append(
            '\nReturn JSON with exactly these keys: '
            + ", ".join(f'"{c}"' for c in batch)
        )

        raw = "".join(
            self._client.stream_chat(
                system=_SELECT_SYSTEM,
                user="\n".join(lines),
                temperature=temperature,
                top_p=0.9,
                seed=seed,
            )
        )
        return self._parse(raw, batch)

    @staticmethod
    def _parse(raw: str, batch: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
        """Pull a control->value mapping out of whatever the model returned."""
        text = re.sub(r"```[a-z0-9]*", "", raw or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}

        out: dict[str, str] = {}
        for control, bank in batch.items():
            value = data.get(control)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value == NO_PREFERENCE:
                out[control] = NO_PREFERENCE
            elif value in bank:
                out[control] = value
            else:
                # Tolerate case and whitespace drift, but nothing looser --
                # a fuzzy match here is how invented options get through.
                folded = {k.casefold(): k for k in bank}
                match = folded.get(value.casefold())
                if match:
                    out[control] = match
        return out
