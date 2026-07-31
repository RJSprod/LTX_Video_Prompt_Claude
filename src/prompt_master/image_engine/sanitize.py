"""
image_engine.sanitize — cleaning what the writer actually returned.

The writing model is instructed not to do any of this and mostly does not. This
is the belt to that braces: a local 9-27B model will periodically open with
"Here's the prompt:", wrap the answer in a fence, or slip in a ``(word:1.3)``
it learned from Stable Diffusion prompts. Each of those measurably damages
output on both target models.

TWO SANITISERS, AND WHY
-----------------------
Generation and editing need different cleaning, and using the wrong one is
destructive rather than merely unhelpful.

In a *generation* prompt, "a quiet street, no cars, without any people" is
harmful — these models cannot subtract, so naming a thing puts it in frame.
Negations are stripped.

In an *edit instruction*, negation is load-bearing, and not mainly because of
removal verbs: "Remove the bins" survives either sanitiser, since "remove" is
not a negation word. The real casualty is the **preservation clause**, which is
very often phrased negatively — "without altering the background", "nothing
else changes", "avoiding any change to the brickwork", "free of any
watermark". Those clauses are the second half of a good edit instruction and
the main defence against unintended global change. Run the generation
sanitiser over an edit and it deletes exactly the part that was holding the
image together.

:func:`sanitize_edit_instruction` therefore leaves negation alone entirely, and
instead checks that a preservation clause is present at all.
"""

from __future__ import annotations

import json
import re
import unicodedata

from .profiles import ModelProfile, PromptShape

__all__ = [
    "UNIVERSAL_AVOID",
    "sanitize_positive",
    "sanitize_edit_instruction",
    "word_count",
]


_FENCE_RE = re.compile(r"```[a-z0-9+#.-]*(?=\s|$)|```")
"""Unanchored on purpose: a small local model often writes a preamble and then
opens the fence, so an anchored pattern would leave the backticks behind.

The language tag is restricted to lowercase and must end at whitespace. A naive
``[a-zA-Z]*`` silently eats the first word of "```A weathered fisherman" — the
tag matcher cannot be allowed to consume prose."""

_WEIGHT_RE = re.compile(r"\(\s*([^():]+?)\s*:\s*[\d.]+\s*\)")
_BRACKET_WEIGHT_RE = re.compile(r"[\[\{]([^\]\}]+)[\]\}]")

_LEAD_LABEL_RE = re.compile(
    r"^\s*(?:prompt|positive prompt|image prompt|edit instruction|instruction"
    r"|output|here(?:'s| is)[^:]*)\s*[:\-]\s*",
    re.IGNORECASE,
)

_NEGATION_RE = re.compile(
    r"(?:^|[.;,]\s*)(?:no|nothing|none of|without|avoid(?:ing)?|free of|"
    r"free from|devoid of|lacking|excluding|minus|not?\s+any|never)\s+"
    r"[^.;,]{2,60}",
    re.IGNORECASE,
)

_DISALLOWED_RE = re.compile(
    r"\b(?:undress(?:ed|ing)?|strip(?:ped|ping)?\s+(?:naked|nude)|nude|naked|"
    r"topless|nsfw|see-through\s+to\s+skin)\b",
    re.IGNORECASE,
)
"""The image engine has no undress control by design. This catches a directive
arriving through one of the free-text fields."""

UNIVERSAL_AVOID: tuple[str, ...] = (
    "masterpiece",
    "best quality",
    "award winning",
    "trending on artstation",
    "4k",
    "8k",
    "ultra hd",
    "highly detailed",
    "stunning",
    "breathtaking",
    "flawless",
    "perfect lighting",
)
"""Words no image model benefits from. Applied on top of whatever the profile
lists, so a new profile inherits the floor without restating it."""


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w[\w'-]*\b", text))


def _strip_disallowed(text: str) -> str:
    """Remove sexualising directives, by sentence rather than by word.

    Word-level removal would leave a sentence with a hole in it; sentence-level
    keeps the result readable.
    """
    if not _DISALLOWED_RE.search(text):
        return text
    kept = [
        s for s in re.split(r"(?<=[.!?])\s+", text) if not _DISALLOWED_RE.search(s)
    ]
    return " ".join(kept).strip()


def _normalise_punctuation(text: str) -> str:
    """Clean up the debris left behind by word and phrase removals.

    Removing a term mid-sentence leaves ", ," and "fog,." behind, which the
    model reads as structure.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"(?:\s*,\s*)+", ", ", text)
    text = re.sub(r"[,;]\s*([.;!?])", r"\1", text)
    text = re.sub(r"([.;])\s*\1+", r"\1", text)
    text = re.sub(r"^[,;.\s]+", "", text)
    text = re.sub(r"[,;\s]+$", "", text)
    return text


def _recapitalise(text: str) -> str:
    """Removing a word from the head of a sentence leaves it lowercase."""
    return re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )


def _strip_wrapper(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _FENCE_RE.sub("", text).strip()
    text = _LEAD_LABEL_RE.sub("", text)
    return text.strip().strip('"').strip()


def _trim_to_ceiling(text: str, ceiling: int) -> str:
    words = text.split()
    if len(words) <= ceiling:
        return text
    clipped = " ".join(words[:ceiling])
    cut = max(clipped.rfind(". "), clipped.rfind("; "))
    return clipped[: cut + 1] if cut > len(clipped) // 2 else clipped


def sanitize_positive(
    text: str,
    profile: ModelProfile,
    *,
    shape: PromptShape = PromptShape.PROSE,
) -> str:
    """Clean a streamed generation prompt.

    Strips preambles and fences, removes attention weights while keeping the
    word, deletes negations, drops quality slop, normalises the punctuation the
    removals leave behind, enforces the hard word ceiling on a sentence
    boundary, and restores capitalisation.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _FENCE_RE.sub("", text).strip()

    # JSON is detected whatever shape was asked for. A model that returns a
    # valid object when prose was requested has still returned something
    # coherent, and prose cleanup would only corrupt it.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    text = _LEAD_LABEL_RE.sub("", text).strip().strip('"').strip()

    if not profile.honours_attention_weights:
        text = _WEIGHT_RE.sub(r"\1", text)
        text = _BRACKET_WEIGHT_RE.sub(r"\1", text)

    text = _strip_disallowed(text)
    text = _NEGATION_RE.sub(
        lambda m: m.group(0)[:1] if m.group(0)[:1] in ".;," else "", text
    )

    for word in (*UNIVERSAL_AVOID, *profile.avoid_vocabulary):
        text = re.sub(rf"\b{re.escape(word)}\b[,;]?\s*", "", text, flags=re.IGNORECASE)

    text = _normalise_punctuation(text)
    text = _trim_to_ceiling(text, profile.hard_word_ceiling)

    if text and text[-1] not in ".!?":
        text += "."
    return _recapitalise(text)


_IMPERATIVE_HINTS = (
    "add", "remove", "replace", "change", "make", "swap", "turn", "put",
    "move", "delete", "convert", "recolour", "recolor", "extend", "restore",
    "colourise", "colorize", "adjust", "set", "give", "place", "transform",
    "relight", "sharpen", "enhance", "keep", "preserve", "take",
)


def sanitize_edit_instruction(text: str, profile: ModelProfile) -> tuple[str, list[str]]:
    """Clean a streamed edit instruction.

    Deliberately does NOT strip negations. "Remove the parked cars" is the
    instruction, not a mistake — running the generation sanitiser here would
    delete the request itself.

    Returns ``(instruction, warnings)``. Warnings are surfaced to the user
    rather than acted on, because an edit instruction that fails a heuristic is
    usually still worth showing.
    """
    warnings: list[str] = []
    text = _strip_wrapper(text)

    if not profile.honours_attention_weights:
        text = _WEIGHT_RE.sub(r"\1", text)
        text = _BRACKET_WEIGHT_RE.sub(r"\1", text)

    text = _strip_disallowed(text)

    # Quality slop is as useless in an edit instruction as in a prompt.
    for word in (*UNIVERSAL_AVOID, *profile.avoid_vocabulary):
        text = re.sub(rf"\b{re.escape(word)}\b[,;]?\s*", "", text, flags=re.IGNORECASE)

    text = _normalise_punctuation(text)
    text = _trim_to_ceiling(text, profile.edit_max_words * 2)

    if text and text[-1] not in ".!?":
        text += "."
    text = _recapitalise(text)

    # Heuristics, reported rather than enforced.
    first = text.split(maxsplit=1)[0].strip(".,").lower() if text else ""
    if first and first not in _IMPERATIVE_HINTS:
        warnings.append(
            "The instruction does not open with an imperative verb. Editing "
            "models follow a direct command more reliably than a description."
        )

    if re.search(r"\b(?:her|his|its|their|them|it)\b", text, re.IGNORECASE):
        if not re.search(r"\b(?:the|that)\s+\w+", text):
            warnings.append(
                "The instruction leans on a pronoun. Naming the element "
                "directly is what scopes the edit."
            )

    lowered = text.lower()
    if not any(
        k in lowered
        for k in ("keep", "preserv", "unchanged", "same", "maintain", "identical")
    ):
        warnings.append(
            "No preservation clause. These models drift on anything not "
            "explicitly held, so naming what must survive is worth a retry."
        )

    return text, warnings
