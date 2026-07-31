"""
image_engine.negative — the gated negative-prompt assembler.

Only one of the three supported profiles has a negative-conditioning path at
all. :func:`build_negative` returns an empty string for the other two, and that
is the correct answer rather than a limitation being papered over: a populated
field on a model that discards it gives the user a false sense of control,
which is worse than no field.

Where a negative IS used, the banks are gated on what the brief actually
contains. An unconditional negative is a negative that fights the prompt it is
attached to: telling a woodblock print to avoid "sensor noise" spends
conditioning on nothing, and telling a photograph to avoid "flat shading" can
flatten it.
"""

from __future__ import annotations

import re
from typing import Iterable

from .brief import ImageBrief, mentions_people

__all__ = ["NEGATIVE_BANKS", "BANK_PRIORITY", "gates_for", "build_negative"]


NEGATIVE_BANKS: dict[str, tuple[str, ...]] = {
    "anatomy": (
        "deformed hands",
        "extra fingers",
        "fused fingers",
        "missing fingers",
        "malformed limbs",
        "extra limbs",
        "twisted joints",
        "asymmetric eyes",
        "misaligned pupils",
    ),
    "face": (
        "warped facial features",
        "uncanny expression",
        "melted features",
        "doubled iris",
    ),
    "ai_look": (
        "plastic skin",
        "waxy skin",
        "airbrushed",
        "over-retouched",
        "oversaturated",
        "overly glossy",
        "HDR halo",
        "clarity slider",
    ),
    "photographic_faults": (
        "blown highlights",
        "crushed blacks",
        "chromatic aberration",
        "heavy vignette",
        "motion blur",
        "out of focus",
        "sensor noise",
        "banding",
    ),
    "illustration_faults": (
        "muddy linework",
        "inconsistent line weight",
        "flat lifeless shading",
        "misregistered colour",
    ),
    "render_faults": (
        "aliased edges",
        "z-fighting",
        "untextured surfaces",
        "floating geometry",
        "raytracing fireflies",
    ),
    "typography": (
        "garbled text",
        "misspelled words",
        "nonsense lettering",
        "duplicated letters",
    ),
    "composition": (
        "cropped head",
        "cut-off limbs at the frame edge",
        "cluttered background",
        "tangent lines",
        "accidental symmetry",
    ),
    "provenance": (
        "watermark",
        "signature",
        "logo",
        "border",
        "frame",
        "caption text",
        "jpeg artefacts",
    ),
    "duplication": (
        "duplicated subject",
        "cloned faces in the crowd",
        "repeating pattern artefacts",
        "tiling seams",
    ),
}

BANK_PRIORITY: tuple[str, ...] = (
    "anatomy",
    "face",
    "ai_look",
    "typography",
    "duplication",
    "composition",
    "photographic_faults",
    "illustration_faults",
    "render_faults",
    "provenance",
)
"""Emission order. Not alphabetical: when the term ceiling truncates the list,
the banks a human eye notices first must survive the cut."""


def gates_for(brief: ImageBrief, positive: str = "") -> set[str]:
    """Decide which banks apply to this particular image."""
    gates: set[str] = {"provenance"}
    group = brief.style_group
    text = f"{brief.intent}\n{positive}".lower()

    if brief.mentions_people or mentions_people(positive):
        gates.update({"anatomy", "face"})

    if group == "Photographic" or not group:
        gates.update({"photographic_faults", "ai_look"})
    elif group == "Illustration":
        gates.add("illustration_faults")
    elif group == "Rendered":
        gates.update({"render_faults", "ai_look"})
    elif group == "Painterly":
        gates.add("illustration_faults")

    if brief.render_text:
        gates.add("typography")

    if any(w in text for w in ("crowd", "many", "group", "row of", "pattern", "repeat")):
        gates.add("duplication")

    if brief.framing or any(
        w in text for w in ("portrait", "close-up", "framed", "composition")
    ):
        gates.add("composition")

    return gates


def build_negative(
    brief: ImageBrief,
    positive: str = "",
    extra_from_refine: Iterable[str] = (),
) -> str:
    """Assemble the negative prompt, or return empty when none is supported."""
    profile = brief.profile
    if not profile.negative_supported:
        return ""

    terms: list[str] = []
    seen: set[str] = set()

    def add(items: Iterable[str]) -> None:
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                terms.append(item.strip())

    gates = gates_for(brief, positive)
    for gate in BANK_PRIORITY:
        if gate in gates:
            add(NEGATIVE_BANKS.get(gate, ()))

    add(extra_from_refine)

    if brief.extra_negative:
        add(t for t in re.split(r"[,\n]", brief.extra_negative) if t.strip())

    if profile.negative_max_terms:
        terms = terms[: profile.negative_max_terms]

    return ", ".join(terms)
