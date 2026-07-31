"""
image_engine.edit — instruction-based image editing.

Edit mode is a different job from generation, not a variation on it. A
generation prompt describes an image that does not exist. An edit instruction
describes a *change* to an image that does, which means it has to carry three
things a generation prompt never does:

  1. **Which element.** "Remove the guy" fails when there are three people.
     The target must be pinned down by position, colour, size or relationship
     before anything else can work.
  2. **What survives.** These models change what they are told to change and
     drift on everything left unmentioned. Preservation is stated, not assumed.
     This is the single biggest quality lever in edit mode.
  3. **Physical consistency.** A new object needs the scene's light direction,
     its perspective and a contact shadow, or it reads as a sticker.

MODEL SUPPORT
-------------
FLUX.2 [klein] does instruction editing natively, with up to 4 reference
images addressed by number *and* role. It has no masked inpainting: a local
edit is scoped by language, not by a mask, which is exactly why target
disambiguation carries so much weight.

Krea 2 does not ship with editing at all — Krea's own technical report lists it
as future work. Editing on those weights exists only through a third-party
LoRA and custom nodes, so the profile marks it EXPERIMENTAL and the UI must say
so. Edit mode defaults to FLUX only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .options import (
    AUTO,
    CHOOSE,
    EDIT_OPERATIONS,
    EDIT_STRENGTHS,
    LOCALIZATIONS,
    PRESERVATION_TARGETS,
    RANDOM,
    REFERENCE_ROLES,
    SELECTABLE_BANKS,
    is_set,
)
from .profiles import DEFAULT_PROFILE_KEY, EditCapability, ModelProfile, profile_for
from .selection import SelectionOutcome, SelectionPass, random_pick

__all__ = [
    "ReferenceImage",
    "EditControls",
    "EditBrief",
    "resolve_edit_sentinels",
    "build_edit_brief",
]


@dataclass
class ReferenceImage:
    """One attached reference, and what it is for.

    ``index`` is 1-based because that is how the prompt addresses it: FLUX.2
    resolves "image 1", "image 2" and so on, and Black Forest Labs' own
    guidance lists forgetting to number the references as a common failure. The
    role is carried alongside so the instruction can say both — "the subject
    from image 1" rather than either half on its own.
    """

    index: int
    role: str = "Subject"
    note: str = ""

    def phrase(self, addressing: str = "index_and_role") -> str:
        role = REFERENCE_ROLES.get(self.role, self.role.lower())
        if addressing == "index_and_role":
            return f"{role} from image {self.index}"
        return role


@dataclass
class EditControls:
    """The state of every control on the Image Prompt page, edit task."""

    profile_key: str = "flux_klein_9b"
    allow_experimental_edit: bool = False
    """Krea 2 editing needs third-party components. Off by default, and the UI
    must explain what is required before turning it on."""

    operation: str = CHOOSE
    """Which kind of change. Defaults to CHOOSE because the user's request
    usually says plainly enough — "remove the bins" is a removal."""

    target_element: str = ""
    """What to act on, in the user's words. Left empty, the writer infers it
    from the request, which is less reliable when the scene has several
    similar objects."""

    localization: str = AUTO
    preservation: tuple[str, ...] = (
        "Identity",
        "Composition",
        "Lighting",
        "Style",
    )
    """Multi-select. These four are the defaults because they are what drifts
    most often and cost least to state."""

    edit_strength: str = "Balanced"
    match_lighting: bool = True
    match_perspective: bool = True
    seamless_blend: bool = True

    references: tuple[ReferenceImage, ...] = ()

    word_target: int = 0
    iterative_hint: bool = True
    """Add a note when the request looks like several edits at once. These
    models degrade over compound instructions and do better with a chain of
    small ones."""

    seed: int = -1

    def resolved_profile(self) -> ModelProfile:
        return profile_for(self.profile_key)

    def sentinel_controls(self, sentinel: str) -> list[str]:
        return [
            name
            for name in ("operation", "localization", "edit_strength")
            if getattr(self, name, None) == sentinel and name in SELECTABLE_BANKS
        ]


@dataclass
class EditBrief:
    """The resolved edit instruction set handed to the writing model."""

    request: str
    profile: ModelProfile
    seed: int

    operation: str = ""
    operation_guidance: str = ""
    target_element: str = ""
    localization: str = ""
    preservation: tuple[str, ...] = ()
    preservation_clauses: tuple[str, ...] = ()
    strength: str = "balanced"

    match_lighting: bool = True
    match_perspective: bool = True
    seamless_blend: bool = True

    references: tuple[ReferenceImage, ...] = ()
    reference_addressing: str = "index_and_role"

    word_target: int = 55
    denoise: float = 0.5
    steps: int = 24
    cfg: float = 1.0

    experimental: bool = False
    requirements: str = ""
    compound_warning: bool = False
    selection: SelectionOutcome | None = None

    def reference_phrases(self) -> list[str]:
        return [r.phrase(self.reference_addressing) for r in self.references]

    def facets(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.operation:
            out.append(("Operation", f"{self.operation} — {self.operation_guidance}"))
        if self.target_element:
            out.append(("Target", self.target_element))
        if self.localization:
            out.append(("Where", self.localization))
        if self.preservation_clauses:
            out.append(("Must survive", "; ".join(self.preservation_clauses)))
        if self.references:
            out.append(("References", "; ".join(self.reference_phrases())))
        physics = []
        if self.match_lighting:
            physics.append("light direction and shadows consistent with the scene")
        if self.match_perspective:
            physics.append("scale and perspective consistent with the scene")
        if self.seamless_blend:
            physics.append("edges integrated, no cut-out look")
        if physics:
            out.append(("Physical consistency", "; ".join(physics)))
        return out


# ---------------------------------------------------------------------------
# Compound-request detection
# ---------------------------------------------------------------------------

_COMPOUND_MARKERS = (
    " and also ",
    " then ",
    " as well as ",
    " plus ",
    "; ",
    " and then ",
)


def _looks_compound(request: str) -> bool:
    """Whether a request asks for several changes at once.

    Not a hard error — the writer still handles it — but these models degrade
    over compound instructions, so the user is told that a chain of small edits
    will hold identity better.
    """
    low = f" {request.lower().strip()} "
    hits = sum(low.count(m) for m in _COMPOUND_MARKERS)
    verbs = sum(
        low.count(f" {v} ")
        for v in ("add", "remove", "replace", "change", "make", "swap", "delete")
    )
    return hits >= 1 and verbs >= 2 or verbs >= 3


# ---------------------------------------------------------------------------
# Sentinel resolution
# ---------------------------------------------------------------------------


def resolve_edit_sentinels(
    request: str,
    controls: EditControls,
    seed: int,
    *,
    selection_pass: SelectionPass | None = None,
    use_model: bool = True,
) -> tuple[EditControls, SelectionOutcome]:
    """Turn CHOOSE and RANDOM on the edit controls into concrete values."""
    outcome = SelectionOutcome()
    updates: dict[str, str] = {}

    for name in controls.sentinel_controls(RANDOM):
        bank = SELECTABLE_BANKS[name]
        value = random_pick(seed, name, list(bank))
        updates[name] = value
        outcome.values[name] = value
        outcome.source[name] = "random"

    choose = controls.sentinel_controls(CHOOSE)
    if choose:
        requests = {name: SELECTABLE_BANKS[name] for name in choose}
        picker = selection_pass or SelectionPass(None)
        picked = picker.resolve(
            request,
            requests,
            seed=seed,
            temperature=controls.resolved_profile().select_temperature,
            use_model=use_model,
        )
        outcome.values.update(picked.values)
        outcome.source.update(picked.source)
        outcome.notes.extend(picked.notes)
        for name, value in picked.values.items():
            updates[name] = value or AUTO

    return replace(controls, **updates), outcome


# ---------------------------------------------------------------------------
# Brief construction
# ---------------------------------------------------------------------------


def build_edit_brief(
    request: str,
    controls: EditControls,
    seed: int,
    *,
    selection: SelectionOutcome | None = None,
) -> EditBrief:
    """Resolve an edit request and settled controls into an :class:`EditBrief`.

    Raises :class:`ValueError` when the chosen model cannot edit at all, or can
    only edit experimentally and the user has not opted in. That is deliberate:
    silently producing an instruction for a model that will ignore it wastes
    the user's time in a way that is hard to diagnose.
    """
    profile = controls.resolved_profile()

    if profile.edit_capability is EditCapability.NONE:
        raise ValueError(
            f"{profile.display_name} does not support instruction editing."
        )
    if profile.edit_is_experimental and not controls.allow_experimental_edit:
        raise ValueError(
            f"{profile.display_name} has no native editing. "
            f"{profile.edit_requirements} Enable the experimental edit option "
            "to proceed."
        )

    request = request.strip()

    brief = EditBrief(
        request=request,
        profile=profile,
        seed=seed,
        selection=selection,
        experimental=profile.edit_is_experimental,
        requirements=profile.edit_requirements if profile.edit_is_experimental else "",
    )

    if is_set(controls.operation) and controls.operation in EDIT_OPERATIONS:
        brief.operation = controls.operation
        brief.operation_guidance = EDIT_OPERATIONS[controls.operation]

    brief.target_element = controls.target_element.strip()
    brief.localization = (
        LOCALIZATIONS.get(controls.localization, "")
        if is_set(controls.localization)
        else ""
    )

    preservation = tuple(
        p for p in controls.preservation if p in PRESERVATION_TARGETS
    )
    brief.preservation = preservation
    brief.preservation_clauses = tuple(
        PRESERVATION_TARGETS[p] for p in preservation
    )

    strength_key = (
        controls.edit_strength
        if is_set(controls.edit_strength)
        else "Balanced"
    )
    brief.strength = EDIT_STRENGTHS.get(strength_key, "balanced")
    brief.denoise = {
        "preserve": profile.edit_denoise_preserve,
        "balanced": profile.edit_denoise_default,
        "transform": profile.edit_denoise_transform,
    }[brief.strength]

    brief.match_lighting = controls.match_lighting
    brief.match_perspective = controls.match_perspective
    brief.seamless_blend = controls.seamless_blend

    # Never promise the model more references than it can take.
    refs = tuple(controls.references)[: profile.max_reference_images]
    brief.references = tuple(
        replace(r, index=i + 1) for i, r in enumerate(refs)
    )
    brief.reference_addressing = profile.reference_addressing

    brief.word_target = profile.clamp_edit_words(
        controls.word_target or profile.edit_target_words
    )
    brief.steps = profile.edit_steps
    brief.cfg = profile.edit_cfg
    brief.compound_warning = controls.iterative_hint and _looks_compound(request)

    return brief
