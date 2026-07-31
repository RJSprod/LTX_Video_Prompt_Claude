"""
image_engine.brief — generation controls and the resolved brief.

Flow: :class:`ImageControls` is what the UI holds. :func:`resolve_sentinels`
turns every CHOOSE and RANDOM into a concrete value (or into nothing, if
CHOOSE found no good fit). :func:`build_brief` then turns the fully-resolved
controls into an :class:`ImageBrief`, which is what the prompt builders read.

The seed is fixed before any of this runs, and the same seed is later handed to
the sampler, so a reported seed reproduces the engine's choices *and* the
model's sampling. That is the same contract LTX Prompt mode offers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Mapping

from .options import (
    AGE_BANDS,
    APERTURES,
    ASPECT_RATIOS,
    AUTO,
    BUILDS,
    CAMERA_ANGLES,
    CHOOSE,
    COLOUR_PALETTES,
    COMPOSITION,
    FEATURE_HOOKS,
    FILM_STOCKS,
    LENSES,
    LIGHTING,
    MATERIALS,
    MOODS,
    RANDOM,
    RENDER_DETAIL,
    SELECTABLE_BANKS,
    SHOT_TYPES,
    SKIN_TEXTURE,
    STYLE_GROUP_OF,
    TIME_OF_DAY,
    VISUAL_STYLES,
    WARDROBE_REGISTERS,
    WEATHER,
    is_set,
)
from .profiles import DEFAULT_PROFILE_KEY, ModelProfile, PromptShape, profile_for
from .selection import SelectionOutcome, SelectionPass, random_pick, seeded_rng_for

__all__ = [
    "ImageControls",
    "ImageBrief",
    "resolve_sentinels",
    "build_brief",
    "resolve_dimensions",
    "mentions_people",
]


# ---------------------------------------------------------------------------
# Person detection
# ---------------------------------------------------------------------------

_PERSON_WORDS: tuple[str, ...] = (
    "person", "people", "man", "men", "woman", "women", "boy", "girl",
    "child", "children", "kid", "kids", "teenager", "adult", "elder",
    "figure", "figures", "crowd", "couple", "family", "group of",
    "portrait", "face", "hands", "someone", "somebody", "himself",
    "herself", "themselves",
    "worker", "labourer", "laborer", "welder", "blacksmith", "shipwright",
    "carpenter", "joiner", "mason", "plumber", "electrician", "mechanic",
    "engineer", "machinist", "miner", "fisherman", "fisherwoman", "farmer",
    "shepherd", "baker", "butcher", "chef", "cook", "barista", "waiter",
    "waitress", "bartender", "barber", "tailor", "seamstress", "weaver",
    "potter", "glassblower", "luthier", "watchmaker", "cobbler",
    "soldier", "sailor", "officer", "guard", "firefighter", "paramedic",
    "nurse", "doctor", "surgeon", "dentist", "vet", "scientist",
    "researcher", "teacher", "professor", "student", "librarian",
    "lawyer", "judge", "clerk", "accountant", "banker",
    "dancer", "musician", "singer", "cellist", "violinist", "pianist",
    "drummer", "guitarist", "conductor", "actor", "actress", "painter",
    "sculptor", "photographer", "writer", "poet", "architect", "designer",
    "athlete", "runner", "swimmer", "climber", "boxer", "player",
    "cyclist", "skater", "surfer", "rider", "jockey",
    "pilot", "driver", "captain", "navigator", "astronaut",
    "monk", "nun", "priest", "rabbi", "imam", "pilgrim",
    "vendor", "merchant", "trader", "shopkeeper", "grocer",
    "hunter", "ranger", "guide", "explorer", "traveller", "traveler",
    "beekeeper", "gardener", "florist", "fisher", "diver",
)

_PERSON_HINTS = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _PERSON_WORDS) + r")\b"
    r"|\b\w{3,}(?:man|woman|person|keeper|smith|wright|monger)\b"
    r"|\b(?:he|she|they|his|her|hers|their|theirs|him)\b",
    re.IGNORECASE,
)


def mentions_people(text: str) -> bool:
    """Whether a description implies at least one human figure.

    Gates the identity, wardrobe and skin-texture banks, and the anatomy
    negative bank. A trade name counts: scenes are usually populated by naming
    an occupation, not by saying "a person".
    """
    return bool(_PERSON_HINTS.search(text))


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


@dataclass
class ImageControls:
    """The state of every control on the Image Prompt page, generation task.

    Any field may carry a real value or one of the three sentinels:
    :data:`AUTO` (omit it), :data:`CHOOSE` (engine picks the best fit for the
    intent) or :data:`RANDOM` (seeded draw).
    """

    profile_key: str = DEFAULT_PROFILE_KEY

    # framing and optics
    shot_type: str = AUTO
    camera_angle: str = AUTO
    composition: str = AUTO
    lens: str = AUTO
    aperture: str = AUTO

    # light and air
    lighting: str = AUTO
    time_of_day: str = AUTO
    weather: str = AUTO

    # look
    visual_style: str = AUTO
    film_stock: str = AUTO
    colour_palette: str = AUTO
    use_hex_swatches: bool = False
    material_focus: str = AUTO
    mood: str = AUTO
    render_detail: str = "Balanced"

    # people
    wardrobe: str = AUTO
    concrete_identity: bool = True
    skin_texture: bool = True

    # text in image
    render_text: str = ""
    text_surface: str = ""

    # geometry
    aspect_ratio: str = "3:2 landscape"
    long_edge: int = 0

    # output shape
    prompt_shape: PromptShape = PromptShape.PROSE
    word_target: int = 0

    # negative (surfaced only when the profile supports it)
    extra_negative: str = ""
    smart_negative: bool = True

    # reference image
    has_reference_image: bool = False
    reference_role: str = "style"

    seed: int = -1

    def resolved_profile(self) -> ModelProfile:
        return profile_for(self.profile_key)

    def sentinel_controls(self, sentinel: str) -> list[str]:
        """Names of the controls currently set to a given sentinel."""
        return [
            name
            for name in SELECTABLE_BANKS
            if getattr(self, name, None) == sentinel
        ]


# ---------------------------------------------------------------------------
# Brief
# ---------------------------------------------------------------------------


@dataclass
class ImageBrief:
    """The resolved instruction set handed to the writing model."""

    intent: str
    profile: ModelProfile
    seed: int

    framing: list[str] = field(default_factory=list)
    optics: list[str] = field(default_factory=list)
    light: list[str] = field(default_factory=list)
    look: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    palette_prose: str = ""
    palette_hexes: tuple[str, ...] = ()
    mood: str = ""
    material: str = ""
    style_name: str = ""
    style_group: str = ""
    render_detail: str = ""

    render_text: str = ""
    text_surface: str = ""

    width: int = 1024
    height: int = 1024
    aspect_label: str = ""

    word_target: int = 80
    prompt_shape: PromptShape = PromptShape.PROSE

    has_reference_image: bool = False
    reference_role: str = "style"

    extra_negative: str = ""
    smart_negative: bool = False

    mentions_people: bool = False
    selection: SelectionOutcome | None = None

    def facets(self) -> list[tuple[str, str]]:
        """Ordered (label, text) pairs for the user turn."""
        out: list[tuple[str, str]] = []
        if self.framing:
            out.append(("Framing", "; ".join(self.framing)))
        if self.optics:
            out.append(("Optics", "; ".join(self.optics)))
        if self.light:
            out.append(("Light", "; ".join(self.light)))
        if self.style_name:
            out.append(("Medium", VISUAL_STYLES.get(self.style_name, self.style_name)))
        if self.look:
            out.append(("Look", "; ".join(self.look)))
        if self.palette_prose:
            palette = self.palette_prose
            if self.palette_hexes:
                palette += " (" + ", ".join(self.palette_hexes) + ")"
            out.append(("Palette", palette))
        if self.material:
            out.append(("Material", self.material))
        if self.people:
            out.append(("Figures", "; ".join(self.people)))
        if self.mood:
            out.append(("Mood", self.mood))
        if self.render_text:
            surface = self.text_surface or "a clearly readable surface"
            out.append(("Text in image", f'the words "{self.render_text}" on {surface}'))
        if self.render_detail:
            out.append(("Detail", self.render_detail))
        return out


# ---------------------------------------------------------------------------
# Sentinel resolution
# ---------------------------------------------------------------------------


def resolve_sentinels(
    intent: str,
    controls: ImageControls,
    seed: int,
    *,
    selection_pass: SelectionPass | None = None,
    use_model: bool = True,
) -> tuple[ImageControls, SelectionOutcome]:
    """Turn every CHOOSE and RANDOM into a concrete value.

    Returns a new controls object — the caller's is not mutated, so the UI can
    keep showing "Choose for me" while the resolved value is displayed beside
    it.

    RANDOM is drawn per control from the seed. CHOOSE is batched into as few
    model calls as possible, then validated, then falls back to keyword
    matching for anything the model did not settle. A CHOOSE that finds no good
    fit becomes AUTO rather than a bad guess.
    """
    outcome = SelectionOutcome()
    updates: dict[str, str] = {}

    # RANDOM first: it needs no model and cannot fail.
    for name in controls.sentinel_controls(RANDOM):
        bank = SELECTABLE_BANKS[name]
        value = random_pick(seed, name, list(bank))
        updates[name] = value
        outcome.values[name] = value
        outcome.source[name] = "random"

    # CHOOSE next, batched.
    choose = controls.sentinel_controls(CHOOSE)
    if choose:
        requests = {name: SELECTABLE_BANKS[name] for name in choose}
        picker = selection_pass or SelectionPass(None)
        picked = picker.resolve(
            intent,
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
# Geometry
# ---------------------------------------------------------------------------


def resolve_dimensions(
    aspect_label: str,
    long_edge: int,
    profile: ModelProfile,
) -> tuple[int, int]:
    """Turn an aspect ratio and a long edge into legal pixel dimensions.

    Both models want each edge on a multiple of 16 and cap total pixels. This
    clamps rather than raising: a UI spinner must not be able to produce an
    unusable request.
    """
    w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_label, (3, 2))
    edge = long_edge or profile.native_edge
    edge = max(profile.min_edge, min(edge, profile.max_edge))

    if w_ratio >= h_ratio:
        width, height = edge, round(edge * h_ratio / w_ratio)
    else:
        width, height = round(edge * w_ratio / h_ratio), edge

    def snap(v: int) -> int:
        m = profile.edge_multiple
        return max(m, min(int(round(v / m)) * m, profile.max_edge))

    width, height = snap(width), snap(height)

    while width * height > profile.max_pixels:
        width, height = snap(int(width * 0.92)), snap(int(height * 0.92))
        if width <= profile.min_edge and height <= profile.min_edge:
            break

    return width, height


# ---------------------------------------------------------------------------
# Brief construction
# ---------------------------------------------------------------------------


def _phrase(value: str, table: Mapping[str, str]) -> str:
    """The prose for a control value, or "" when it is unset or a sentinel."""
    return table.get(value, "") if is_set(value) else ""


def build_brief(
    intent: str,
    controls: ImageControls,
    seed: int,
    *,
    selection: SelectionOutcome | None = None,
) -> ImageBrief:
    """Resolve fully-settled controls into an :class:`ImageBrief`.

    Call :func:`resolve_sentinels` first. Any sentinel still present here is
    treated as AUTO, which means the facet is omitted — omission is meaningful,
    it is how the user says "you decide".
    """
    profile = controls.resolved_profile()
    intent = intent.strip()
    people = mentions_people(intent)

    brief = ImageBrief(
        intent=intent,
        profile=profile,
        seed=seed,
        mentions_people=people,
        selection=selection,
    )

    for value, table in (
        (controls.shot_type, SHOT_TYPES),
        (controls.camera_angle, CAMERA_ANGLES),
        (controls.composition, COMPOSITION),
    ):
        if phrase := _phrase(value, table):
            brief.framing.append(phrase)

    for value, table in ((controls.lens, LENSES), (controls.aperture, APERTURES)):
        if phrase := _phrase(value, table):
            brief.optics.append(phrase)

    for value, table in (
        (controls.lighting, LIGHTING),
        (controls.time_of_day, TIME_OF_DAY),
        (controls.weather, WEATHER),
    ):
        if phrase := _phrase(value, table):
            brief.light.append(phrase)

    if is_set(controls.visual_style) and controls.visual_style in VISUAL_STYLES:
        brief.style_name = controls.visual_style
        brief.style_group = STYLE_GROUP_OF.get(controls.visual_style, "")

    if stock := _phrase(controls.film_stock, FILM_STOCKS):
        brief.look.append(stock)

    if is_set(controls.colour_palette) and controls.colour_palette in COLOUR_PALETTES:
        prose, hexes = COLOUR_PALETTES[controls.colour_palette]
        brief.palette_prose = prose
        if controls.use_hex_swatches and profile.supports_hex_colour:
            brief.palette_hexes = hexes

    brief.material = _phrase(controls.material_focus, MATERIALS)
    brief.mood = _phrase(controls.mood, MOODS)
    brief.render_detail = _phrase(controls.render_detail, RENDER_DETAIL)

    if people:
        rng = seeded_rng_for(seed, "casting")
        if wardrobe := _phrase(controls.wardrobe, WARDROBE_REGISTERS):
            brief.people.append(wardrobe)
        if controls.concrete_identity:
            brief.people.append(
                f"{AGE_BANDS[rng.randrange(len(AGE_BANDS))]}, "
                f"{BUILDS[rng.randrange(len(BUILDS))]}, with "
                f"{FEATURE_HOOKS[rng.randrange(len(FEATURE_HOOKS))]}"
            )
        if controls.skin_texture:
            brief.people.append(SKIN_TEXTURE[rng.randrange(len(SKIN_TEXTURE))])

    if controls.render_text.strip():
        brief.render_text = controls.render_text.strip()
        brief.text_surface = controls.text_surface.strip()

    aspect = controls.aspect_ratio if is_set(controls.aspect_ratio) else "3:2 landscape"
    brief.aspect_label = aspect
    brief.width, brief.height = resolve_dimensions(aspect, controls.long_edge, profile)

    shape = controls.prompt_shape
    if shape is PromptShape.JSON and not profile.supports_json_prompt:
        shape = PromptShape.PROSE
    brief.prompt_shape = shape
    brief.word_target = profile.clamp_words(controls.word_target or profile.target_words)

    brief.has_reference_image = (
        controls.has_reference_image and profile.max_reference_images > 0
    )
    brief.reference_role = controls.reference_role

    if profile.negative_supported:
        brief.extra_negative = controls.extra_negative.strip()
        brief.smart_negative = controls.smart_negative

    return brief
