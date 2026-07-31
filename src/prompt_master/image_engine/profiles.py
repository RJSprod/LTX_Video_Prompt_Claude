"""
image_engine.profiles — what differs between one image model and another.

Every model-specific behaviour lives in a frozen :class:`ModelProfile`. The
builders read the profile and nothing else, so adding a fourth model means
adding a profile, not editing a builder. If you find yourself writing
``if profile.family == "flux"`` outside this module and the two helper
functions in ``system.py`` that already do it, the property you need is
missing from the profile — add it here instead.

Sources for the numbers are noted inline. Several postdate the author's
knowledge and were taken from research; ``docs/IMAGE_PROMPT_MODE_SPEC.md`` §12
lists what to re-verify before release. All of them are one-line edits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

__all__ = [
    "NegativeMode",
    "PromptShape",
    "EditCapability",
    "ModelProfile",
    "FLUX_KLEIN_9B",
    "KREA_2_TURBO",
    "KREA_2_RAW",
    "PROFILES",
    "DEFAULT_PROFILE_KEY",
    "profile_for",
    "editing_profiles",
]


class NegativeMode(str, Enum):
    """How a model treats a negative prompt.

    NONE  no negative conditioning path at all, or guidance-distilled to the
          point where one is discarded. The engine returns an empty negative
          and the UI hides the field.
    CFG   real classifier-free guidance. The negative is assembled from the
          gated banks and is worth writing precisely.
    """

    NONE = "none"
    CFG = "cfg"


class PromptShape(str, Enum):
    """The surface form the writer pass should produce."""

    PROSE = "prose"
    JSON = "json"


class EditCapability(str, Enum):
    """Whether, and how, a model accepts instruction-based editing.

    NONE          text-to-image only. Edit mode is unavailable.
    NATIVE        the shipped weights do instruction editing.
    EXPERIMENTAL  editing exists only through a third-party LoRA or custom
                  nodes. Available, but the UI must say so plainly and it must
                  not be the default.
    """

    NONE = "none"
    NATIVE = "native"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class ModelProfile:
    """Everything that differs between one image model and another."""

    key: str
    display_name: str
    family: str

    # --- text understanding -------------------------------------------------
    text_encoder: str
    understands_prose: bool = True
    understands_tags: bool = False
    honours_attention_weights: bool = False
    """``(word:1.4)``. False on every supported model — the encoder is an
    instruction-tuned LLM and the parentheses arrive as literal text."""

    front_load_subject: bool = True

    # --- length, generation -------------------------------------------------
    min_words: int = 30
    target_words: int = 80
    max_words: int = 140
    hard_word_ceiling: int = 300

    # --- length, editing ----------------------------------------------------
    edit_min_words: int = 20
    edit_target_words: int = 55
    edit_max_words: int = 110
    """Edit instructions run shorter than generation prompts. The instruction
    plus the intended result plus preservation clauses is the whole job."""

    # --- negative handling --------------------------------------------------
    supports_negative: bool = False
    negative_mode: NegativeMode = NegativeMode.NONE
    negative_max_terms: int = 0

    # --- editing ------------------------------------------------------------
    edit_capability: EditCapability = EditCapability.NONE
    max_reference_images: int = 0
    reference_addressing: str = "role"
    """"index_and_role" — the model resolves "image 1" and wants the role too.
    "role" — describe the role only. "none" — no references."""

    supports_masked_inpainting: bool = False
    """Whole-image instruction editing is not the same as masked inpainting.
    Neither supported model does masks; local edits are instruction-scoped."""

    edit_steps: int = 0
    edit_cfg: float = 0.0
    edit_denoise_default: float = 0.5
    edit_denoise_preserve: float = 0.35
    edit_denoise_transform: float = 0.7
    edit_requirements: str = ""
    """Non-empty when the capability is EXPERIMENTAL: what the user must
    install before this will work at all."""

    # --- sampling defaults reported to the user -----------------------------
    default_steps: int = 8
    default_cfg: float = 0.0
    cfg_range: tuple[float, float] = (0.0, 0.0)

    # --- capabilities -------------------------------------------------------
    supports_hex_colour: bool = False
    text_render_quality: str = "fair"  # "strong" | "fair" | "weak"
    supports_json_prompt: bool = False

    # --- geometry -----------------------------------------------------------
    min_edge: int = 512
    max_edge: int = 2048
    edge_multiple: int = 16
    max_pixels: int = 4_194_304
    native_edge: int = 1024
    max_edit_input_edge: int = 4000
    """Inputs larger than this degrade on edit even with more steps."""

    # --- writer pass --------------------------------------------------------
    writer_temperature: float = 0.85
    writer_top_p: float = 0.92
    refine_temperature: float = 0.45
    select_temperature: float = 0.25
    """Choose-for-me runs cold. It is a classification task, not a creative
    one, and warmth buys nothing but invalid options."""

    # --- prose guidance injected into the system prompt ---------------------
    house_style: str = ""
    edit_house_style: str = ""
    avoid_vocabulary: tuple[str, ...] = ()
    known_biases: tuple[str, ...] = ()
    edit_biases: tuple[str, ...] = ()
    licence_note: str = ""

    # -- derived -------------------------------------------------------------

    @property
    def negative_supported(self) -> bool:
        return self.supports_negative and self.negative_mode is not NegativeMode.NONE

    @property
    def can_edit(self) -> bool:
        return self.edit_capability is not EditCapability.NONE

    @property
    def edit_is_experimental(self) -> bool:
        return self.edit_capability is EditCapability.EXPERIMENTAL

    def clamp_words(self, n: int) -> int:
        return max(self.min_words, min(n, self.max_words))

    def clamp_edit_words(self, n: int) -> int:
        return max(self.edit_min_words, min(n, self.edit_max_words))


# ---------------------------------------------------------------------------
# FLUX.2 [klein] 9B
#
# Black Forest Labs. Generation and editing unified in one 9B rectified-flow
# transformer with an 8B Qwen3 text embedder. Step-distilled variants run at 4
# steps; the 9B base is tunable and wants ~20-30 steps for complex edits.
# BFL's guidance is explicit that FLUX takes no negative prompt.
# ---------------------------------------------------------------------------

FLUX_KLEIN_9B = ModelProfile(
    key="flux_klein_9b",
    display_name="FLUX.2 [klein] 9B",
    family="flux",
    text_encoder="Qwen3 8B embedder",
    min_words=40,
    target_words=85,
    max_words=120,
    hard_word_ceiling=300,
    edit_min_words=20,
    edit_target_words=55,
    edit_max_words=110,
    supports_negative=False,
    negative_mode=NegativeMode.NONE,
    negative_max_terms=0,
    edit_capability=EditCapability.NATIVE,
    max_reference_images=4,
    reference_addressing="index_and_role",
    supports_masked_inpainting=False,
    edit_steps=24,
    edit_cfg=1.0,
    edit_denoise_default=0.5,
    edit_denoise_preserve=0.35,
    edit_denoise_transform=0.7,
    default_steps=4,
    default_cfg=1.0,
    cfg_range=(1.0, 1.5),
    supports_hex_colour=True,
    text_render_quality="strong",
    supports_json_prompt=True,
    min_edge=128,
    max_edge=2048,
    edge_multiple=16,
    max_pixels=4_194_304,
    native_edge=1024,
    max_edit_input_edge=4000,
    writer_temperature=0.80,
    writer_top_p=0.92,
    refine_temperature=0.40,
    select_temperature=0.25,
    house_style=(
        "FLUX.2 reads the prompt with a large instruction-tuned text encoder. "
        "It parses grammar, so write in ordinary sentences rather than a list "
        "of comma-separated tags. Word order is emphasis: whatever appears in "
        "the first clause dominates the frame. Structure the description as "
        "subject, then what the subject is doing, then the style, then the "
        "surrounding context, then any secondary detail. The model does not "
        "expand or rewrite a short prompt internally, so every element that "
        "should appear must be stated."
    ),
    edit_house_style=(
        "FLUX.2 edits by instruction rather than by mask. The instruction goes "
        "first as a plain imperative, then the intended result is described in "
        "full, then everything that must survive the edit is named explicitly. "
        "The model changes what it is told to change and will drift on "
        "anything left unmentioned, so preservation is stated, not assumed. "
        "Reference images are addressed by number and by role together."
    ),
    avoid_vocabulary=(
        "ultra realistic",
        "hyperrealistic",
        "highly detailed",
    ),
    known_biases=(
        "centres the subject unless the composition is stated explicitly",
        "defaults to a very shallow depth of field unless an aperture is given",
        "converges on one generic face unless age, build and features are named",
        "renders skin waxy and poreless unless surface texture is described",
        "loses hands when two or more people interact, so give hands a simple "
        "resting position unless the hands are the subject",
    ),
    edit_biases=(
        "applies a local instruction globally when the target is not pinned "
        "down, so the element being changed must be identified unambiguously",
        "drifts on identity across successive edits, so facial features and "
        "expression are restated every turn",
        "degrades over long chains of edits, so several small instructions "
        "beat one compound one",
        "mismatches light direction when compositing, so the light already in "
        "the scene is described rather than left implicit",
    ),
    licence_note=(
        "FLUX.2 [klein] 9B is distributed under the FLUX non-commercial "
        "licence; the 4B variant is Apache-2.0. Confirm licensing before "
        "commercial use."
    ),
)


# ---------------------------------------------------------------------------
# Krea 2
#
# Krea AI. 12.9B single-stream DiT, Qwen3-VL-4B-Instruct text encoder, trained
# for aesthetic range and specifically against the oversaturated plastic look.
# The shipped model is text-to-image only: Krea's own technical report lists
# editing as future work. Instruction editing on these weights exists only via
# a third-party LoRA plus custom nodes, so the capability is EXPERIMENTAL.
# ---------------------------------------------------------------------------

_KREA_EDIT_REQUIREMENTS = (
    "Stock Krea 2 does not do instruction editing — Krea's technical report "
    "lists it as future work. This path needs the third-party Krea 2 Identity "
    "Edit LoRA and the ComfyUI Krea2Edit node pack installed. Neither is "
    "official. Keep inputs at or below 2 megapixels, and where two references "
    "are used, put the scene first and the person second."
)

KREA_2_TURBO = ModelProfile(
    key="krea_2_turbo",
    display_name="Krea 2 (Turbo)",
    family="krea",
    text_encoder="Qwen3-VL-4B-Instruct",
    min_words=30,
    target_words=75,
    max_words=140,
    hard_word_ceiling=300,
    edit_min_words=15,
    edit_target_words=45,
    edit_max_words=90,
    supports_negative=False,
    negative_mode=NegativeMode.NONE,
    negative_max_terms=0,
    edit_capability=EditCapability.EXPERIMENTAL,
    max_reference_images=2,
    reference_addressing="role",
    supports_masked_inpainting=False,
    edit_steps=10,
    edit_cfg=1.0,
    edit_denoise_default=0.5,
    edit_denoise_preserve=0.35,
    edit_denoise_transform=0.65,
    edit_requirements=_KREA_EDIT_REQUIREMENTS,
    default_steps=8,
    default_cfg=0.0,
    cfg_range=(0.0, 0.0),
    supports_hex_colour=False,
    text_render_quality="fair",
    supports_json_prompt=False,
    min_edge=512,
    max_edge=2048,
    edge_multiple=16,
    max_pixels=4_194_304,
    native_edge=1536,
    max_edit_input_edge=2048,
    writer_temperature=0.88,
    writer_top_p=0.94,
    refine_temperature=0.45,
    select_temperature=0.25,
    house_style=(
        "Krea 2 was trained for aesthetic range and specifically against the "
        "flat, oversaturated, plastic look that image models drift toward. It "
        "rewards concrete, physical description and punishes decorative "
        "quality adjectives. Write plain sentences about what is actually "
        "there: the material, the light source, the surface, the weather. "
        "Emphasis comes from position and from restating an idea in different "
        "words, never from weighting syntax. Prefer a specific colour name "
        "over a vague one. The model runs distilled with no classifier-free "
        "guidance, so a negative prompt does nothing and should be left empty."
    ),
    edit_house_style=(
        "This is a community editing path built on a model that does not ship "
        "with editing. Keep the instruction short and concrete: name the "
        "element, name the change, name what must not move. Avoid compound "
        "instructions entirely — one change per generation."
    ),
    avoid_vocabulary=(
        "beautiful",
        "perfect",
        "vibrant colors",
        "hyperrealistic",
    ),
    known_biases=(
        "drifts to glossy magazine finish when quality adjectives are stacked",
        "over-smooths skin unless pores, freckles or fine lines are named",
        "renders in-image typography only adequately, so keep rendered text "
        "short and give it a clear surface to sit on",
        "occasionally produces structural errors in hands and limbs in dense "
        "multi-figure scenes",
    ),
    edit_biases=(
        "was not trained for editing, so compound instructions fail more often "
        "than they do on a native editor",
        "removals are the weakest operation and want the undistilled "
        "checkpoint with more steps",
    ),
    licence_note=(
        "Krea 2 is released under the Krea 2 Community Licence: free "
        "commercial use below roughly $1M revenue and 50 seats, enterprise "
        "licence required above that. Verify current terms."
    ),
)


KREA_2_RAW = replace(
    KREA_2_TURBO,
    key="krea_2_raw",
    display_name="Krea 2 (RAW)",
    supports_negative=True,
    negative_mode=NegativeMode.CFG,
    negative_max_terms=48,
    default_steps=52,
    default_cfg=3.5,
    cfg_range=(2.5, 5.0),
    edit_steps=20,
    edit_cfg=3.0,
    writer_temperature=0.90,
    house_style=KREA_2_TURBO.house_style.replace(
        "The model runs distilled with no classifier-free guidance, so a "
        "negative prompt does nothing and should be left empty.",
        "This is the undistilled checkpoint running real classifier-free "
        "guidance, so a negative prompt genuinely conditions the result and is "
        "worth writing precisely.",
    ),
)


PROFILES: dict[str, ModelProfile] = {
    p.key: p for p in (FLUX_KLEIN_9B, KREA_2_TURBO, KREA_2_RAW)
}

DEFAULT_PROFILE_KEY = KREA_2_TURBO.key


def profile_for(key: str) -> ModelProfile:
    """Look up a profile, falling back to the default rather than raising.

    The UI persists a profile key. A key written by a newer build must not stop
    an older one from opening.
    """
    return PROFILES.get(key, PROFILES[DEFAULT_PROFILE_KEY])


def editing_profiles(include_experimental: bool = False) -> list[ModelProfile]:
    """Profiles that can be offered in edit mode.

    By default this is FLUX.2 [klein] alone. Krea 2 appears only when the user
    has explicitly opted into the experimental path.
    """
    out = [p for p in PROFILES.values() if p.edit_capability is EditCapability.NATIVE]
    if include_experimental:
        out += [p for p in PROFILES.values() if p.edit_is_experimental]
    return out
