"""
image_engine.system — the system and user prompts for each pass.

Parallel in role to the video engine's ``brain.build_system``, sharing no code
with it and producing entirely different text.

Two families of prompt live here:

  build_system / build_user_turn            generation
  build_edit_system / build_edit_user_turn  editing

The generation prompts differ per model family because FLUX and Krea want to be
spoken to differently — FLUX wants subject-first Subject/Action/Style/Context
ordering, Krea wants plain physical description and specifically punishes the
glossy vocabulary other models reward. The edit prompts differ because only one
of the two models edits natively.
"""

from __future__ import annotations

from .brief import ImageBrief
from .edit import EditBrief
from .profiles import ModelProfile, PromptShape

__all__ = [
    "build_system",
    "build_user_turn",
    "build_edit_system",
    "build_edit_user_turn",
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_COMMON_RULES = """\
You are a prompt writer for a text-to-image model. You are given a director's
intent and a set of resolved production choices. You return the finished image
prompt and nothing else.

Absolute rules:
- Output only the prompt. No preamble, no explanation, no markdown, no quotes
  around the whole thing, no trailing commentary.
- Never write about what should be absent. An image model cannot subtract; a
  sentence about what is missing puts the thing in the frame.
- Never use attention weighting such as (word:1.4), [word], or {word}. This
  model reads that as literal punctuation.
- Never stack quality adjectives. They do not add quality; they add a generic
  commercial finish.
- Describe one moment. Not a sequence, not a before and after, not motion over
  time.
- Everything in the production choices must appear in the prompt, expressed
  naturally rather than pasted in as a list.
- Anything the intent asks for that the production choices do not mention is
  yours to decide, consistent with the rest.
"""


def _bias_block(biases: tuple[str, ...], lead: str) -> str:
    if not biases:
        return ""
    lines = "\n".join(f"- {b}" for b in biases)
    return f"\n{lead}\n{lines}\n"


def _structure_block(profile: ModelProfile) -> str:
    if profile.family == "flux":
        return (
            "\nStructure, in this order: the subject, then what the subject is "
            "doing, then the medium and style, then the setting and light, "
            "then any secondary detail. The first clause dominates the image, "
            "so the subject goes there and nothing else does.\n"
        )
    return (
        "\nStructure: open on the subject in a plain declarative clause, then "
        "build outward through action, setting, light and surface. Emphasis "
        "comes from position and from restating an idea in different words "
        "later in the paragraph.\n"
    )


def _text_block(profile: ModelProfile, wanted: bool) -> str:
    if not wanted:
        return ""
    advice = {
        "strong": (
            "This model renders text well. Put the exact string in double "
            "quotes, name the surface it is printed on, and describe the "
            "lettering style and weight."
        ),
        "fair": (
            "This model renders short text adequately. Put the exact string in "
            "double quotes, keep it short, give it an unambiguous surface, and "
            "do not ask for a second block of text elsewhere in the frame."
        ),
        "weak": (
            "This model renders text poorly. Keep the string very short and "
            "expect it to need a retry."
        ),
    }[profile.text_render_quality]
    return f"\nText in image: {advice}\n"


def _negative_policy_block(profile: ModelProfile) -> str:
    if profile.negative_supported:
        return (
            "\nA separate negative prompt is assembled by the application. Do "
            "not write one, and do not mention avoidance anywhere in your "
            "output.\n"
        )
    return (
        f"\n{profile.display_name} has no negative conditioning. There is "
        "nowhere for a list of unwanted things to go, so every quality you "
        "want must be stated positively. If you want clean hands, describe "
        "where the hands are.\n"
    )


def build_system(brief: ImageBrief) -> str:
    """Compose the system prompt for the generation writer pass."""
    p = brief.profile
    parts = [
        _COMMON_RULES,
        f"\nTarget model: {p.display_name}.\n{p.house_style}\n",
        _structure_block(p),
        _bias_block(p.known_biases, "This model has known tendencies. Write against them:"),
        (
            f"\nLength: aim for about {brief.word_target} words. Stay between "
            f"{p.min_words} and {p.max_words}. Past roughly {p.hard_word_ceiling} "
            "words this model stops reading, so nothing important may sit at "
            "the end.\n"
        ),
        _negative_policy_block(p),
        _text_block(p, bool(brief.render_text)),
    ]

    if brief.palette_hexes:
        parts.append(
            "\nColour: hex values are given. Bind each one to a named object — "
            '"the door is #5C1A1B" works, "use #5C1A1B somewhere" does not.\n'
        )

    if brief.has_reference_image:
        role = {
            "style": (
                "A reference image is attached for style. Describe the "
                "treatment you can see in it — palette, light quality, grain, "
                "finish — in words. Do not describe its subject matter unless "
                "the intent asks for it."
            ),
            "subject": (
                "A reference image is attached showing the subject. Describe "
                "the subject's visible attributes in words so the prompt "
                "stands alone. Never refer to it as 'the reference'."
            ),
            "composition": (
                "A reference image is attached for composition. Describe its "
                "layout, the placement of masses and the eye path in words."
            ),
        }.get(brief.reference_role, "")
        if role:
            parts.append(f"\n{role}\n")

    if brief.prompt_shape is PromptShape.JSON:
        parts.append(
            "\nOutput format: a single JSON object and nothing else, with the "
            "keys scene, subjects, composition, camera, lighting, "
            "color_palette, style, mood, background. Each value is a short "
            "descriptive string; subjects is an array of strings.\n"
        )

    if p.avoid_vocabulary:
        parts.append(
            f"\nNever use these words: {', '.join(p.avoid_vocabulary)}. They "
            "push this model toward its most generic output.\n"
        )

    if brief.mentions_people:
        parts.append(
            "\nThe scene contains people. Give each one enough concrete "
            "description to be a specific person rather than a type, keep "
            "clothing described plainly, and place hands somewhere simple "
            "unless the hands are the subject.\n"
        )

    return "".join(parts).strip()


def build_user_turn(brief: ImageBrief) -> str:
    """The director's brief, as the generation writer receives it."""
    lines = [f"INTENT\n{brief.intent}\n"]

    if facets := brief.facets():
        lines.append("PRODUCTION CHOICES")
        lines += [f"- {label}: {text}" for label, text in facets]
        lines.append("")

    shape = "as JSON" if brief.prompt_shape is PromptShape.JSON else "as a single paragraph"
    lines.append(
        f"OUTPUT\nOne image prompt for {brief.profile.display_name}, "
        f"about {brief.word_target} words, {shape}. "
        f"Frame is {brief.width}x{brief.height} ({brief.aspect_label})."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

_EDIT_RULES = """\
You are writing an editing instruction for an image model that edits by
language rather than by mask. You are given the user's edit request, the
element being changed, and a list of things that must survive the edit. You
return the finished instruction and nothing else.

Shape of the instruction, in this order:
1. The change, as a plain imperative. "Replace the wooden door with a steel
   one." Not "the image should have" and not a question.
2. The intended result, described in enough detail that the model knows what
   it is aiming at — material, colour, condition, how it sits in the scene.
3. What must not change, named explicitly.

Absolute rules:
- Output only the instruction. No preamble, no explanation, no markdown, no
  code fence, no commentary.
- Identify the element being changed unambiguously. Never rely on a pronoun.
  "The woman in the red coat on the left", not "her". If the scene could hold
  more than one of the thing being changed, say which one by position, colour,
  size or relationship to something else.
- State what survives. This model changes what it is told to change and drifts
  on whatever is left unsaid, so the preservation clauses are not padding —
  they are the instruction's second half.
- Do not use attention weighting such as (word:1.4) or [word].
- Do not describe the whole image from scratch. This is a change to an image
  that already exists, not a new one.
- One change. If the request contains several, write the most important one and
  say nothing about the rest.
"""


def build_edit_system(brief: EditBrief) -> str:
    """Compose the system prompt for the edit writer pass."""
    p = brief.profile
    parts = [_EDIT_RULES, f"\nTarget model: {p.display_name}.\n{p.edit_house_style}\n"]

    parts.append(
        _bias_block(
            p.edit_biases,
            "This model has known editing tendencies. Write against them:",
        )
    )

    parts.append(
        f"\nLength: aim for about {brief.word_target} words, between "
        f"{p.edit_min_words} and {p.edit_max_words}. An edit instruction is "
        "shorter than a generation prompt — the change, the result, what "
        "survives, and nothing else.\n"
    )

    if brief.references:
        if brief.reference_addressing == "index_and_role":
            parts.append(
                "\nReference images are attached. Refer to each one by its "
                "number and its role together — \"the subject from image 1\", "
                "\"the background from image 2\". A reference mentioned "
                "without its number may not be resolved, and a number without "
                "a role tells the model nothing about what to take from it. "
                "Available references:\n"
                + "\n".join(f"  - {r}" for r in brief.reference_phrases())
                + "\n"
            )
        else:
            parts.append(
                "\nReference images are attached. Refer to each by its role in "
                "words. Available references:\n"
                + "\n".join(f"  - {r}" for r in brief.reference_phrases())
                + "\n"
            )

    if not p.supports_masked_inpainting:
        parts.append(
            "\nThis model has no mask. A local change is scoped only by how "
            "precisely the target is described, so the target description is "
            "doing the work a mask would otherwise do.\n"
        )

    if p.negative_supported:
        parts.append(
            "\nA separate negative prompt is assembled by the application. Do "
            "not write one.\n"
        )
    else:
        parts.append(
            f"\n{p.display_name} takes no negative prompt, so an unwanted "
            "outcome cannot be excluded after the fact. Everything must be "
            "phrased as what the result should be.\n"
        )

    if brief.strength == "preserve":
        parts.append(
            "\nStrength: the smallest change that satisfies the request. Lean "
            "hard on the preservation clauses and change nothing that was not "
            "asked for.\n"
        )
    elif brief.strength == "transform":
        parts.append(
            "\nStrength: commit to the change. Describe the new state fully "
            "and confidently, while still holding the preservation clauses.\n"
        )

    if brief.experimental:
        parts.append(
            "\nThis model was not trained for editing. Keep the instruction "
            "shorter and more concrete than you otherwise would, and do not "
            "attempt anything compound.\n"
        )

    return "".join(parts).strip()


def build_edit_user_turn(brief: EditBrief) -> str:
    """The edit request, as the writer receives it."""
    lines = [f"EDIT REQUEST\n{brief.request}\n"]

    if facets := brief.facets():
        lines.append("RESOLVED")
        lines += [f"- {label}: {text}" for label, text in facets]
        lines.append("")

    if brief.compound_warning:
        lines.append(
            "NOTE\nThis request appears to contain more than one change. Write "
            "the single most important one only.\n"
        )

    lines.append(
        f"OUTPUT\nOne editing instruction for {brief.profile.display_name}, "
        f"about {brief.word_target} words: the change, the intended result, "
        "and what must survive."
    )
    return "\n".join(lines)
