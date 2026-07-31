"""
image_engine.engine — orchestration for both tasks.

One class drives both jobs. :meth:`ImagePromptEngine.stream` writes a
generation prompt; :meth:`ImagePromptEngine.stream_edit` writes an editing
instruction. They share the client, the seeding contract and the sanitising
philosophy, and nothing else — the passes, the prompts and the post-processing
all differ.

The streaming client is injected rather than imported, so this module loads
with no Qt, no llama-server and no network. That keeps the engine testable
against a scripted double and, more importantly, means nothing in the
application's inference stack can be dragged into the LTX parity surface by
accident.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from .brief import ImageBrief, ImageControls, build_brief, resolve_sentinels
from .edit import EditBrief, EditControls, build_edit_brief, resolve_edit_sentinels
from .negative import build_negative
from .profiles import ModelProfile, PromptShape, profile_for
from .sanitize import sanitize_edit_instruction, sanitize_positive, word_count
from .selection import SelectionOutcome, SelectionPass
from .system import (
    build_edit_system,
    build_edit_user_turn,
    build_system,
    build_user_turn,
)

__all__ = [
    "ENGINE_VERSION",
    "ChatClient",
    "Generation",
    "EditGeneration",
    "ImagePromptEngine",
]

ENGINE_VERSION = "2.0.0"


class ChatClient(Protocol):
    """The slice of the application's LlamaClient this engine needs.

    Declared structurally so this module imports nothing from
    ``prompt_master.inference``. The real client should already satisfy this
    shape; if its signature differs, adapt it rather than editing the engine.
    """

    def stream_chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        seed: int,
        image_jpeg_b64: str | None = None,
    ) -> Iterator[str]:
        """Yield response fragments as the model writes them."""
        ...


def _draw_seed(seed: int | None) -> int:
    if seed is None or seed < 0:
        return random.randrange(0, 2**31 - 1)
    return seed


@dataclass
class Generation:
    """Everything the UI needs to display and reproduce one generation."""

    positive: str
    negative: str
    seed: int
    profile_key: str
    width: int
    height: int
    steps: int
    cfg: float
    word_count: int
    shape: PromptShape
    chosen: dict[str, str] = field(default_factory=dict)
    chosen_source: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def status_line(self) -> str:
        p = profile_for(self.profile_key)
        bits = [
            p.display_name,
            f"seed {self.seed}",
            f"{self.width}x{self.height}",
            f"{self.steps} steps",
            f"CFG {self.cfg:g}",
            f"{self.word_count} words",
        ]
        if not p.negative_supported:
            bits.append("no negative")
        return " · ".join(bits)

    def as_dict(self) -> dict[str, object]:
        return {
            "task": "generate",
            "positive": self.positive,
            "negative": self.negative,
            "seed": self.seed,
            "model": self.profile_key,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg": self.cfg,
            "shape": self.shape.value,
            "chosen": self.chosen,
            "engine_version": ENGINE_VERSION,
        }


@dataclass
class EditGeneration:
    """Everything the UI needs to display and reproduce one edit instruction."""

    instruction: str
    negative: str
    seed: int
    profile_key: str
    operation: str
    steps: int
    cfg: float
    denoise: float
    word_count: int
    references: tuple[str, ...] = ()
    chosen: dict[str, str] = field(default_factory=dict)
    chosen_source: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def status_line(self) -> str:
        p = profile_for(self.profile_key)
        bits = [
            p.display_name,
            "edit",
            f"seed {self.seed}",
            f"{self.steps} steps",
            f"CFG {self.cfg:g}",
            f"denoise {self.denoise:g}",
            f"{self.word_count} words",
        ]
        if self.references:
            bits.append(f"{len(self.references)} refs")
        return " · ".join(bits)

    def as_dict(self) -> dict[str, object]:
        return {
            "task": "edit",
            "instruction": self.instruction,
            "negative": self.negative,
            "seed": self.seed,
            "model": self.profile_key,
            "operation": self.operation,
            "steps": self.steps,
            "cfg": self.cfg,
            "denoise": self.denoise,
            "references": list(self.references),
            "chosen": self.chosen,
            "engine_version": ENGINE_VERSION,
        }


_REFINE_SYSTEM = """\
You read a finished image prompt and list the specific failure modes that this
particular description invites. Return a comma-separated list of short noun
phrases and nothing else. No more than twelve. Do not list generic faults that
apply to every image; list only what this description makes likely. If nothing
specific stands out, return an empty line.
"""


class ImagePromptEngine:
    """Orchestrates one generation or one edit, start to finish.

    Generation::

        engine = ImagePromptEngine(client)
        for chunk in engine.stream(intent, controls):
            ...
        result = engine.finish()

    Editing::

        for chunk in engine.stream_edit(request, edit_controls):
            ...
        result = engine.finish_edit()

    Or non-streaming: :meth:`generate` and :meth:`edit`.
    """

    def __init__(self, client: ChatClient | None = None) -> None:
        self._client = client
        self._selector = SelectionPass(client)
        self._brief: ImageBrief | None = None
        self._edit_brief: EditBrief | None = None
        self._selection: SelectionOutcome | None = None
        self._buffer: list[str] = []
        self._notes: list[str] = []

    # -- planning ----------------------------------------------------------

    def plan(
        self,
        intent: str,
        controls: ImageControls,
        seed: int | None = None,
        *,
        use_model_for_choice: bool = True,
    ) -> ImageBrief:
        """Resolve sentinels and build the brief without writing a prompt.

        Exposed so the UI can show the resolved seed, dimensions and every
        "Choose for me" result before the writer pass runs. Surfacing the
        chosen values is the point of CHOOSE — a hidden choice is no better
        than AUTO.
        """
        seed = _draw_seed(seed if seed is not None else controls.seed)
        settled, selection = resolve_sentinels(
            intent,
            controls,
            seed,
            selection_pass=self._selector,
            use_model=use_model_for_choice and self._client is not None,
        )
        brief = build_brief(intent, settled, seed, selection=selection)

        self._brief = brief
        self._edit_brief = None
        self._selection = selection
        self._buffer = []
        self._notes = list(selection.notes)
        return brief

    def plan_edit(
        self,
        request: str,
        controls: EditControls,
        seed: int | None = None,
        *,
        use_model_for_choice: bool = True,
    ) -> EditBrief:
        """Resolve sentinels and build the edit brief without writing.

        Raises :class:`ValueError` when the selected model cannot edit, so the
        UI can refuse before the user waits on a generation.
        """
        seed = _draw_seed(seed if seed is not None else controls.seed)
        settled, selection = resolve_edit_sentinels(
            request,
            controls,
            seed,
            selection_pass=self._selector,
            use_model=use_model_for_choice and self._client is not None,
        )
        brief = build_edit_brief(request, settled, seed, selection=selection)

        self._edit_brief = brief
        self._brief = None
        self._selection = selection
        self._buffer = []
        self._notes = list(selection.notes)
        return brief

    # -- generation --------------------------------------------------------

    def stream(
        self,
        intent: str,
        controls: ImageControls,
        *,
        seed: int | None = None,
        image_jpeg_b64: str | None = None,
        use_model_for_choice: bool = True,
    ) -> Iterator[str]:
        """Yield the generation prompt as it is written."""
        brief = self.plan(
            intent, controls, seed, use_model_for_choice=use_model_for_choice
        )
        profile = brief.profile

        if image_jpeg_b64 and not brief.has_reference_image:
            self._notes.append(
                "A reference image was attached but this profile accepts none; "
                "it was not sent."
            )
            image_jpeg_b64 = None

        yield from self._run(
            system=build_system(brief),
            user=build_user_turn(brief),
            temperature=profile.writer_temperature,
            top_p=profile.writer_top_p,
            seed=brief.seed,
            image_jpeg_b64=image_jpeg_b64,
        )

    def finish(self) -> Generation:
        """Sanitise, assemble the negative, and return the result."""
        if self._brief is None:
            raise RuntimeError("finish() called before stream() or plan()")

        brief = self._brief
        profile = brief.profile
        selection = self._selection or SelectionOutcome()

        positive = sanitize_positive(
            "".join(self._buffer), profile, shape=brief.prompt_shape
        )

        refine_terms: list[str] = []
        if profile.negative_supported and brief.smart_negative and positive:
            refine_terms = self._refine_negative(positive, brief)

        negative = build_negative(brief, positive, refine_terms)
        words = word_count(positive)

        if words < profile.min_words:
            self._notes.append(
                f"The writer returned {words} words; {profile.display_name} "
                f"reads best from about {profile.min_words}. Consider "
                "regenerating or adding detail to the intent."
            )
        if not profile.negative_supported:
            self._notes.append(
                f"{profile.display_name} has no negative conditioning, so no "
                "negative prompt was written."
            )
        if profile.licence_note:
            self._notes.append(profile.licence_note)

        return Generation(
            positive=positive,
            negative=negative,
            seed=brief.seed,
            profile_key=profile.key,
            width=brief.width,
            height=brief.height,
            steps=profile.default_steps,
            cfg=profile.default_cfg,
            word_count=words,
            shape=brief.prompt_shape,
            chosen=selection.resolved(),
            chosen_source=dict(selection.source),
            notes=list(self._notes),
        )

    def generate(
        self,
        intent: str,
        controls: ImageControls,
        *,
        seed: int | None = None,
        image_jpeg_b64: str | None = None,
        use_model_for_choice: bool = True,
    ) -> Generation:
        """Non-streaming convenience wrapper."""
        for _ in self.stream(
            intent,
            controls,
            seed=seed,
            image_jpeg_b64=image_jpeg_b64,
            use_model_for_choice=use_model_for_choice,
        ):
            pass
        return self.finish()

    # -- editing -----------------------------------------------------------

    def stream_edit(
        self,
        request: str,
        controls: EditControls,
        *,
        seed: int | None = None,
        image_jpeg_b64: str | None = None,
        use_model_for_choice: bool = True,
    ) -> Iterator[str]:
        """Yield the edit instruction as it is written."""
        brief = self.plan_edit(
            request, controls, seed, use_model_for_choice=use_model_for_choice
        )
        profile = brief.profile

        if brief.experimental and brief.requirements:
            self._notes.append(brief.requirements)
        if brief.compound_warning:
            self._notes.append(
                "This request looks like more than one change. These models "
                "hold identity better across a chain of small edits than "
                "across one compound instruction — consider splitting it."
            )

        yield from self._run(
            system=build_edit_system(brief),
            user=build_edit_user_turn(brief),
            temperature=profile.writer_temperature,
            top_p=profile.writer_top_p,
            seed=brief.seed,
            image_jpeg_b64=image_jpeg_b64,
        )

    def finish_edit(self) -> EditGeneration:
        """Sanitise the instruction and return the result."""
        if self._edit_brief is None:
            raise RuntimeError("finish_edit() called before stream_edit()")

        brief = self._edit_brief
        profile = brief.profile
        selection = self._selection or SelectionOutcome()

        instruction, warnings = sanitize_edit_instruction(
            "".join(self._buffer), profile
        )

        if profile.licence_note:
            self._notes.append(profile.licence_note)

        return EditGeneration(
            instruction=instruction,
            negative="",  # no supported editing profile takes one
            seed=brief.seed,
            profile_key=profile.key,
            operation=brief.operation,
            steps=brief.steps,
            cfg=brief.cfg,
            denoise=brief.denoise,
            word_count=word_count(instruction),
            references=tuple(brief.reference_phrases()),
            chosen=selection.resolved(),
            chosen_source=dict(selection.source),
            warnings=warnings,
            notes=list(self._notes),
        )

    def edit(
        self,
        request: str,
        controls: EditControls,
        *,
        seed: int | None = None,
        image_jpeg_b64: str | None = None,
        use_model_for_choice: bool = True,
    ) -> EditGeneration:
        """Non-streaming convenience wrapper."""
        for _ in self.stream_edit(
            request,
            controls,
            seed=seed,
            image_jpeg_b64=image_jpeg_b64,
            use_model_for_choice=use_model_for_choice,
        ):
            pass
        return self.finish_edit()

    # -- internals ---------------------------------------------------------

    def _run(self, **kwargs) -> Iterator[str]:
        if self._client is None:
            raise RuntimeError("No client configured; use plan() for a dry run.")
        for chunk in self._client.stream_chat(**kwargs):
            self._buffer.append(chunk)
            yield chunk

    def _refine_negative(self, positive: str, brief: ImageBrief) -> list[str]:
        """Ask the model which faults *this* description invites.

        Only ever runs for a profile with real classifier-free guidance.
        Failure is non-fatal: an exception leaves the gated banks alone rather
        than losing the generation.
        """
        try:
            out = "".join(
                self._client.stream_chat(
                    system=_REFINE_SYSTEM,
                    user=positive,
                    temperature=brief.profile.refine_temperature,
                    top_p=0.9,
                    seed=brief.seed,
                )
            )
        except Exception:  # noqa: BLE001 - never lose a generation to this
            self._notes.append("The smart-negative pass failed; banks only.")
            return []

        terms = [t.strip(" .;") for t in re.split(r"[,\n]", out) if t.strip(" .;")]
        terms = [t for t in terms if 2 <= len(t) <= 48]

        # A refine pass that echoes the positive would point the negative
        # straight at the image being asked for. Drop anything already said.
        haystack = positive.lower()
        terms = [t for t in terms if t.lower() not in haystack][:12]

        if not terms:
            self._notes.append("The smart-negative pass added nothing.")
        return terms
