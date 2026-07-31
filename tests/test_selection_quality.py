"""
tests/test_selection_quality.py — a standing benchmark for the CHOOSE fallback.

The deterministic heuristic is what runs when the model is unavailable, returns
something invalid, or was never asked. It is a fallback, not the primary path,
so it does not need to be perfect — but it does need to not regress silently
when someone edits the cue tables.

These cases were built by writing down what a person would actually type. Every
one that fails should be fixed by adding or demoting a cue in
``options.SELECTION_CUES``, not by loosening the threshold.

If a case here becomes genuinely ambiguous, delete it rather than weakening the
suite around it.

Only the import lines differ from the delivered bundle; see the note in
``test_image_engine.py``.
"""

from __future__ import annotations

import pytest

from prompt_master import image_engine as ie
from prompt_master.image_engine import heuristic_pick

OPS = ie.options.EDIT_OPERATIONS

EDIT_CASES = [
    ("please delete the bins by the wall", "Remove an object"),
    ("get rid of the guy on the right", "Remove an object"),
    ("put a bench under the tree", "Add an object"),
    ("add a flock of birds to the sky", "Add an object"),
    ("swap the sky for a stormy one", "Replace an object"),
    ("make it sunset", "Change time of day"),
    ("change it to night", "Change time of day"),
    ("change the background to a beach", "Replace the background"),
    ("turn it into an oil painting", "Change the style"),
    ("make him look 30 years older", "Change age"),
    ("she should be smiling", "Change expression"),
    ("put her in a red coat", "Change clothing"),
    ("dress him in a uniform", "Change clothing"),
    ("colourise this old photo", "Colourise or restore"),
    ("make the sign say CLOSED", "Edit text in the image"),
    ("take the person from image 1 into image 2", "Transfer a subject"),
    # "make it snow" was here and was removed: falling snow is weather, snow
    # on the ground is season, and the phrase genuinely means either. Cases
    # that need a coin-flip do not belong in a regression suite.
    ("make it pour with rain", "Change the weather"),
    ("give the trees autumn leaves", "Change the season"),
    ("zoom out and show more of the street", "Extend the frame"),
]

LOOK_CASES = [
    ("a neon-lit street at night in the rain", ie.options.LIGHTING, "Neon"),
    ("a campfire in the woods after dark", ie.options.LIGHTING, "Firelight"),
    ("a macro shot of a dew droplet", ie.options.LENSES, "100mm macro"),
    ("a headshot for a company website", ie.options.LENSES, "85mm portrait"),
    ("an anime panel of a schoolgirl", ie.options.VISUAL_STYLES, "Manga"),
    ("a blueprint of a steam engine", ie.options.VISUAL_STYLES, "Technical drawing"),
    ("a foggy harbour at dawn", ie.options.TIME_OF_DAY, "Dawn"),
    ("a brutalist car park", ie.options.MATERIALS, "Raw concrete"),
    ("a lonely figure in the rain, everything feels sad", ie.options.MOODS, "Melancholy"),
    ("an aerial view of a rooftop", ie.options.CAMERA_ANGLES, "Bird's eye"),
]


@pytest.mark.parametrize("request_text,expected", EDIT_CASES)
def test_edit_operation_inference(request_text: str, expected: str) -> None:
    assert heuristic_pick(request_text, OPS) == expected


@pytest.mark.parametrize("intent,bank,expected", LOOK_CASES)
def test_look_inference(intent: str, bank, expected: str) -> None:
    assert heuristic_pick(intent, bank) == expected


def test_nonsense_yields_nothing() -> None:
    """A bad guess is worse than saying nothing — this must stay empty."""
    assert heuristic_pick("qqq zzz wibble", ie.options.LIGHTING) == ""
    assert heuristic_pick("", ie.options.LIGHTING) == ""


def test_generic_verbs_do_not_dominate_specific_nouns() -> None:
    """Regression: "put her in a red coat" resolved to Add an object."""
    assert heuristic_pick("put her in a red coat", OPS) == "Change clothing"


def test_ambiguous_state_words_lose_to_specific_ones() -> None:
    """Regression: "after dark" beat "campfire" for a firelit scene."""
    assert heuristic_pick("a campfire in the woods after dark", ie.options.LIGHTING) == (
        "Firelight"
    )


def test_selection_is_deterministic() -> None:
    for intent, expected in EDIT_CASES[:5]:
        assert {heuristic_pick(intent, OPS) for _ in range(5)} == {expected}
