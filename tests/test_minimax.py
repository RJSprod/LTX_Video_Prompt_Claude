"""The MiniMax H3 enhancer, checked against the tool it was ported from.

Three things are checked here, and they are the three that would make this a
different enhancer wearing WanGP's name: instructions that are not the vendored
ones, a request that is not shaped the way the H3 models were trained to be
asked, and a sampler set to something other than what WanGP sets it to.

The numbers and the strings the assertions use are written out rather than read
from the module under test wherever writing them out is possible: a constant
compared against itself passes whatever it has been changed to.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from prompt_master.minimax import enhancer
from prompt_master.minimax.prompt_enhancer import (FL2VA_IMAGE_SYSTEM_PROMPT,
                                                   FL2VA_PROMPT_INFOS,
                                                   FL2VA_TEXT_SYSTEM_PROMPT,
                                                   REF2VA_IMAGE_SYSTEM_PROMPT,
                                                   REF2VA_PROMPT_INFOS,
                                                   REF2VA_TEXT_SYSTEM_PROMPT)

VENDORED = Path(enhancer.__file__).with_name("prompt_enhancer.py")
PROVENANCE = Path(enhancer.__file__).with_name("UPSTREAM_SOURCE.txt")


# ── the instructions are the vendored ones, untouched ────────────────────────

def test_the_vendored_file_still_matches_the_digest_it_arrived_with():
    """The whole claim of this port is that the prompt text is WanGP's. A digest
    recorded beside the file is what keeps that a fact rather than an intention."""
    recorded = re.findall(r"^([0-9a-f]{64})\s+prompt_enhancer\.py\s*$",
                          PROVENANCE.read_text(encoding="utf-8"), re.MULTILINE)
    assert len(recorded) == 1, "the provenance note should name one digest"
    assert hashlib.sha256(VENDORED.read_bytes()).hexdigest() == recorded[0]


def test_enhancer_py_writes_no_prompt_text_of_its_own():
    """Every instruction is imported. The only English of any length this module
    adds is quoted from WanGP: the sentence it puts in front of a user's own
    instructions, and the sentence it gives its captioner."""
    from prompt_master.minimax import prompt_enhancer as vendored

    quoted = {"SUFFIX_PREAMBLE", "CAPTION_INSTRUCTION"}
    written = {name for name, value in vars(enhancer).items()
               if isinstance(value, str) and len(value) > 120 and not name.startswith("__")
               and getattr(vendored, name, None) is not value}
    assert written <= quoted
    assert enhancer.SUFFIX_PREAMBLE == (
        "Follow these additional user instructions with higher priority if they conflict "
        "with the guidance above:")


@pytest.mark.parametrize("variant,has_image,expected", [
    (enhancer.FL2VA, False, FL2VA_TEXT_SYSTEM_PROMPT),
    (enhancer.FL2VA, True, FL2VA_IMAGE_SYSTEM_PROMPT),
    (enhancer.REF2VA, False, REF2VA_TEXT_SYSTEM_PROMPT),
    (enhancer.REF2VA, True, REF2VA_IMAGE_SYSTEM_PROMPT),
])
def test_the_four_generations_use_the_four_instructions(variant, has_image, expected):
    """text_prompt_enhancer_instructions without a picture, video_prompt_enhancer_
    instructions with one, on whichever H3 model definition is chosen."""
    assert enhancer.instructions(variant, has_image) is expected


def test_the_variants_carry_their_own_token_budgets_and_guides():
    assert enhancer.max_tokens(enhancer.FL2VA) == 1024
    assert enhancer.max_tokens(enhancer.REF2VA) == 2048
    assert enhancer.infos(enhancer.FL2VA) is FL2VA_PROMPT_INFOS
    assert enhancer.infos(enhancer.REF2VA) is REF2VA_PROMPT_INFOS


def test_an_unknown_variant_falls_back_to_the_one_wangp_lists_first():
    assert enhancer.variant_of("") == enhancer.FL2VA
    assert enhancer.variant_of("minimax_h3_ref2va") == enhancer.FL2VA
    assert enhancer.variant_of(enhancer.REF2VA) == enhancer.REF2VA


def test_wangps_own_names_for_the_four_generations():
    assert enhancer.label(enhancer.FL2VA, False) == "Write an H3 Prompt from Text"
    assert enhancer.label(enhancer.FL2VA, True) == "Write an H3 Prompt from Text + Start Image"
    assert enhancer.label(enhancer.REF2VA, False) == "Write an H3 Reference Prompt from Text"
    assert enhancer.label(enhancer.REF2VA, True) == (
        "Write an H3 Reference Prompt from Text + First Reference Image")
    assert enhancer.BUTTON_LABEL == "Write H3 Prompt"


# ── the request is shaped the way the model was asked in training ────────────

def test_a_text_request_is_the_instructions_and_one_labelled_line():
    """The instructions arrive trailing-whitespace-stripped, which is what
    WanGP's own merge does to them before every generation."""
    messages = enhancer.messages("a red ball rolls", variant=enhancer.FL2VA)
    assert messages == [{"role": "system", "content": FL2VA_TEXT_SYSTEM_PROMPT.rstrip()},
                        {"role": "user", "content": "user_prompt: a red ball rolls"}]


def test_a_caption_becomes_the_second_line_of_the_same_turn():
    messages = enhancer.messages("it starts rolling", variant=enhancer.REF2VA,
                                 image_caption="A red ball on a table.")
    assert messages[0]["content"] == REF2VA_IMAGE_SYSTEM_PROMPT.rstrip()
    assert messages[1]["content"] == ("user_prompt: it starts rolling\n"
                                      "image_caption: A red ball on a table.")


def test_the_captioner_is_asked_exactly_what_wangp_asks_it():
    messages = enhancer.caption_messages("data:image/jpeg;base64,AAAA")
    assert messages == [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
        {"type": "text", "text": "Describe this image accurately in one concise paragraph, "
                                 "focusing on the main subject, setting, and notable objects. "
                                 "Output only the description."}]}]
    assert enhancer.CAPTION_MAX_TOKENS == 128
    # do_sample=False upstream, which is what temperature 0 is here.
    assert (enhancer.CAPTION_TEMPERATURE, enhancer.CAPTION_TOP_P) == (0.0, 1.0)


def test_the_sampler_is_the_one_wangp_ships_with():
    assert (enhancer.TEMPERATURE, enhancer.TOP_P) == (0.6, 0.9)


# ── a prompt may carry instructions of its own ───────────────────────────────

def test_one_at_sign_adds_to_the_instructions():
    body, suffix, replace = enhancer.split_system_suffix(" a red ball @ keep it to one shot ")
    assert (body, suffix, replace) == ("a red ball", "keep it to one shot", False)

    messages = enhancer.messages("a red ball @ keep it to one shot")
    assert messages[0]["content"] == (
        FL2VA_TEXT_SYSTEM_PROMPT.rstrip() + "\n" + enhancer.SUFFIX_PREAMBLE
        + "\nkeep it to one shot")
    assert messages[1]["content"] == "user_prompt: a red ball"


def test_two_at_signs_replace_them_entirely():
    body, suffix, replace = enhancer.split_system_suffix("a red ball @@ write it in French")
    assert (body, suffix, replace) == ("a red ball", "write it in French", True)

    messages = enhancer.messages("a red ball @@ write it in French")
    assert messages[0]["content"] == "write it in French"


def test_a_prompt_with_no_at_sign_is_left_exactly_as_it_is():
    assert enhancer.split_system_suffix("  a red ball rolls  ") == ("a red ball rolls", "", False)
    assert enhancer.merge_system(FL2VA_TEXT_SYSTEM_PROMPT, "") == \
        FL2VA_TEXT_SYSTEM_PROMPT.rstrip()


# ── what comes back ──────────────────────────────────────────────────────────

def test_the_fields_of_a_prompt_keep_the_blank_lines_between_them():
    """WanGP folds newlines into spaces for its one-prompt-per-line box. An H3
    prompt is fields separated by blank lines, so they are kept."""
    written = ("integrated_multimodal_description: [Shot 1] A red ball rolls.\n\n"
               "overall_soundscape: A low rumble.\n\n"
               "non_diegetic_music: N/A")
    assert enhancer.clean(written) == written


def test_a_fence_or_a_leaked_thought_is_taken_off():
    assert enhancer.clean("```text\nintegrated_multimodal_description: a\n```") == \
        "integrated_multimodal_description: a"
    assert enhancer.clean("<think>hmm</think>overall_soundscape: rain") == \
        "overall_soundscape: rain"
    assert enhancer.clean("  \n") == ""


def test_a_fence_inside_the_prompt_is_not_mistaken_for_a_wrapper():
    written = ("integrated_multimodal_description: a\n\n```not a wrapper```\n\n"
               "non_diegetic_music: N/A")
    assert enhancer.clean(written) == written
