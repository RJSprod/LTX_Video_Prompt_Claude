"""The dialogue slider, checked where it could quietly stop being what it claims.

Four things are asserted here, and they are the four that would make this
feature something other than a slider on top of an untouched port:

*Off is off.* Not "nearly the same request" — the same request, element for
element, and the same token budget WanGP set. A control nobody touched has to
leave the H3 prompt exactly as it was before this file existed.

*The vendored instructions stay vendored.* The directive is appended after them,
never woven through them, so removing it removes the whole of the difference.

*A position on the slider means something.* Off through 10 has to be
monotonic in both of the quantities it drives, and 10 has to actually mean
"talking throughout" rather than "more than 9".

*What comes back is checked, not trusted.* The pass runs a model, and a model
returns prose, refusals, missing brackets and lines it already wrote. None of
that may reach an H3 prompt.
"""

from __future__ import annotations

import pytest

from prompt_master.minimax import dialogue, enhancer
from prompt_master.minimax.prompt_enhancer import (FL2VA_TEXT_SYSTEM_PROMPT,
                                                   REF2VA_IMAGE_SYSTEM_PROMPT)

SCENE = "A woman in a red coat waits at a rain-streaked window."

REPLY = """SPEAKERS
(S1) = the woman in the red coat
(S2) = unseen narrator

LINES
(S1) [quiet, uncertain] "You said you would be back before dark."
(S2) [measured, close] "She had waited three winters for that sentence."
(S1) [sharper now] "I am not asking you again, not tonight."
"""


def cast_and_script(reply: str = REPLY, count: int = 10):
    return dialogue.parse(reply, [], count)


def scripted(reply: str):
    """A chat_stream that answers with ``reply`` and records what it was asked."""
    seen: list[list[dict]] = []

    def chat_stream(messages, **_kwargs):
        seen.append(messages)
        return [reply]

    return chat_stream, seen


# ── off is off ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", [enhancer.FL2VA, enhancer.REF2VA])
@pytest.mark.parametrize("caption", [None, "A red ball on a table."])
def test_at_off_the_request_is_the_one_the_enhancer_builds(variant, caption):
    """The whole claim of the default position: WanGP's request, untouched."""
    assert dialogue.request(SCENE, variant=variant, image_caption=caption) == \
        enhancer.messages(SCENE, variant=variant, image_caption=caption)


def test_at_off_the_token_budget_is_wangps_own():
    assert dialogue.max_tokens(enhancer.FL2VA, 0, dialogue.OFF) == 1024
    assert dialogue.max_tokens(enhancer.REF2VA, 0, dialogue.OFF) == 2048
    # Lines that were somehow written but not asked for buy no extra budget.
    assert dialogue.max_tokens(enhancer.FL2VA, 12, dialogue.OFF) == 1024


def test_at_off_the_at_sign_still_works_exactly_as_it_did():
    assert dialogue.request(f"{SCENE} @ keep it to one shot") == \
        enhancer.messages(f"{SCENE} @ keep it to one shot")
    assert dialogue.request(f"{SCENE} @@ write it in French") == \
        enhancer.messages(f"{SCENE} @@ write it in French")


def test_at_off_no_pass_runs_at_all():
    roster, lines, note = dialogue.write(
        SCENE, dialogue.OFF, lambda *a, **k: pytest.fail("no pass at Off"))
    assert (roster, lines, note) == ([], [], "")


def test_an_empty_prompt_asks_the_model_nothing():
    roster, lines, note = dialogue.write(
        "   ", dialogue.MOST, lambda *a, **k: pytest.fail("nothing to write dialogue about"))
    assert (roster, lines, note) == ([], [], "")


# ── the vendored instructions stay vendored ──────────────────────────────────

def test_the_directive_is_appended_after_the_instructions_never_into_them():
    """Cut the directive off the end and the vendored text is what is left."""
    _, lines = cast_and_script()
    system = dialogue.request(SCENE, intensity=6, lines=lines)[0]["content"]
    assert system.startswith(FL2VA_TEXT_SYSTEM_PROMPT.rstrip())
    added = system[len(FL2VA_TEXT_SYSTEM_PROMPT.rstrip()):]
    assert added.lstrip("\n") == dialogue.directive(6, len(lines))


def test_the_directive_names_the_field_the_chosen_variant_actually_writes():
    fl2va = dialogue.directive(5, 3, variant=enhancer.FL2VA)
    ref2va = dialogue.directive(5, 3, variant=enhancer.REF2VA)
    assert "integrated_multimodal_description" in fl2va and "detailed_description" not in fl2va
    assert "detailed_description" in ref2va and "integrated_multimodal_description" not in ref2va


def test_an_image_still_chooses_the_image_instructions():
    _, lines = cast_and_script()
    system = dialogue.request(SCENE, variant=enhancer.REF2VA, image_caption="A window.",
                              intensity=4, lines=lines)[0]["content"]
    assert system.startswith(REF2VA_IMAGE_SYSTEM_PROMPT.rstrip())


def test_the_users_own_at_instructions_are_still_the_last_word():
    """WanGP's preamble promises a prompt's own instructions higher priority.
    The directive goes in ahead of them so that stays true."""
    _, lines = cast_and_script()
    system = dialogue.request(f"{SCENE} @ keep it to one shot", intensity=9,
                              lines=lines)[0]["content"]
    assert system.index(dialogue.HEADER) < system.index(enhancer.SUFFIX_PREAMBLE)
    assert system.endswith("keep it to one shot")


def test_two_at_signs_replace_wangps_instructions_but_not_this_pages_control():
    """``@@`` is aimed at the vendored instructions. The slider is not those —
    it was deliberately moved on this page — so it survives, ahead of the
    replacement text, which keeps the user's own words last."""
    _, lines = cast_and_script()
    system = dialogue.request(f"{SCENE} @@ write it in French", intensity=7,
                              lines=lines)[0]["content"]
    assert FL2VA_TEXT_SYSTEM_PROMPT.rstrip() not in system
    assert system == f"{dialogue.directive(7, len(lines))}\nwrite it in French"


# ── a position on the slider means something ─────────────────────────────────

def test_both_quantities_the_slider_drives_only_ever_go_up():
    shares = [dialogue.share(i) for i in range(dialogue.OFF, dialogue.MOST + 1)]
    counts = [dialogue.target_lines(i) for i in range(dialogue.OFF, dialogue.MOST + 1)]
    assert shares == sorted(shares) and counts == sorted(counts)
    assert shares[0] == 0 and counts[0] == 0
    assert shares[1] > 0 and counts[1] > 0
    assert counts[-1] == dialogue.LINE_CEILING


def test_the_far_end_asks_for_a_voice_that_does_not_stop():
    """The one position with its own wording: a percentage is a share of the
    running time, and "essentially all of it" is not a share."""
    assert dialogue.CONTINUOUS in dialogue.directive(dialogue.MOST, 4)
    assert dialogue.CONTINUOUS not in dialogue.directive(1, 4)
    assert "no silence longer than about half a second" in dialogue.CONTINUOUS


def test_speech_the_prompt_already_has_counts_toward_the_target():
    """Somebody who typed six lines and set the slider low did not ask for six
    more of them."""
    assert dialogue.wanted(dialogue.MOST, 0) == dialogue.LINE_CEILING
    assert dialogue.wanted(dialogue.MOST, 4) == dialogue.LINE_CEILING - 4
    assert dialogue.wanted(1, 9) == 0                    # already past the target
    assert dialogue.wanted(dialogue.OFF, 0) == 0


def test_a_prompt_that_already_speaks_enough_is_left_alone_but_still_directed():
    talkative = " ".join(f'"Line number {n} of this scene here."' for n in range(6))
    roster, lines, note = dialogue.write(
        talkative, 1, lambda *a, **k: pytest.fail("nothing more to write"))
    assert (roster, lines) == ([], [])
    assert note == "the prompt already speaks this much"
    # The intensity is still a thing the user asked for, and the enhancer can
    # act on it with no lines of ours at all.
    assert dialogue.directive(1, 0) != ""


def test_with_no_lines_the_directive_asks_the_enhancer_to_write_them_itself():
    assert dialogue.INVENT in dialogue.directive(5, 0)
    assert dialogue.SUPPLIED not in dialogue.directive(5, 0)
    assert dialogue.SUPPLIED in dialogue.directive(5, 3)


def test_the_budget_grows_with_the_speech_and_stops_growing():
    """A timeline carrying twenty-four lines does not fit in the budget for a
    timeline carrying none, and would be cut off mid-sentence."""
    assert dialogue.max_tokens(enhancer.FL2VA, 8, 5) > enhancer.max_tokens(enhancer.FL2VA)
    assert dialogue.max_tokens(enhancer.REF2VA, 24, dialogue.MOST) <= dialogue.TOKEN_CEILING


def test_a_slider_position_that_is_not_one_falls_back_to_off():
    for value in (None, "", -3, 99, "loud"):
        assert dialogue.clamp(value) in (dialogue.OFF, dialogue.MOST)
    assert dialogue.clamp(None) == dialogue.OFF
    assert dialogue.clamp(-3) == dialogue.OFF
    assert dialogue.clamp(99) == dialogue.MOST


# ── what the pass is asked, and what is believed of its answer ───────────────

def test_the_pass_is_told_the_scene_the_picture_and_the_pace():
    chat_stream, seen = scripted(REPLY)
    dialogue.write(SCENE, dialogue.MOST, chat_stream, image_caption="Rain on glass.")
    system, user = seen[0]
    assert system["content"] is dialogue.SYSTEM
    assert f"VIDEO: {SCENE}" in user["content"]
    assert "WHAT THE OPENING PICTURE SHOWS: Rain on glass." in user["content"]
    assert dialogue.pace(dialogue.MOST) in user["content"]
    assert f"Write the speakers and {dialogue.LINE_CEILING} lines now." in user["content"]


def test_speech_already_in_the_prompt_is_shown_to_the_pass_so_it_can_match_it():
    chat_stream, seen = scripted(REPLY)
    dialogue.write(f'{SCENE} She says "Not tonight, not ever again."', 5, chat_stream)
    assert 'LINES ALREADY SPOKEN IN IT:\n"Not tonight, not ever again."' in seen[0][1]["content"]


def test_the_pass_is_asked_about_the_video_not_about_the_prompts_own_instructions():
    chat_stream, seen = scripted(REPLY)
    dialogue.write(f"{SCENE} @@ write it in French", 5, chat_stream)
    assert "write it in French" not in seen[0][1]["content"]
    assert f"VIDEO: {SCENE}" in seen[0][1]["content"]


def test_a_reply_becomes_a_cast_and_a_script():
    roster, lines = cast_and_script()
    assert roster == [("S1", "the woman in the red coat"), ("S2", "unseen narrator")]
    assert [line.speaker for line in lines] == ["S1", "S2", "S1"]
    assert lines[0].delivery == "quiet, uncertain"
    assert lines[0].text == "You said you would be back before dark."
    assert str(lines[0]) == '(S1) [quiet, uncertain] "You said you would be back before dark."'


def test_the_delivery_bracket_never_reaches_the_words_that_are_spoken():
    """H3 puts only the exact spoken content inside ``<d>``. A delivery cue that
    got in there would be read out loud."""
    _, lines = cast_and_script()
    for line in lines:
        assert "[" not in line.text and "]" not in line.text
    system = dialogue.request(SCENE, intensity=6, lines=lines)[0]["content"]
    assert dialogue.DELIVERY in system


def test_prose_refusals_and_repeats_do_not_reach_an_h3_prompt():
    reply = ('Here are the lines you asked for.\n'
             '(S1) [flat] "Go."\n'                            # too short
             '(S1) [warm] "This one is a perfectly usable line."\n'
             '(S1) [warm] "This one is a perfectly usable line."\n'   # repeated
             f'(S2) [droning] "{" word" * 40}"\n'              # too long
             'I hope these work for your scene.\n')
    _, lines = dialogue.parse(reply, [], 10)
    assert [line.text for line in lines] == ["This one is a perfectly usable line."]


def test_a_line_the_prompt_already_had_is_not_handed_back_as_a_new_one():
    existing = ["You said you would be back before dark."]
    _, lines = dialogue.parse(REPLY, existing, 10)
    assert existing[0] not in [line.text for line in lines]
    assert len(lines) == 2


def test_a_speaker_who_ends_up_saying_nothing_is_dropped_from_the_cast():
    """Naming somebody and then never having them speak is an invitation to
    write them a line, and the line count is what the slider controls."""
    roster, lines = dialogue.parse(REPLY, [], 1)
    assert [line.speaker for line in lines] == ["S1"]
    assert roster == [("S1", "the woman in the red coat")]


def test_a_reply_that_lost_its_speaker_ids_is_still_a_script():
    """A monologue is a usable answer where an empty pane is not."""
    roster, lines = dialogue.parse('"The first thing that anyone said."\n'
                                   '"And then the second thing, after it."', [], 10)
    assert roster == []
    assert [line.speaker for line in lines] == ["S1", "S1"]


def test_no_more_lines_come_back_than_were_asked_for():
    reply = "\n".join(f'(S1) [flat] "This is spoken line number {n} of many."'
                      for n in range(30))
    _, lines = dialogue.parse(reply, [], 4)
    assert len(lines) == 4


def test_a_pass_that_fails_costs_the_speech_and_never_the_prompt():
    def explodes(*_args, **_kwargs):
        raise RuntimeError("llama-server went away")

    roster, lines, note = dialogue.write(SCENE, 5, explodes)
    assert (roster, lines) == ([], [])
    assert note == "skipped (llama-server went away)"
    # And the request still carries the intensity that was asked for.
    assert dialogue.directive(5, 0) in dialogue.request(SCENE, intensity=5)[0]["content"]


def test_a_pass_that_comes_back_with_nothing_usable_says_so():
    chat_stream, _ = scripted("I am unable to help with this request.")
    roster, lines, note = dialogue.write(SCENE, 5, chat_stream)
    assert (roster, lines) == ([], [])
    assert note == "none came back — the intensity still applies"


def test_the_note_counts_the_lines_and_the_voices():
    chat_stream, _ = scripted(REPLY)
    _, lines, note = dialogue.write(SCENE, dialogue.MOST, chat_stream)
    assert len(lines) == 3
    assert note == "3 lines in 2 voices"

    one_voice = '(S1) [flat] "Only ever the one person speaking here."'
    chat_stream, _ = scripted(one_voice)
    _, _, note = dialogue.write(SCENE, dialogue.MOST, chat_stream)
    assert note == "1 lines in one voice"


# ── how the script reaches the request ───────────────────────────────────────

def test_the_script_travels_as_part_of_what_the_user_asked_for():
    """H3 takes its content from ``user_prompt:``. Lines bolted onto the system
    message would be instructions about a video rather than content of one."""
    roster, lines = cast_and_script()
    user = dialogue.request(SCENE, intensity=8, roster=roster, lines=lines)[1]["content"]
    assert user.startswith(f"user_prompt: {SCENE}")
    assert dialogue.transcript(roster, lines) in user
    assert "(S2) = unseen narrator" in user


def test_the_caption_line_is_still_the_second_line_of_the_same_turn():
    roster, lines = cast_and_script()
    user = dialogue.request(SCENE, image_caption="Rain on glass.", intensity=8,
                            roster=roster, lines=lines)[1]["content"]
    assert user.endswith("\nimage_caption: Rain on glass.")
