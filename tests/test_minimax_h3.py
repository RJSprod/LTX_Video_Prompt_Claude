"""MiniMax-H3 mode, checked against the guide it was written from.

Two things are being held here, and they are not the same thing.

The first is that the brief this tab produces obeys
``VIDEO_PROMPT_WRITING_GUIDE_base_en``: the three fields in their order, the
alignment line first with exactly one blank line under it, times to two
decimals, ``[Shot 1]`` without a timestamp and every later shot with a strictly
increasing one, speech inside ``<d>[Language] … </d>``, a soundscape that does
not repeat the dialogue, a score described by its instruments. The assertions
below are those rules, one each.

The second is that H3 mode cannot reach the LTX engine. ``PARITY_REPORT.md``
rests on ``prompt_engine.upstream`` being byte-for-byte what was ported, and a
second model's rules leaking into it — even by importing a helper — is exactly
the kind of change that would be invisible until a prompt came out different.
``test_h3_never_touches_the_ltx_engine`` reads the imports and holds the line.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from prompt_master.prompt_engine import minimax_h3 as h3

SOURCE = Path(h3.__file__)


def request(**overrides) -> h3.H3Request:
    base = dict(intent="a courier runs up a wet stairwell", seconds=10.0)
    base.update(overrides)
    return h3.H3Request(**base)


def brief_fields(description="[Shot 1] A stairwell.", soundscape="Rain on a skylight.",
                 music="Low strings hold under it.") -> dict[str, str]:
    return {h3.DESCRIPTION: description, h3.SOUNDSCAPE: soundscape, h3.MUSIC: music}


# ── the two engines never meet ───────────────────────────────────────────────

def test_h3_never_touches_the_ltx_engine():
    """No import of ``upstream``, directly or by way of the adapter.

    Read from the syntax tree rather than by grepping, so a mention in a
    docstring — of which this module has several — is not mistaken for one.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not [name for name in imported if "upstream" in name or "adapter" in name]
    assert not [name for name in imported if name.startswith("PySide6")]


def test_the_ltx_engine_never_hears_about_h3():
    """And the other direction: nothing under ``upstream`` mentions H3 at all."""
    upstream = SOURCE.parent / "upstream"
    guilty = [path.name for path in upstream.glob("*.py")
              if "minimax" in path.read_text(encoding="utf-8").lower()]
    assert guilty == []


# ── the shape of a finished brief ────────────────────────────────────────────

def test_a_text_to_video_brief_is_the_three_fields_and_nothing_before_them():
    built = h3.assemble(brief_fields(), request(mode=h3.T2VA))
    assert built.startswith(f"{h3.DESCRIPTION}:")
    assert h3.alignment(request(mode=h3.T2VA)) == ""


def test_the_fields_keep_the_guides_order_whatever_order_they_arrived_in():
    scrambled = {h3.MUSIC: "Strings.", h3.DESCRIPTION: "[Shot 1] A room.",
                 h3.SOUNDSCAPE: "Rain."}
    built = h3.assemble(scrambled, request())
    assert [line.split(":")[0] for line in built.splitlines() if ":" in line][:3] == list(h3.FIELDS)


def test_a_first_frame_brief_opens_with_one_alignment_line_and_one_blank_line():
    built = h3.assemble(brief_fields(), request(mode=h3.I2VA))
    lines = built.splitlines()
    assert lines[0] == ("For the target video, at 0.00 seconds into the target video, "
                        "<Picture 1> (from [Shot 1]) is fully referenced.")
    assert lines[1] == ""
    assert lines[2].startswith(f"{h3.DESCRIPTION}:")


def test_the_alignment_line_writes_its_times_to_exactly_two_decimals():
    line = h3.alignment(request(mode=h3.FL2VA, seconds=12.0, shots="3"))
    assert "at 0.00 seconds" in line
    assert "at 12.00 seconds" in line
    assert "12.000 seconds" not in line


def test_a_pinned_shot_count_numbers_the_last_frames_shot_and_auto_names_it():
    assert "<Picture 2> (from [Shot 3])" in h3.alignment(request(mode=h3.FL2VA, shots="3"))
    assert "<Picture 2> (from the final shot)" in h3.alignment(request(mode=h3.FL2VA, shots="auto"))


def test_a_missing_score_is_answered_rather_than_dropped():
    """The format is three fields. Two of them is not a shorter brief."""
    built = h3.assemble({h3.DESCRIPTION: "[Shot 1] A room.", h3.SOUNDSCAPE: "Rain."},
                        request(music="off"))
    assert h3.NO_MUSIC in built
    assert built.count(f"{h3.MUSIC}:") == 1


# ── budgets ──────────────────────────────────────────────────────────────────

def test_timecode_is_the_forms_the_guide_uses():
    assert h3.timecode(0) == "00:00.000"
    assert h3.timecode(3) == "00:03.000"
    assert h3.timecode(8.5) == "00:08.500"
    assert h3.timecode(63.25) == "01:03.250"


def test_a_pinned_shot_count_is_exact_and_auto_grows_with_the_duration():
    assert h3.shot_range(request(shots="2")) == (2, 2)
    assert h3.shot_range(request(shots="auto", seconds=5)) == (1, 2)
    assert h3.shot_range(request(shots="auto", seconds=15)) == (2, 4)
    short = h3.shot_range(request(shots="auto", seconds=6))
    long = h3.shot_range(request(shots="auto", seconds=14))
    assert long[1] > short[1]


def test_the_word_budget_grows_with_the_duration_and_never_collapses():
    low, high = h3.word_budget(10)
    assert low < high
    assert h3.word_budget(15)[1] > high
    assert h3.word_budget(0)[0] >= 90


def test_the_duration_is_held_to_the_five_to_fifteen_seconds_h3_accepts():
    assert h3.clamp_seconds(2) == h3.MIN_SECONDS
    assert h3.clamp_seconds(40) == h3.MAX_SECONDS
    assert h3.clamp_seconds("nonsense") == 10.0


def test_speech_needs_a_voice_to_be_assigned_to():
    """A talk budget with nobody to speak is a disagreement, and the count wins:
    a line has to belong to an ID, and there is no ID to give it."""
    assert h3.spoken_lines(request(speakers=0, talk="dense")) == 0
    assert h3.spoken_lines(request(speakers=2, talk="none")) == 0
    assert h3.spoken_lines(request(speakers=2, talk="sparse")) >= 1
    assert (h3.spoken_lines(request(speakers=2, talk="dense"))
            > h3.spoken_lines(request(speakers=2, talk="sparse")))


def test_the_token_ceiling_clears_the_longest_brief_the_budget_allows():
    for seconds in (5, 10, 15):
        assert h3.max_tokens(request(seconds=seconds)) > h3.word_budget(seconds)[1] * 2


# ── reading the writer back ──────────────────────────────────────────────────

def test_labelled_text_is_read_field_by_field():
    fields = h3.parse("integrated_multimodal_description: [Shot 1] A room.\n\n"
                      "overall_soundscape: Rain.\n\n"
                      "non_diegetic_music: Strings.")
    assert fields == {h3.DESCRIPTION: "[Shot 1] A room.", h3.SOUNDSCAPE: "Rain.",
                      h3.MUSIC: "Strings."}


@pytest.mark.parametrize("label", [
    "**integrated_multimodal_description:**",
    "## Integrated Multimodal Description",
    "- integrated multimodal description -",
])
def test_a_label_survives_whatever_the_writer_decorated_it_with(label):
    assert h3.parse(f"{label}\n[Shot 1] A room.").get(h3.DESCRIPTION) == "[Shot 1] A room."


def test_a_json_object_is_read_as_the_object_it_is():
    fields = h3.parse('{"integrated_multimodal_description": "[Shot 1] A room.", '
                      '"overall_soundscape": "Rain.", "non_diegetic_music": "Strings."}')
    assert fields[h3.DESCRIPTION] == "[Shot 1] A room."
    assert fields[h3.MUSIC] == "Strings."


def test_a_fence_and_a_reasoning_leak_are_stripped_before_anything_is_read():
    fields = h3.parse("<think>let me plan</think>\n```json\n"
                      '{"integrated_multimodal_description": "[Shot 1] A room."}\n```')
    assert fields == {h3.DESCRIPTION: "[Shot 1] A room."}


def test_prose_that_ignored_the_format_is_kept_as_the_description():
    """Nothing is thrown away for being the wrong shape — it is put where the
    checks below will notice what is missing from it."""
    assert h3.parse("[Shot 1] A room, and nobody labelled anything.") == {
        h3.DESCRIPTION: "[Shot 1] A room, and nobody labelled anything."}


def test_nothing_at_all_reads_as_nothing_at_all():
    assert h3.parse("") == {}
    assert h3.parse("   \n  ") == {}


# ── the checks ───────────────────────────────────────────────────────────────

def clean() -> tuple[str, dict, h3.H3Request]:
    asked = request(seconds=12.0, shots="3")
    fields = brief_fields(
        "[Shot 1] Cinematic live-action, a courier at the foot of a stairwell.\n"
        "[Shot 2] At 00:04.000, the camera cuts to the landing.\n"
        "[Shot 3] At 00:08.500, the shot cuts to a door swinging open.")
    return h3.assemble(fields, asked), fields, asked


def test_a_brief_that_follows_the_rules_is_flagged_for_nothing():
    assert h3.checks(*clean()) == []


def test_a_first_shot_that_grew_a_timestamp_is_flagged():
    _, fields, asked = clean()
    fields[h3.DESCRIPTION] = "[Shot 1] At 00:00.000, a courier at the stairwell."
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("first shot never does" in note for note in notes)


def test_cut_times_that_go_backwards_are_flagged():
    _, fields, asked = clean()
    fields[h3.DESCRIPTION] = ("[Shot 1] A stairwell.\n[Shot 2] At 00:06.000, a landing.\n"
                              "[Shot 3] At 00:02.000, a door.")
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("not after the shot before it" in note for note in notes)


def test_a_cut_at_or_past_the_end_is_flagged():
    _, fields, asked = clean()
    fields[h3.DESCRIPTION] = "[Shot 1] A stairwell.\n[Shot 2] At 00:20.000, a landing."
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("past the 12 s end" in note for note in notes)


def test_a_later_shot_with_no_cut_time_is_flagged():
    _, fields, asked = clean()
    fields[h3.DESCRIPTION] = "[Shot 1] A stairwell.\n[Shot 2] The camera cuts to a landing."
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("no cut time" in note for note in notes)


def test_shots_that_do_not_start_at_one_or_run_in_order_are_flagged():
    _, fields, asked = clean()
    fields[h3.DESCRIPTION] = "[Shot 2] A stairwell.\n[Shot 4] At 00:04.000, a landing."
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("numbered from 1" in note for note in notes)


def test_a_description_with_no_shot_marker_at_all_is_flagged():
    asked = request()
    fields = brief_fields("A courier climbs a stairwell in one unbroken take.")
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("[Shot 1]" in note for note in notes)


def test_a_missing_field_is_flagged_by_name():
    asked = request()
    fields = {h3.DESCRIPTION: "[Shot 1] A room."}
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any(h3.SOUNDSCAPE in note and "missing" in note for note in notes)


def test_a_brief_over_the_api_limit_is_flagged_with_its_length():
    asked = request()
    fields = brief_fields("[Shot 1] " + ("a very wet stairwell, " * 700))
    brief = h3.assemble(fields, asked)
    notes = h3.checks(brief, fields, asked)
    assert any(str(h3.CHARACTER_LIMIT) in note for note in notes)


def test_the_sound_fields_are_held_to_their_sentence_counts():
    asked = request()
    long_ambience = brief_fields(soundscape="A. B. C. D. E.")
    assert any("1 to 4" in note for note in h3.checks("", long_ambience, asked))
    long_score = brief_fields(music="A. B. C. D.")
    assert any("1 to 3" in note for note in h3.checks("", long_score, asked))


def test_speech_that_was_asked_for_and_never_written_is_flagged():
    asked = request(speakers=2, talk="steady")
    fields = brief_fields("[Shot 1] The courier climbs, saying nothing at all.")
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("<d>" in note for note in notes)
    assert any("speaker ID" in note for note in notes)


def test_an_unclosed_speech_tag_is_flagged():
    asked = request(speakers=1, talk="sparse")
    fields = brief_fields("[Shot 1] The courier (S1) says: <d>[English] Almost there.")
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("unclosed" in note for note in notes)


def test_speech_written_without_the_language_tag_is_flagged():
    asked = request(speakers=1, talk="sparse", language="Japanese")
    fields = brief_fields("[Shot 1] The courier (S1) says: <d>[English] Almost there.</d>")
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("[Japanese]" in note for note in notes)


def test_well_formed_speech_passes():
    asked = request(seconds=10.0, shots="1", speakers=1, talk="sparse")
    fields = brief_fields("[Shot 1] The courier (S1) says: <d>[English] Almost there.</d>")
    assert h3.checks(h3.assemble(fields, asked), fields, asked) == []


# ── the instruction the model is given ───────────────────────────────────────

def test_the_output_contract_names_the_three_fields_in_order():
    system = h3.build_system(request())
    positions = [system.index(f"{label}:") for label in h3.FIELDS]
    assert positions == sorted(positions)


def test_a_frame_anchored_brief_is_told_not_to_write_its_own_alignment_line():
    """It is added afterwards, and two of them is worse than none."""
    assert "alignment or reference line of your own" in h3.build_system(request(mode=h3.I2VA))
    assert "alignment or reference line of your own" not in h3.build_system(request(mode=h3.T2VA))


def test_the_speech_syntax_appears_only_when_somebody_speaks():
    silent = h3.build_system(request(speakers=0))
    spoken = h3.build_system(request(speakers=2, talk="steady", language="Spanish"))
    # The silent brief still names the tag, to forbid it. What it must not carry
    # is the syntax — the IDs, the example line, the identity rules.
    assert "Nobody speaks" in silent
    assert "(S1,S2)" not in silent and "stable ID" not in silent
    assert "(S1,S2)" in spoken
    assert "<d>[Spanish]" in spoken


def test_a_deliberate_transition_is_written_and_a_straight_cut_offers_the_alternates():
    dissolve = h3.build_system(request(shots="3", cut="cross_dissolve"))
    assert "the shot cross-dissolves to" in dissolve
    straight = h3.build_system(request(shots="3", cut="cut"))
    for phrase in h3.CUT_ALTERNATES:
        assert phrase in straight


def test_a_single_shot_brief_is_not_given_the_cutting_rules_at_all():
    system = h3.build_system(request(shots="1"))
    assert "[Shot 2] At 00:03.000" not in system


def test_the_camera_is_asked_for_as_an_action_and_only_marked_when_unusual():
    plain = h3.build_system(request(camera="push_in"))
    assert "Camera: Push In." in plain
    marked = h3.build_system(request(camera="push_in", amplitude="large", speed="slow"))
    assert "Camera: Push In, large amplitude, slow." in marked


def test_the_style_is_asked_for_at_the_start_of_the_first_shot():
    assert "Style: claymation." in h3.build_system(request(style="claymation"))
    assert "Style:" not in h3.build_system(request(style="auto"))


def test_the_score_block_says_what_to_write_when_there_is_no_score():
    assert h3.NO_MUSIC in h3.build_system(request(music="off"))
    scored = h3.build_system(request(music="score", music_brief="low strings"))
    assert "low strings" in scored
    assert "never name a song" in scored


def test_the_cast_and_the_on_screen_text_appear_only_when_given():
    bare = h3.build_system(request())
    assert "THE CAST" not in bare and "ON-SCREEN TEXT" not in bare
    full = h3.build_system(request(cast="Mara = a courier in a soaked jacket",
                                   on_screen_text="FLOOR 12"))
    assert "Mara = a courier in a soaked jacket" in full
    assert "FLOOR 12" in full


def test_each_mode_gets_the_law_about_the_frames_it_actually_has():
    assert "THE FIRST FRAME" in h3.build_system(request(mode=h3.I2VA))
    assert "THE FIRST AND LAST FRAMES" in h3.build_system(request(mode=h3.FL2VA))
    text_only = h3.build_system(request(mode=h3.T2VA))
    assert "THE FIRST FRAME" not in text_only


def test_the_user_turn_carries_the_intent_and_says_what_the_pictures_are():
    assert "a courier runs up a wet stairwell" in h3.build_user(request())
    assert "first frame" in h3.build_user(request(mode=h3.I2VA))
    assert "first and the last frame" in h3.build_user(request(mode=h3.FL2VA))


# ── the frames on the wire ───────────────────────────────────────────────────

def test_a_text_only_request_sends_plain_text():
    turns = h3.messages(request(mode=h3.T2VA))
    assert turns[0]["role"] == "system"
    assert isinstance(turns[1]["content"], str)


def test_the_frames_go_first_and_in_order():
    turns = h3.messages(request(mode=h3.FL2VA, first_frame="data:image/jpeg;base64,AAA",
                                last_frame="data:image/jpeg;base64,BBB"))
    parts = turns[1]["content"]
    assert [part["type"] for part in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].endswith("AAA")
    assert parts[1]["image_url"]["url"].endswith("BBB")


def test_a_mode_that_needs_a_frame_refuses_to_go_without_one():
    with pytest.raises(h3.VisionUnavailable):
        h3.messages(request(mode=h3.I2VA))
    with pytest.raises(h3.VisionUnavailable):
        h3.messages(request(mode=h3.FL2VA, first_frame="data:image/jpeg;base64,AAA"))


def test_a_frame_is_never_quietly_dropped_when_nothing_can_see_it():
    """The alignment line at the top of the brief states that the picture is the
    first frame. Sending it to a model that cannot see the picture makes that
    line a false statement about a video nobody chose."""
    with pytest.raises(h3.VisionUnavailable):
        h3.messages(request(mode=h3.I2VA, first_frame="data:image/jpeg;base64,AAA"),
                    vision_available=False)


# ── the option lists the UI is built from ────────────────────────────────────

def test_every_option_list_is_pairs_of_key_and_label():
    for options in (h3.MODES, h3.STYLES, h3.CAMERAS, h3.AMPLITUDES, h3.SPEEDS, h3.CUTS,
                    h3.SHOTS, h3.LANGUAGES, h3.TALK, h3.MUSIC_MODES):
        assert options
        assert all(isinstance(value, str) and isinstance(label, str) and value
                   for value, label in options)
        keys = [value for value, _ in options]
        assert len(keys) == len(set(keys))


def test_the_camera_list_is_h3s_own_vocabulary_and_nothing_invented():
    labels = [label for value, label in h3.CAMERAS if value != "auto"]
    assert labels == list(h3.CAMERA_MOVES)


def test_every_cut_has_a_phrase_and_every_style_that_is_not_auto_has_one_too():
    assert set(dict(h3.CUTS)) == set(h3.CUT_PHRASES)
    assert set(value for value, _ in h3.STYLES if value != "auto") == set(h3.STYLE_PHRASES)


def test_every_talk_setting_has_a_rate():
    assert set(value for value, _ in h3.TALK) == set(h3.TALK_RATE)
