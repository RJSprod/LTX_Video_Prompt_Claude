"""MiniMax-H3 mode, checked against the guide it was written from.

Two things are being held here, and they are not the same thing.

The first is that the brief this mode produces obeys the *Video Prompt Writing
Guide (T2VA / I2VA / FL2VA / L2VA)*: the three fields in their order, each
mode's instruction line in the guide's own wording — down to where it brackets
``<Picture 1>`` and where it does not — naming the real final shot and the
duration to two decimals; ``[Shot 1]`` without a timestamp and every later shot
with a strictly increasing one; camera motion as an action with the guide's
amplitude and speed phrases; speech inside ``<d>[Language] … </d>`` with stable
IDs; a soundscape that does not repeat the dialogue; a score described by its
instruments; and ``N/A`` — the token, not a sentence — wherever a sound field
has nothing to say. The assertions below are those rules, one each.

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
                 music=h3.NOT_APPLICABLE) -> dict[str, str]:
    """A well-formed set of fields for the default request, which asks for no
    score — so the score field is the token the guide gives for that."""
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


# ── the four tasks ───────────────────────────────────────────────────────────

def test_the_guides_four_modes_are_all_offered():
    assert [value for value, _ in h3.MODES] == [h3.T2VA, h3.I2VA, h3.FL2VA, h3.L2VA]


def test_which_frames_each_mode_anchors_to():
    """L2VA is the one that catches a wrong assumption: it has a picture, and
    that picture is the *last* frame, so "has a first frame" is not the same
    question as "has a picture"."""
    assert not h3.anchored(h3.T2VA)
    assert (h3.needs_first_frame(h3.I2VA), h3.needs_last_frame(h3.I2VA)) == (True, False)
    assert (h3.needs_first_frame(h3.FL2VA), h3.needs_last_frame(h3.FL2VA)) == (True, True)
    assert (h3.needs_first_frame(h3.L2VA), h3.needs_last_frame(h3.L2VA)) == (False, True)
    assert h3.anchored(h3.L2VA)


# ── the instruction line, in the guide's own wording ─────────────────────────

def test_a_text_to_video_brief_has_no_instruction_and_starts_at_the_first_field():
    built = h3.assemble(brief_fields(), request(mode=h3.T2VA))
    assert h3.instruction(request(mode=h3.T2VA)) == ""
    assert built.startswith(f"{h3.DESCRIPTION}:")


def test_the_first_frame_instruction_is_the_guides_wording():
    assert h3.instruction(request(mode=h3.I2VA)) == (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.")


def test_the_first_and_last_frame_instruction_is_the_guides_wording():
    """Reproduced from the guide's Case 3, including that FL2VA alone writes
    ``Picture 1 (from Shot 1)`` bare where the other two bracket both."""
    assert h3.instruction(request(mode=h3.FL2VA, seconds=8.0), final_shot=1) == (
        "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
        "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) "
        "aligns with the 8.00-second mark of the target video.")


def test_the_last_frame_instruction_is_the_guides_wording():
    """The guide's Case 4."""
    assert h3.instruction(request(mode=h3.L2VA, seconds=6.0), final_shot=1) == (
        "How the reference pictures align with the target video — <Picture 1> "
        "(from [Shot 1]) aligns with the 6.00-second mark of the target video.")


def test_the_instruction_writes_its_duration_to_exactly_two_decimals():
    for seconds, mark in ((8.0, "8.00-second"), (12.5, "12.50-second"), (6.0, "6.00-second")):
        assert mark in h3.instruction(request(mode=h3.L2VA, seconds=seconds))


def test_the_instruction_names_the_shot_the_writer_actually_ended_on():
    """The guide's ``N`` is the index of the *actual* final shot, so the line is
    built from the brief that came back rather than from what was asked for."""
    asked = request(mode=h3.FL2VA, seconds=8.0, shots="2")
    fields = brief_fields("[Shot 1] A cyclist under a closed umbrella.\n"
                          "[Shot 2] At 00:04.000, the camera cuts to the canopy opening.")
    assert "Picture 2 (from Shot 2)" in h3.assemble(fields, asked)
    single = brief_fields("[Shot 1] A cyclist, one continuous move to the canopy.")
    assert "Picture 2 (from Shot 1)" in h3.assemble(single, request(mode=h3.FL2VA, seconds=8.0))


def test_an_anchored_brief_puts_the_instruction_first_with_one_blank_line_under_it():
    lines = h3.assemble(brief_fields(), request(mode=h3.I2VA)).splitlines()
    assert lines[0].startswith("For the target video,")
    assert lines[1] == ""
    assert lines[2].startswith(f"{h3.DESCRIPTION}:")


def test_the_fields_keep_the_guides_order_whatever_order_they_arrived_in():
    scrambled = {h3.MUSIC: h3.NOT_APPLICABLE, h3.DESCRIPTION: "[Shot 1] A room.",
                 h3.SOUNDSCAPE: "Rain."}
    built = h3.assemble(scrambled, request())
    assert [line.split(":")[0] for line in built.splitlines() if ":" in line][:3] == list(h3.FIELDS)


# ── N/A is a value, not a way out ────────────────────────────────────────────

def test_no_score_is_the_token_rather_than_a_sentence_about_there_being_none():
    built = h3.assemble({h3.DESCRIPTION: "[Shot 1] A room.", h3.SOUNDSCAPE: "Rain."},
                        request(music="off"))
    assert f"{h3.MUSIC}: {h3.NOT_APPLICABLE}" in built
    assert built.count(f"{h3.MUSIC}:") == 1


def test_complete_silence_is_the_one_case_that_takes_the_token_in_the_soundscape():
    silent = request(ambience="silence")
    built = h3.assemble({h3.DESCRIPTION: "[Shot 1] A room."}, silent)
    assert f"{h3.SOUNDSCAPE}: {h3.NOT_APPLICABLE}" in built
    assert h3.NOT_APPLICABLE in h3.build_system(silent)


def test_a_soundscape_of_N_A_that_nobody_asked_for_is_flagged():
    fields = brief_fields(soundscape=h3.NOT_APPLICABLE)
    notes = h3.checks("", fields, request(ambience="scene"))
    assert any("only when complete silence" in note for note in notes)


def test_silence_that_was_asked_for_and_then_described_anyway_is_flagged():
    fields = brief_fields(soundscape="Rain taps the skylight.")
    notes = h3.checks("", fields, request(ambience="silence"))
    assert any("should be N/A" in note for note in notes)


def test_a_score_that_was_asked_for_and_came_back_N_A_is_flagged():
    fields = brief_fields(music=h3.NOT_APPLICABLE)
    notes = h3.checks("", fields, request(music="score"))
    assert any("should not be N/A" in note for note in notes)


def test_a_score_nobody_asked_for_is_flagged():
    fields = brief_fields(music="Low strings hold under the climb.")
    notes = h3.checks("", fields, request(music="off"))
    assert any("should be N/A" in note for note in notes)


def test_the_token_is_never_measured_in_sentences():
    fields = brief_fields(soundscape=h3.NOT_APPLICABLE, music=h3.NOT_APPLICABLE)
    notes = h3.checks("", fields, request(ambience="silence", music="off"))
    assert not [note for note in notes if "sentences" in note]


# ── budgets ──────────────────────────────────────────────────────────────────

def test_timecode_is_the_form_the_guide_uses():
    assert h3.timecode(0) == "00:00.000"
    assert h3.timecode(3.5) == "00:03.500"
    assert h3.timecode(5) == "00:05.000"
    assert h3.timecode(63.25) == "01:03.250"


def test_the_two_interpolating_modes_default_to_a_single_shot():
    """The guide's instruction, not a preference: one shot is what lets the
    model interpolate continuously towards the frame it has to land on."""
    assert h3.shot_range(request(mode=h3.FL2VA, seconds=15)) == (1, 1)
    assert h3.shot_range(request(mode=h3.L2VA, seconds=15)) == (1, 1)
    # Unless more were explicitly specified, which pinning a number is.
    assert h3.shot_range(request(mode=h3.FL2VA, shots="3")) == (3, 3)


def test_a_pinned_shot_count_is_exact_and_auto_grows_with_the_duration():
    assert h3.shot_range(request(shots="2")) == (2, 2)
    assert h3.shot_range(request(shots="auto", seconds=5)) == (1, 2)
    assert h3.shot_range(request(shots="auto", seconds=15)) == (2, 4)


def test_the_word_budget_matches_the_scale_of_the_guides_own_cases():
    """The four cases run roughly 75 to 95 words for six to ten seconds. A brief
    three times that length is not a richer specification, it is prose."""
    for seconds, observed in ((6, 95), (8, 90), (10, 75)):
        low, high = h3.word_budget(seconds)
        assert low <= observed <= high, f"{observed} words at {seconds}s is outside {low}-{high}"
    assert h3.word_budget(15)[1] > h3.word_budget(5)[1]


def test_the_duration_is_held_to_the_five_to_fifteen_seconds_h3_accepts():
    assert h3.clamp_seconds(2) == h3.MIN_SECONDS
    assert h3.clamp_seconds(40) == h3.MAX_SECONDS
    assert h3.clamp_seconds("nonsense") == 10.0


def test_speech_needs_a_voice_to_be_assigned_to():
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
                      "non_diegetic_music: N/A")
    assert fields == {h3.DESCRIPTION: "[Shot 1] A room.", h3.SOUNDSCAPE: "Rain.",
                      h3.MUSIC: "N/A"}


@pytest.mark.parametrize("label", [
    "**integrated_multimodal_description:**",
    "## Integrated Multimodal Description",
    "- integrated multimodal description -",
])
def test_a_label_survives_whatever_the_writer_decorated_it_with(label):
    assert h3.parse(f"{label}\n[Shot 1] A room.").get(h3.DESCRIPTION) == "[Shot 1] A room."


@pytest.mark.parametrize("written", [
    "For the target video, at 0.00 seconds into the target video, <Picture 1> "
    "(from [Shot 1]) is fully referenced.",
    "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
    "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) "
    "aligns with the 8.00-second mark of the target video.",
])
def test_an_instruction_line_the_writer_wrote_itself_is_dropped(written):
    """It is added afterwards with the real final-shot number in it, and two of
    them is worse than none."""
    fields = h3.parse(f"{written}\n\nintegrated_multimodal_description: [Shot 1] A room.")
    assert fields == {h3.DESCRIPTION: "[Shot 1] A room."}


def test_a_json_object_is_read_as_the_object_it_is():
    fields = h3.parse('{"integrated_multimodal_description": "[Shot 1] A room.", '
                      '"overall_soundscape": "Rain.", "non_diegetic_music": "N/A"}')
    assert fields[h3.DESCRIPTION] == "[Shot 1] A room."
    assert fields[h3.MUSIC] == "N/A"


def test_a_fence_and_a_reasoning_leak_are_stripped_before_anything_is_read():
    fields = h3.parse("<think>let me plan</think>\n```json\n"
                      '{"integrated_multimodal_description": "[Shot 1] A room."}\n```')
    assert fields == {h3.DESCRIPTION: "[Shot 1] A room."}


def test_prose_that_ignored_the_format_is_kept_as_the_description():
    assert h3.parse("[Shot 1] A room, and nobody labelled anything.") == {
        h3.DESCRIPTION: "[Shot 1] A room, and nobody labelled anything."}


def test_nothing_at_all_reads_as_nothing_at_all():
    assert h3.parse("") == {}
    assert h3.parse("   \n  ") == {}


# ── the checks ───────────────────────────────────────────────────────────────

def clean() -> tuple[str, dict, h3.H3Request]:
    asked = request(seconds=12.0, shots="3")
    fields = brief_fields(
        "[Shot 1] Live-action, cinematic, a courier at the foot of a stairwell.\n"
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
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any(str(h3.CHARACTER_LIMIT) in note for note in notes)


def test_the_sound_fields_are_held_to_their_sentence_counts():
    asked = request(music="score")
    assert any("1 to 4" in note for note in
               h3.checks("", brief_fields(soundscape="A. B. C. D. E.", music="Strings."), asked))
    assert any("1 to 3" in note for note in
               h3.checks("", brief_fields(music="A. B. C. D."), asked))


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


def test_a_line_carried_across_a_cut_has_to_be_marked_on_both_sides():
    asked = request(shots="2")
    fields = brief_fields("[Shot 1] She begins the sentence, <scenetrans>\n"
                          "[Shot 2] At 00:04.000, the camera cuts to the hall.")
    notes = h3.checks(h3.assemble(fields, asked), fields, asked)
    assert any("<scenetrans>" in note for note in notes)


def test_a_line_marked_on_both_sides_passes():
    asked = request(shots="2", seconds=10.0)
    fields = brief_fields(
        "[Shot 1] She begins, <scenetrans> and the line continues seamlessly across the cut.\n"
        "[Shot 2] At 00:04.000, the camera cuts to the hall, <scenetrans> carrying it over.")
    assert h3.checks(h3.assemble(fields, asked), fields, asked) == []


def test_well_formed_speech_passes():
    asked = request(seconds=10.0, shots="1", speakers=1, talk="sparse")
    fields = brief_fields("[Shot 1] The courier (S1) says: <d>[English] Almost there.</d>")
    assert h3.checks(h3.assemble(fields, asked), fields, asked) == []


# ── the instruction the model is given ───────────────────────────────────────

def test_the_output_contract_names_the_three_fields_in_order():
    system = h3.build_system(request())
    positions = [system.index(f"{label}:") for label in h3.FIELDS]
    assert positions == sorted(positions)


def test_an_anchored_brief_is_told_not_to_write_its_own_instruction_line():
    for mode in (h3.I2VA, h3.FL2VA, h3.L2VA):
        assert "reference or alignment line of your own" in h3.build_system(request(mode=mode))
    assert "reference or alignment line of your own" not in h3.build_system(request(mode=h3.T2VA))


def test_the_speech_syntax_appears_only_when_somebody_speaks():
    silent = h3.build_system(request(speakers=0))
    spoken = h3.build_system(request(speakers=2, talk="steady", language="Spanish"))
    assert "Nobody speaks" in silent
    assert "(S1,S2)" not in silent and "stable ID" not in silent
    assert "(S1,S2)" in spoken
    assert "<d>[Spanish]" in spoken


def test_the_speech_block_carries_the_voiceover_and_continuity_rules():
    spoken = h3.build_system(request(speakers=1, talk="sparse", shots="3"))
    assert "says in an off-screen voiceover" in spoken
    assert "lips remain" in spoken
    assert "<scenetrans>" in spoken and "<cutoff>" in spoken


def test_the_rule_for_a_line_crossing_a_cut_is_absent_where_there_is_no_cut():
    """A single-shot brief has nothing to cross, and a paragraph about crossing
    it is a paragraph the writer has to read past."""
    single = h3.build_system(request(speakers=1, talk="sparse", shots="1"))
    assert "<scenetrans>" not in single
    assert "<cutoff>" in single                       # the end of the video is still an end


def test_a_deliberate_transition_is_written_and_an_ordinary_cut_offers_the_alternates():
    dissolve = h3.build_system(request(shots="3", cut="cross_dissolve"))
    assert "the shot cross-dissolves to" in dissolve
    straight = h3.build_system(request(shots="3", cut="cut"))
    for phrase in h3.CUT_ALTERNATES:
        assert phrase in straight


def test_a_single_shot_brief_is_not_given_the_cutting_rules_at_all():
    system = h3.build_system(request(shots="1"))
    assert "[Shot 2] At 00:03.500" not in system


def test_the_interpolating_modes_are_told_there_are_no_cuts():
    system = h3.build_system(request(mode=h3.FL2VA))
    assert "no cuts and no timestamps" in system


def test_the_camera_table_is_the_guides_own_and_names_what_each_move_does():
    system = h3.build_system(request())
    for name, meaning in h3.CAMERA_MOVES:
        assert f"{name} — {meaning}" in system
    assert "with small amplitude" in system and "at slow speed" in system


def test_the_camera_is_asked_for_as_an_action_and_only_marked_when_unusual():
    plain = h3.build_system(request(camera="push_in"))
    assert "Camera: Push In." in plain
    marked = h3.build_system(request(camera="push_in", amplitude="large", speed="slow"))
    assert "Camera: Push In with large amplitude at slow speed." in marked


def test_the_style_is_asked_for_at_the_start_of_the_first_shot():
    assert "Style: claymation." in h3.build_system(request(style="claymation"))
    assert "Style: Live-action, cinematic." in h3.build_system(request(style="live_action_cinematic"))


def test_an_unnamed_style_comes_from_the_text_or_from_the_picture():
    """The guide splits this: keyframe tasks derive the style from the reference
    image; T2VA selects it from the user's text."""
    assert "Style:" not in h3.build_system(request(mode=h3.T2VA, style="auto"))
    assert "take it from the attached picture" in h3.build_system(request(mode=h3.I2VA,
                                                                          style="auto"))


def test_the_score_block_says_what_to_write_when_there_is_no_score():
    off = h3.build_system(request(music="off"))
    assert f"Write exactly: {h3.NOT_APPLICABLE}" in off
    scored = h3.build_system(request(music="score", music_brief="low strings"))
    assert "low strings" in scored
    assert "never name a song" in scored


def test_the_sound_fields_are_told_not_to_repeat_each_other():
    system = h3.build_system(request())
    assert "a sound described twice is asked for twice" in system
    assert "radio, television" in system or "radio, a television" in system


def test_the_cast_and_the_on_screen_text_appear_only_when_given():
    bare = h3.build_system(request())
    assert "THE CAST" not in bare and "ON-SCREEN TEXT" not in bare
    full = h3.build_system(request(cast="Mara = a courier in a soaked jacket",
                                   on_screen_text="FLOOR 12"))
    assert "Mara = a courier in a soaked jacket" in full
    assert "FLOOR 12" in full


def test_each_mode_gets_the_law_and_the_structure_the_guide_gives_it():
    i2va = h3.build_system(request(mode=h3.I2VA))
    assert "first-frame anchor → action onset → continuous development" in i2va
    fl2va = h3.build_system(request(mode=h3.FL2VA))
    assert "first-frame state → observable intermediate changes" in fl2va
    l2va = h3.build_system(request(mode=h3.L2VA))
    assert "plausible preceding state → explicit action and transition path" in l2va
    assert "does not belong to [Shot 1]" in l2va
    assert "THE FIRST FRAME" not in h3.build_system(request(mode=h3.T2VA))


def test_the_user_turn_carries_the_intent_and_says_what_the_pictures_are():
    assert "a courier runs up a wet stairwell" in h3.build_user(request())
    assert "first frame" in h3.build_user(request(mode=h3.I2VA))
    assert "first and the last frame" in h3.build_user(request(mode=h3.FL2VA))
    assert "last frame" in h3.build_user(request(mode=h3.L2VA))


# ── the frames on the wire ───────────────────────────────────────────────────

def test_a_text_only_request_sends_plain_text():
    turns = h3.messages(request(mode=h3.T2VA))
    assert turns[0]["role"] == "system"
    assert isinstance(turns[1]["content"], str)


def test_the_frames_go_first_and_in_picture_order():
    turns = h3.messages(request(mode=h3.FL2VA, first_frame="data:image/jpeg;base64,AAA",
                                last_frame="data:image/jpeg;base64,BBB"))
    parts = turns[1]["content"]
    assert [part["type"] for part in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].endswith("AAA")
    assert parts[1]["image_url"]["url"].endswith("BBB")


def test_l2va_sends_one_picture_and_it_is_the_last_frame():
    """<Picture 1> in L2VA is the *end* of the video, so the last-frame slot is
    what fills it and a first frame is neither wanted nor sent."""
    turns = h3.messages(request(mode=h3.L2VA, last_frame="data:image/jpeg;base64,BBB",
                                first_frame="data:image/jpeg;base64,AAA"))
    parts = turns[1]["content"]
    assert [part["type"] for part in parts] == ["image_url", "text"]
    assert parts[0]["image_url"]["url"].endswith("BBB")


def test_a_mode_that_needs_a_frame_refuses_to_go_without_one():
    with pytest.raises(h3.VisionUnavailable):
        h3.messages(request(mode=h3.I2VA))
    with pytest.raises(h3.VisionUnavailable):
        h3.messages(request(mode=h3.FL2VA, first_frame="data:image/jpeg;base64,AAA"))
    with pytest.raises(h3.VisionUnavailable):
        h3.messages(request(mode=h3.L2VA))


def test_a_frame_is_never_quietly_dropped_when_nothing_can_see_it():
    """The instruction line at the top of the brief states that the picture is a
    frame of the video. Sending it to a model that cannot see the picture makes
    that line a false statement about a video nobody chose."""
    with pytest.raises(h3.VisionUnavailable):
        h3.messages(request(mode=h3.I2VA, first_frame="data:image/jpeg;base64,AAA"),
                    vision_available=False)


# ── the option lists the UI is built from ────────────────────────────────────

def test_every_option_list_is_pairs_of_key_and_label():
    for options in (h3.MODES, h3.STYLES, h3.CAMERAS, h3.AMPLITUDES, h3.SPEEDS, h3.CUTS,
                    h3.SHOTS, h3.LANGUAGES, h3.TALK, h3.MUSIC_MODES, h3.SOUNDSCAPES):
        assert options
        assert all(isinstance(value, str) and isinstance(label, str) and value
                   for value, label in options)
        keys = [value for value, _ in options]
        assert len(keys) == len(set(keys))


def test_the_camera_list_is_h3s_own_vocabulary_and_nothing_invented():
    """Twenty moves: each side of the paired ones, plus the four that stand
    alone. Nothing added, nothing renamed."""
    labels = [label for value, label in h3.CAMERAS if value != "auto"]
    assert labels == list(h3.CAMERA_NAMES)
    assert len(h3.CAMERA_NAMES) == 20
    for name in ("Zoom In", "Zoom Out", "Pedestal Down", "Arc Shot", "POV",
                 "Roll Counterclockwise"):
        assert name in labels


def test_every_cut_has_a_phrase_and_every_style_that_is_not_auto_has_one_too():
    assert set(dict(h3.CUTS)) == set(h3.CUT_PHRASES)
    assert set(value for value, _ in h3.STYLES if value != "auto") == set(h3.STYLE_PHRASES)


def test_every_talk_setting_has_a_rate():
    assert set(value for value, _ in h3.TALK) == set(h3.TALK_RATE)


def test_the_amplitude_and_speed_defaults_are_the_ones_left_unwritten():
    assert "medium" not in h3.AMPLITUDE_PHRASES
    assert "normal" not in h3.SPEED_PHRASES
    assert h3.AMPLITUDE_PHRASES["small"] == "with small amplitude"
    assert h3.SPEED_PHRASES["fast"] == "at fast speed"
