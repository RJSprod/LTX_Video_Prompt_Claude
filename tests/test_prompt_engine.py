"""Adapter and option-source tests.

The engine's own behaviour is covered by ``test_upstream_parity.py`` (the
ported upstream self-test). What is tested here is the seam the port adds: that
the adapter hands upstream exactly what the upstream node would have handed it,
and that the UI's option lists are the engine's own constants rather than a
second copy that can drift.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from prompt_master.core.models import PromptRequest
from prompt_master.prompt_engine import options as opt
from prompt_master.prompt_engine.adapter import PromptEngine, VisionUnavailable
from prompt_master.prompt_engine.upstream import brain
from prompt_master.prompt_engine.upstream import negative as neg
from prompt_master.prompt_engine.upstream.accents import ACCENT_KEYS, STRENGTHS
from prompt_master.prompt_engine.upstream.cinematics import CAMERA_KEYS, TRANSITION_KEYS
from prompt_master.prompt_engine.upstream.music import MUSIC_KEYS
from prompt_master.prompt_engine.upstream.shotscript import FORMATS
from prompt_master.prompt_engine.upstream.styles import STYLE_KEYS


def data_url(color=(200, 40, 40), size=(64, 64)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── the adapter adds nothing and loses nothing ───────────────────────────────

def test_system_prompt_is_upstreams_byte_for_byte():
    request = PromptRequest(
        intent="a jamaican woman dances in a club and shouts over the music",
        video_mode="t2v", pov="off", accent="jamaican", accent_strength="thick",
        dialogue=45, wardrobe="her", undress=False, camera="orbit",
        transition="morph", music="auto", music_bg=False, lexicon="",
        fmt="shotscript", fps=24, seconds=20, style="neo_noir", seed=7,
    )
    built = PromptEngine().build(request)
    expected = brain.build_system(
        mode="t2v", pov="off", accent="jamaican", accent_strength="thick",
        dialogue=45, wardrobe="her", undress=False, seed=7,
        intent=request.intent, camera="orbit", transition="morph",
        music="auto", music_bg=False, lexicon="", fmt="shotscript",
        fps=24, seconds=20, style="neo_noir", style_hint="", has_image=False,
    )
    assert built.system == expected


def test_user_prompt_is_upstreams_byte_for_byte():
    request = PromptRequest(intent="she turns to the window", video_mode="t2v")
    built = PromptEngine().build(request)
    assert built.user == brain.build_user(
        intent="she turns to the window", mode="t2v", has_image=False, style_hint="")


def test_budgets_and_frames_come_from_upstream():
    request = PromptRequest(intent="x", video_mode="t2v", seconds=17.5, fps=30, dialogue=65)
    built = PromptEngine().build(request)
    wsec = brain.write_seconds(17.5)
    assert built.frames == brain.frame_count(30, 17.5)
    assert built.max_tokens == brain.max_tokens(17.5, 65, fmt="flowing")
    assert built.word_budget == brain.word_budget(wsec, 65)
    assert built.beat_budget == brain.beat_budget(wsec)
    assert built.dialogue_lines == brain.dialogue_lines(
        wsec, brain.talk_pct(65), beats=brain.beat_budget(wsec)[0])
    # LTX wants 8n+1 frames.
    assert (built.frames - 1) % 8 == 0


def test_base_negative_is_upstreams_byte_for_byte():
    request = PromptRequest(
        intent="she dances", video_mode="t2v", pov="female", dialogue=0,
        undress=True, camera="locked_off", transition="hard_cut",
        style="anime_cel", fmt="bracket", negative_extra="jpeg noise",
    )
    assert PromptEngine().base_negative(request) == brain.build_negative(
        pov="female", dialogue=0, undress=True, fmt="bracket",
        transition="hard_cut", intent="she dances", extra="jpeg noise",
        camera="locked_off", style="anime_cel", mode="t2v", auto="")


# ── image handling ───────────────────────────────────────────────────────────

def test_image_rides_in_one_multimodal_user_message_upstream_order():
    request = PromptRequest(intent="bring it to life", image_data_url=data_url(),
                            video_mode="i2v")
    messages = PromptEngine().build(request).messages
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    parts = messages[1]["content"]
    # upstream/routes.py puts the image part first, then the text part.
    assert [p["type"] for p in parts] == ["image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vision_policy_matches_upstream_max_side():
    """Upstream sends the still at max_side=768; a larger image must come back
    resized with its aspect ratio intact."""
    request = PromptRequest(intent="x", image_data_url=data_url(size=(2000, 1000)),
                            video_mode="i2v")
    url = PromptEngine().build(request).messages[1]["content"][0]["image_url"]["url"]
    raw = base64.b64decode(url.split(",", 1)[1])
    sent = Image.open(io.BytesIO(raw))
    assert max(sent.size) == 768
    assert abs(sent.size[0] / sent.size[1] - 2.0) < 0.02


def test_i2v_without_vision_raises_rather_than_degrading_silently():
    request = PromptRequest(intent="x", image_data_url=data_url(), video_mode="i2v")
    with pytest.raises(VisionUnavailable):
        PromptEngine().build(request, vision_available=False)


def test_text_only_retry_uses_the_blind_opener_and_says_so():
    request = PromptRequest(intent="x", image_data_url=data_url(), video_mode="i2v")
    retried = PromptEngine().retry_text_only(request)
    assert retried.send_vision is False
    assert "still NOT visible to you" in retried.system
    assert "Frame one is the attached image." not in retried.user


def test_t2v_never_sends_an_image_even_when_one_is_attached():
    request = PromptRequest(intent="x", image_data_url=data_url(), video_mode="t2v")
    built = PromptEngine().build(request)
    assert built.send_vision is False
    assert isinstance(built.messages[1]["content"], str)


def test_flat_art_triggers_upstream_cartoon_hint():
    """A single flat colour is the clearest possible cel image, so upstream's
    conservative detector must fire on it and harden the medium law."""
    built = PromptEngine().build(
        PromptRequest(intent="she waves", image_data_url=data_url(color=(20, 160, 220)),
                      video_mode="i2v"))
    assert built.style_hint == "cartoon"
    assert "MEDIUM (detector" in built.system
    assert "MEDIUM CHECK (locked)" in built.user


# ── smart negative ───────────────────────────────────────────────────────────

def test_smart_negative_uses_upstream_pass_and_guards():
    engine = PromptEngine()
    # Guard 1 is a literal space-padded substring test, so the protected phrase
    # has to sit mid-sentence with spaces on both sides — punctuation next to a
    # term defeats it. That is upstream's behaviour and is pinned, not fixed.
    script = "The camera holds on wet asphalt while rain falls through neon."
    assert engine.smart_negative_messages(script) == neg.auto_messages(script)
    raw = ("daylight, overcast, locked tripod frame, wet asphalt, x, "
           "a phrase that is far too long to be a visual term, daylight")
    cleaned = engine.clean_smart_negative(raw, script)
    terms = [t.strip() for t in cleaned.split(",")]
    assert "daylight" in terms and "locked tripod frame" in terms
    assert terms.count("daylight") == 1                  # deduplicated
    assert "wet asphalt" not in terms                    # in the script: dropped
    assert all(len(t.split()) <= 4 for t in terms)       # four words max
    assert "x" not in terms                              # too short


def test_smart_negative_guard_is_upstreams_literal_match():
    """Pins the punctuation sensitivity rather than papering over it: a term
    followed by a comma in the script is NOT dropped, because upstream tests
    for the term with a space on each side."""
    engine = PromptEngine()
    assert "wet asphalt" not in engine.clean_smart_negative(
        "wet asphalt", "rain on wet asphalt tonight")
    assert "wet asphalt" in engine.clean_smart_negative(
        "wet asphalt", "rain on wet asphalt, tonight")


def test_smart_negative_merges_through_upstream_dedupe():
    request = PromptRequest(intent="x", video_mode="t2v", negative_extra="blurry, my own term")
    merged = PromptEngine().merge_negative(request, auto="daylight, blurry")
    assert merged.count("blurry") == 1
    assert "daylight" in merged and "my own term" in merged


def test_smart_negative_failure_falls_back_to_base():
    def exploding(messages, **kw):
        raise RuntimeError("no server")

    assert PromptEngine().run_smart_negative("a shot", exploding) == ""


def test_clean_positive_is_upstream_clean_script():
    raw = "<think>planning</think>```\nShe turns.\n```"
    assert PromptEngine().clean_positive(raw) == brain.clean_script(raw)
    assert "<think>" not in PromptEngine().clean_positive(raw)


# ── the UI cannot drift from the engine ──────────────────────────────────────

@pytest.mark.parametrize("options,keys", [
    (opt.ACCENTS, ACCENT_KEYS),
    (opt.CAMERAS, CAMERA_KEYS),
    (opt.TRANSITIONS, TRANSITION_KEYS),
    (opt.STYLES, STYLE_KEYS),
    (opt.OUTPUT_FORMATS, FORMATS),
    (opt.ACCENT_STRENGTHS, list(STRENGTHS)),
])
def test_option_values_are_the_engine_keys_in_engine_order(options, keys):
    assert opt.values(options) == list(keys)


def test_music_options_are_auto_plus_every_genre():
    assert opt.values(opt.MUSIC) == ["auto"] + list(MUSIC_KEYS)


def test_upstream_option_counts_are_preserved():
    assert len(ACCENT_KEYS) - 1 == 47
    assert len(MUSIC_KEYS) - 1 == 35
    assert len(STYLE_KEYS) - 1 == 20
    assert len(CAMERA_KEYS) - 1 == 10
    assert len(TRANSITION_KEYS) - 1 == 10
    assert len(STRENGTHS) == 3
    assert len(FORMATS) == 3


def test_labels_and_values_stay_separate():
    """A label is display text; feeding one back as a key must not silently work."""
    assert opt.label_for(opt.ACCENTS, "rp_british") == "RP British"
    assert "RP British" not in ACCENT_KEYS
    assert opt.value_for(opt.ACCENTS, "RP British") == "rp_british"


def test_every_default_is_a_real_engine_key():
    request = PromptRequest(intent="x")
    for field, options in [("accent", opt.ACCENTS), ("accent_strength", opt.ACCENT_STRENGTHS),
                           ("music", opt.MUSIC), ("style", opt.STYLES),
                           ("camera", opt.CAMERAS), ("transition", opt.TRANSITIONS),
                           ("fmt", opt.OUTPUT_FORMATS), ("pov", opt.POV),
                           ("wardrobe", opt.WARDROBE), ("video_mode", opt.VIDEO_MODES)]:
        assert getattr(request, field) in opt.values(options), field


def test_every_control_value_builds_a_brief():
    """Sweep each control across its whole range — no key may crash the engine."""
    engine = PromptEngine()
    base = dict(intent="a woman speaks at a window", video_mode="t2v", dialogue=30)
    for field, values in [("accent", ACCENT_KEYS), ("style", STYLE_KEYS),
                          ("camera", CAMERA_KEYS), ("transition", TRANSITION_KEYS),
                          ("music", ["auto"] + list(MUSIC_KEYS)), ("fmt", FORMATS),
                          ("accent_strength", list(STRENGTHS)),
                          ("pov", ["off", "male", "female"]),
                          ("wardrobe", ["auto", "off", "her", "him"])]:
        for value in values:
            built = engine.build(PromptRequest(**{**base, field: value}))
            assert built.system.strip(), (field, value)
            assert built.base_negative.strip(), (field, value)
