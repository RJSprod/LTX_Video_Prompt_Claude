"""The window, checked for the things a finger needs.

Every assertion here is one of two questions: is this target big enough to hit,
and does this area scroll by dragging it? They run against a real window built
offscreen, because a style sheet that says ``min-height`` and a widget that is
actually that tall are different claims, and only the second one matters.

Skipped where Qt cannot open a display at all; the rest of the suite does not
need it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from prompt_master.core.paths import AppPaths

# A fingertip. Platform guidelines put it between 44 and 48 logical pixels; the
# smaller number is the floor these tests hold the window to.
FINGERTIP = 44


@pytest.fixture(scope="module")
def qt():
    import os

    # Before QtWidgets is imported: a test run must not open a window, and on a
    # machine with no display it must not fail for the want of one.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    application = widgets.QApplication.instance() or widgets.QApplication([])
    yield widgets
    application.processEvents()


@pytest.fixture
def window(qt, tmp_path):
    from prompt_master.ui.main_window import MainWindow

    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    made = MainWindow(paths)
    made.resize(1400, 900)
    yield made
    made.close()


# ── everything can be hit ────────────────────────────────────────────────────

def targets(qt, window):
    """Every widget a finger aims at.

    The editor inside a spin box is skipped: it is not aimed at, it fills the
    spin box that is, and it reports the height of its text rather than of the
    control drawn around it.
    """
    kinds = (qt.QPushButton, qt.QComboBox, qt.QCheckBox, qt.QSpinBox,
             qt.QDoubleSpinBox, qt.QLineEdit)
    return [widget for kind in kinds for widget in window.findChildren(kind)
            if not isinstance(widget.parentWidget(), qt.QAbstractSpinBox)]


def test_every_control_is_at_least_a_fingertip_tall(qt, window):
    """The check that matters: not what the style sheet asks for, but what the
    widgets in a built window actually report."""
    found = targets(qt, window)
    assert len(found) > 20, "the controls were not found, so nothing was checked"
    small = [(type(widget).__name__, widget.sizeHint().height())
             for widget in found if widget.sizeHint().height() < FINGERTIP]
    assert small == []


def test_each_page_has_the_biggest_button_on_the_action_it_is_for(qt, window):
    """The button pressed most is the one that should never be hunted for, and
    there is now one of those per page: Generate in either prompt mode, Send in
    conversation mode. Each has to win on its own page rather than in the
    window, because only one page is ever on screen."""
    for page, primary in ((window.pages.widget(0), window.generate_button),
                          (window.h3, window.h3.generate_button),
                          (window.chat, window.chat.send_button)):
        others = [b for b in page.findChildren(qt.QPushButton) if b is not primary]
        assert primary.objectName() == "primary"
        assert primary.sizeHint().height() > max(button.sizeHint().height() for button in others)
    assert window.generate_button.minimumWidth() >= 4 * FINGERTIP
    assert window.h3.generate_button.minimumWidth() >= 4 * FINGERTIP
    assert window.chat.send_button.minimumWidth() >= 3 * FINGERTIP


def test_the_action_bar_cannot_scroll_away(qt, window):
    """Status, Cancel and Generate live outside every scroll area, so no amount
    of scrolling can put them off the screen."""
    for widget in (window.generate_button, window.cancel_button, window.status):
        parents, node = [], widget.parentWidget()
        while node is not None:
            parents.append(node); node = node.parentWidget()
        assert not any(isinstance(parent, qt.QScrollArea) for parent in parents)


def test_spin_boxes_are_driven_by_their_own_steppers(qt, window):
    """Qt's stacked arrows are half a control tall each. These replace them, and
    have to actually move the value."""
    before = window.fps.value()
    steppers = [b for b in window.findChildren(qt.QPushButton) if b.objectName() == "stepper"]
    plus = next(b for b in steppers if b.text() == "+" and b.parentWidget() is window.fps.parentWidget())
    minus = next(b for b in steppers if b.text() == "−" and b.parentWidget() is window.fps.parentWidget())

    plus.click(); plus.click()
    assert window.fps.value() == before + 2
    minus.click()
    assert window.fps.value() == before + 1
    assert window.fps.buttonSymbols() == qt.QAbstractSpinBox.ButtonSymbols.NoButtons
    assert all(button.autoRepeat() for button in (plus, minus))   # hold to run it up


# ── everything scrolls by dragging it ────────────────────────────────────────

def test_the_areas_that_scroll_are_all_flickable(qt, window):
    """Including the text boxes: reading a long prompt on a touch screen means
    dragging the text, not finding a scroll bar."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScroller

    page = window.h3
    scrollable = [window.intent, window.lexicon, window.negative_extra,
                  window.positive, window.negative,
                  page.intent, page.on_screen, page.music_brief, page.ambience,
                  page.cast, page.notes, page.prompt, page.report]
    scrollable += window.findChildren(qt.QScrollArea)
    for widget in scrollable:
        viewport = widget.viewport()
        assert QScroller.hasScroller(viewport), f"{type(widget).__name__} does not flick"
        assert viewport.testAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)


def test_long_drop_downs_flick_and_have_finger_sized_rows(qt, window):
    """The accent list is 47 rows. A popup that only scrolls by its scroll bar
    is the worst target in the window."""
    from PySide6.QtWidgets import QScroller, QStyledItemDelegate

    assert window.accent.count() > 40
    for box in (window.accent, window.style, window.music):
        assert QScroller.hasScroller(box.view().viewport())
        # The stock popup delegate ignores the row height a style sheet asks for.
        assert type(box.itemDelegate()) is QStyledItemDelegate


# ── the display size is a setting ────────────────────────────────────────────

def test_display_size_scales_the_window_and_is_remembered(qt, window, tmp_path):
    from prompt_master.ui import touch

    before = window.generate_button.sizeHint().height()
    window.apply_scale("Larger")
    assert window.generate_button.sizeHint().height() > before
    assert touch.load_scale(AppPaths(tmp_path)) == "Larger"

    window.apply_scale("Comfortable")
    assert window.generate_button.sizeHint().height() == before


def test_an_unknown_saved_size_falls_back_instead_of_failing(tmp_path):
    from prompt_master.core.config import atomic_write_json
    from prompt_master.ui import touch

    paths = AppPaths(tmp_path)
    atomic_write_json(paths.data / touch.SETTINGS_FILE, {"display_size": "Enormous"})
    assert touch.load_scale(paths) == touch.DEFAULT_SCALE
    with pytest.raises(ValueError):
        touch.save_scale(paths, "Enormous")


def test_metrics_grow_with_the_scale_and_the_touch_sizes_stay_a_fingertip():
    """Every size keeps its proportions. The fingertip floor holds for all of
    them except the two compact ones, which are under it deliberately and only
    when they are asked for by name."""
    from prompt_master.ui import touch

    for name, scale in touch.SCALES.items():
        sizes = touch.metrics(scale)
        assert sizes["primary"] > sizes["action"] >= sizes["target"], name
        if name in touch.COMPACT:
            # Smaller than a fingertip, but still a control rather than a line
            # of text — and still tall enough to hit with a mouse.
            assert 28 <= sizes["target"] < FINGERTIP, name
        else:
            assert sizes["target"] >= FINGERTIP, name
    assert touch.metrics(1.35)["text"] > touch.metrics(1.0)["text"] > touch.metrics(0.72)["text"]


def test_the_compact_sizes_are_the_two_below_comfortable():
    from prompt_master.ui import touch

    assert list(touch.SCALES) == ["Smaller", "Small", "Comfortable", "Large", "Larger"]
    assert touch.COMPACT == ("Smaller", "Small")
    assert touch.DEFAULT_SCALE == "Comfortable" and touch.DEFAULT_SCALE not in touch.COMPACT
    assert touch.SCALES["Smaller"] < touch.SCALES["Small"] < touch.SCALES["Comfortable"]


def test_the_display_size_menu_offers_the_compact_sizes_and_they_shrink_it(qt, window, tmp_path):
    from prompt_master.ui import touch

    assert [action.data() for action in window.size_actions.actions()] == list(touch.SCALES)
    comfortable = window.generate_button.sizeHint().height()

    window.apply_scale("Small")
    small = window.generate_button.sizeHint().height()
    window.apply_scale("Smaller")
    smaller = window.generate_button.sizeHint().height()
    assert smaller < small < comfortable
    assert touch.load_scale(AppPaths(tmp_path)) == "Smaller"

    window.apply_scale("Comfortable")
    assert window.generate_button.sizeHint().height() == comfortable


# ── the rewiring survived the new layout ─────────────────────────────────────

def test_the_controls_still_carry_upstream_keys_into_the_request(qt, window):
    """The layout was rebuilt around the controls; what each one contributes to
    a PromptRequest must not have moved with it."""
    window.intent.setPlainText("A runner on a bridge")
    window.select(window.mode, "t2v")
    window.select(window.accent, window.accent.itemData(3))
    window.seconds.setValue(8.5); window.fps.setValue(30); window.seed.setValue(11)
    window.dialogue.setValue(45); window.undress.setChecked(True); window.smart.setChecked(True)
    window.select(window.dimensions, "1216x704")
    window.lexicon.setPlainText("Ada = a runner")
    window.negative_extra.setPlainText("logo")

    request = window.request()
    assert request.intent == "A runner on a bridge"
    assert request.video_mode == "t2v" and request.accent == window.accent.itemData(3)
    assert (request.seconds, request.fps, request.seed, request.dialogue) == (8.5, 30, 11, 45)
    assert request.undress and request.smart_negative
    assert (request.output_width, request.output_height) == (1216, 704)
    assert request.lexicon == "Ada = a runner" and request.negative_extra == "logo"


def test_a_seed_of_minus_one_draws_a_new_one_for_every_generation(qt, window):
    """The dial stays on -1 — the next run should be random too — while each
    request carries a real number, because upstream seeds its casting with it."""
    from prompt_master.core.models import RANDOM_SEED

    window.intent.setPlainText("A runner on a bridge")
    assert window.seed.minimum() == RANDOM_SEED
    assert window.seed.specialValueText()                 # shown in place of -1

    window.seed.setValue(RANDOM_SEED)
    drawn = {window.request().seed for _ in range(12)}
    assert window.seed.value() == RANDOM_SEED
    assert all(seed >= 0 for seed in drawn)
    assert len(drawn) > 8, "the same seed keeps coming back"

    window.seed.setValue(1234)
    assert {window.request().seed for _ in range(3)} == {1234}


def test_the_motion_preset_is_offered_and_carried(qt, window):
    from prompt_master.prompt_engine import motion

    window.intent.setPlainText("A runner on a bridge")
    assert [window.motion.itemData(i) for i in range(window.motion.count())] == list(motion.PRESETS)
    assert window.request().motion == motion.DEFAULT       # opens on upstream behaviour

    window.select(window.motion, "flow")
    assert window.request().motion == "flow"


def test_the_speech_slider_goes_where_it_is_tapped(qt, window):
    """Ten positions on a track, inside an area that flicks: a tap has to land
    on a value, because a drag there has to be told apart from a scroll."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from prompt_master.prompt_engine import speech

    slider = window.speech_slider
    slider.resize(400, 48)
    assert (slider.minimum(), slider.maximum(), slider.value()) == (speech.NONE, speech.MOST, speech.NONE)

    def tap(x):
        event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(x, 24), QPointF(x, 24),
                            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        slider.mousePressEvent(event)

    tap(400)
    assert slider.value() == speech.MOST          # the far right is ten times
    tap(0)
    assert slider.value() == speech.NONE          # and the far left is untouched
    tap(200)
    assert speech.NONE < slider.value() < speech.MOST


def test_the_slider_says_what_its_position_means(qt, window):
    from prompt_master.prompt_engine import speech

    window.speech_slider.setValue(speech.NONE)
    assert "exactly" in window.speech_note.text().casefold()

    window.dialogue.setValue(20)
    window.speech_slider.setValue(speech.MOST)
    assert "10×" in window.speech_note.text()
    assert "100%" in window.speech_note.text()          # the budget it will use

    window.dialogue.setValue(60)                        # moving the dial refreshes it
    assert "100%" in window.speech_note.text()
    window.speech_slider.setValue(2)
    assert "2×" in window.speech_note.text()


def test_the_slider_value_reaches_the_request(qt, window):
    window.intent.setPlainText('She says "Get back inside"')
    assert window.request().speech == 1                 # opens on "as written"
    window.speech_slider.setValue(7)
    assert window.request().speech == 7


def test_attaching_an_image_switches_the_mode_and_names_the_file(qt, window, tmp_path):
    from PIL import Image

    picture = tmp_path / "still.png"
    Image.new("RGB", (8, 8), "red").save(picture)
    window.image_path = picture
    window.image_label.setText(f"Image: {picture.name}")
    window.remove_button.setEnabled(True)
    window.select(window.mode, "i2v")
    assert window.request().video_mode == "i2v"
    assert "still.png" in window.image_label.text()

    window.remove_image()
    assert window.image_path is None and not window.remove_button.isEnabled()
    assert window.request().video_mode == "t2v"


# ── MiniMax-H3 mode ──────────────────────────────────────────────────────────
# The engine's own rules are held by test_minimax_h3.py. What is held here is
# the page: that its controls reach the request, that the rows on screen match
# the mode chosen, and that a brief arriving from the model is assembled and
# checked rather than shown raw.

@pytest.fixture
def h3_window(qt, tmp_path):
    from prompt_master.ui.main_window import H3_MODE, MainWindow

    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    made = MainWindow(paths)
    made.resize(1400, 900)
    made.select_mode(H3_MODE)
    yield made
    made.close()


def test_h3_mode_is_its_own_page_in_the_menu_and_is_remembered(qt, h3_window, tmp_path):
    from prompt_master.ui.main_window import H3_MODE, MainWindow

    assert h3_window.pages.currentWidget() is h3_window.h3
    checked = [action.data() for action in h3_window.mode_actions.actions() if action.isChecked()]
    assert checked == [H3_MODE]

    reopened = MainWindow(AppPaths(tmp_path))
    try:
        assert reopened.pages.currentWidget() is reopened.h3
    finally:
        reopened.close()


def test_the_frame_rows_are_the_ones_the_mode_actually_has(qt, h3_window):
    """An attached last frame means nothing in a mode with no last frame, so the
    row for it is not there to attach one into."""
    from prompt_master.prompt_engine import minimax_h3 as h3

    page = h3_window.h3
    for value, first, last in ((h3.T2VA, False, False), (h3.I2VA, True, False),
                               (h3.FL2VA, True, True)):
        page.mode.setCurrentIndex(page.mode.findData(value))
        assert page.first_row.isVisibleTo(page) is first
        assert page.last_row.isVisibleTo(page) is last


def test_the_h3_controls_carry_the_engines_own_keys_into_the_request(qt, h3_window):
    from prompt_master.prompt_engine import minimax_h3 as h3

    page = h3_window.h3
    page.intent.setPlainText("a courier runs up a wet stairwell")
    for box, value in ((page.mode, h3.T2VA), (page.shots, "3"), (page.style, "claymation"),
                       (page.camera, "push_in"), (page.amplitude, "large"),
                       (page.speed, "slow"), (page.cut, "fade"), (page.language, "Japanese"),
                       (page.talk, "steady"), (page.music, "score")):
        box.setCurrentIndex(box.findData(value))
    page.seconds.setValue(12.0)
    page.speakers.setValue(2)
    page.seed.setValue(4242)

    asked = page.request()
    assert (asked.mode, asked.shots, asked.style) == (h3.T2VA, "3", "claymation")
    assert (asked.camera, asked.amplitude, asked.speed) == ("push_in", "large", "slow")
    assert (asked.cut, asked.language, asked.talk, asked.music) == ("fade", "Japanese",
                                                                    "steady", "score")
    assert (asked.seconds, asked.speakers, asked.seed) == (12.0, 2, 4242)
    # The values are the engine's keys, so the engine recognises every one.
    assert h3.build_system(asked)


def test_the_h3_seed_of_minus_one_draws_a_new_one_for_every_generation(qt, h3_window):
    from prompt_master.core.models import RANDOM_SEED

    page = h3_window.h3
    page.seed.setValue(RANDOM_SEED)
    drawn = {page.request().seed for _ in range(12)}
    assert page.seed.value() == RANDOM_SEED
    assert all(seed >= 0 for seed in drawn) and len(drawn) > 8


def test_the_talk_caption_says_what_the_dials_add_up_to(qt, h3_window):
    """Three controls decide how much speech there is, and none of them says so
    on its own — a spin box reading 2 does not mention lines or IDs."""
    page = h3_window.h3
    assert "Nobody speaks" in page.talk_note.text()

    page.speakers.setValue(2)
    page.talk.setCurrentIndex(page.talk.findData("steady"))
    page.seconds.setValue(12.0)
    assert "spoken line" in page.talk_note.text()
    assert "(S1)…(S2)" in page.talk_note.text()


def test_the_score_box_is_only_live_when_there_is_a_score(qt, h3_window):
    page = h3_window.h3
    page.music.setCurrentIndex(page.music.findData("off"))
    assert not page.music_brief.isEnabled()
    page.music.setCurrentIndex(page.music.findData("score"))
    assert page.music_brief.isEnabled()


def test_h3_will_not_generate_without_an_intent_or_without_its_frames(qt, h3_window, monkeypatch):
    from prompt_master.prompt_engine import minimax_h3 as h3
    from prompt_master.ui import h3_page

    said = []
    monkeypatch.setattr(h3_page.QMessageBox, "warning",
                        lambda *args, **kwargs: said.append(args[1]))
    page = h3_window.h3
    page.generate()
    assert said == ["Missing intent"]

    page.intent.setPlainText("a courier runs up a wet stairwell")
    page.mode.setCurrentIndex(page.mode.findData(h3.FL2VA))
    page.generate()
    assert said[-1] == "First frame required"
    assert page.thread is None


def test_a_brief_is_streamed_then_replaced_by_the_one_that_was_assembled(qt, h3_window):
    """Watching it write and reading what it wrote are different needs: the raw
    fields arrive as they come, and the assembled brief replaces them."""
    from prompt_master.prompt_engine import minimax_h3 as h3

    page = h3_window.h3
    reply = ("integrated_multimodal_description: [Shot 1] Cinematic live-action, a courier "
             "at the foot of a wet stairwell.\n"
             "[Shot 2] At 00:04.000, the camera cuts to the landing above.\n\n"
             "overall_soundscape: Rain drums on the skylight. Wet boots slap concrete.\n\n"
             "non_diegetic_music: Low strings hold under the climb.")
    service = _ScriptedService([reply])
    page.service_provider = lambda: service
    page.intent.setPlainText("a courier runs up a wet stairwell")
    page.mode.setCurrentIndex(page.mode.findData(h3.I2VA))
    page.mode.setCurrentIndex(page.mode.findData(h3.T2VA))
    page.seconds.setValue(10.0)
    page.seed.setValue(77)

    page.generate()
    _finish(qt, page)

    written = page.prompt.toPlainText()
    assert written.startswith(f"{h3.DESCRIPTION}:")           # T2VA: no alignment line
    assert f"{h3.SOUNDSCAPE}: Rain drums" in written
    assert f"{h3.MUSIC}: Low strings" in written
    assert f"{h3.CHARACTER_LIMIT}" in page.count.text()
    assert "Nothing to flag" in page.report.toPlainText()

    sent = service.scripted.calls[0]
    assert sent["seed"] == 77
    assert sent["temperature"] == h3.TEMPERATURE
    assert "a courier runs up a wet stairwell" in sent["messages"][-1]["content"]
    assert service.vision_asked == [False]                    # no frames, no projector wanted


def test_a_brief_that_breaks_a_rule_is_shown_with_the_rule_it_broke(qt, h3_window):
    from prompt_master.prompt_engine import minimax_h3 as h3

    page = h3_window.h3
    service = _ScriptedService([
        "integrated_multimodal_description: [Shot 1] At 00:00.000, a stairwell.\n"
        "[Shot 2] At 00:40.000, the camera cuts to a landing.\n\n"
        "overall_soundscape: Rain.\n\nnon_diegetic_music: Strings."])
    page.service_provider = lambda: service
    page.intent.setPlainText("a courier runs up a wet stairwell")
    page.seconds.setValue(10.0)

    page.generate()
    _finish(qt, page)

    report = page.report.toPlainText()
    assert "first shot never does" in report
    assert "past the 10 s end" in report
    # The brief itself is left exactly as it came back: the checks report, they
    # never rewrite.
    assert "[Shot 2] At 00:40.000" in page.prompt.toPlainText()
    assert h3.DESCRIPTION in page.prompt.toPlainText()


def test_clearing_h3_takes_the_frames_and_the_checks_with_it(qt, h3_window, tmp_path):
    from PIL import Image
    from prompt_master.prompt_engine import minimax_h3 as h3

    page = h3_window.h3
    picture = tmp_path / "first.png"
    Image.new("RGB", (8, 8), "red").save(picture)
    page.mode.setCurrentIndex(page.mode.findData(h3.I2VA))
    page.first_path = picture
    page.first_label.setText(f"First frame: {picture.name}")
    page.first_remove.setEnabled(True)
    page.intent.setPlainText("a courier")
    page.report.setPlainText("• something")

    page.clear()
    assert page.first_path is None and not page.first_remove.isEnabled()
    assert page.intent.toPlainText() == "" and page.report.toPlainText() == ""
    assert page.count.text() == ""


# ── conversation mode ────────────────────────────────────────────────────────

@pytest.fixture
def chat_window(qt, tmp_path):
    """A window with one character saved, opened on conversation mode."""
    from prompt_master.chat.characters import Character, CharacterStore
    from prompt_master.ui.main_window import CONVERSATION_MODE, MainWindow

    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    CharacterStore.from_paths(paths).save(
        Character(name="Ada", context="A runner.", greeting="Hello {{user}}."))
    made = MainWindow(paths)
    made.resize(1400, 900)
    made.select_mode(CONVERSATION_MODE)
    yield made
    made.close()


def test_the_mode_lives_in_the_menu_bar_and_is_remembered(qt, chat_window, tmp_path):
    """Settings → Mode, not a control taking a row off the top of the window."""
    from prompt_master.ui.main_window import CONVERSATION_MODE, PROMPT_MODE, MainWindow

    from prompt_master.ui.main_window import H3_MODE

    assert not hasattr(chat_window, "mode_selector"), "the mode is a menu item now"
    chosen = {action.data(): action for action in chat_window.mode_actions.actions()}
    assert set(chosen) == {PROMPT_MODE, H3_MODE, CONVERSATION_MODE}
    assert chosen[CONVERSATION_MODE].isChecked() and not chosen[PROMPT_MODE].isChecked()

    chosen[PROMPT_MODE].trigger()
    assert chat_window.pages.currentWidget() is chat_window.pages.widget(0)
    chosen[CONVERSATION_MODE].trigger()
    assert chat_window.pages.currentWidget() is chat_window.chat

    reopened = MainWindow(AppPaths(tmp_path))
    try:
        assert reopened.pages.currentWidget() is reopened.chat
        checked = [a.data() for a in reopened.mode_actions.actions() if a.isChecked()]
        assert checked == [CONVERSATION_MODE]
    finally:
        reopened.close()


def test_the_view_menu_hides_each_chat_bar_on_its_own_and_remembers_it(qt, chat_window, tmp_path):
    from prompt_master.ui.main_window import MainWindow

    page = chat_window.chat
    keys = [key for key, _label in page.BARS]
    assert keys == ["character", "chat", "actions"]
    assert all(page.bars[key].isVisibleTo(page) for key in keys)
    assert all(chat_window.bar_actions[key].isChecked() for key in keys)

    chat_window.bar_actions["chat"].setChecked(False)
    assert not page.bars["chat"].isVisibleTo(page)
    assert page.bars["character"].isVisibleTo(page)      # one each, not one switch
    assert page.bars["actions"].isVisibleTo(page)

    reopened = MainWindow(AppPaths(tmp_path))
    try:
        assert not reopened.chat.bars["chat"].isVisibleTo(reopened.chat)
        assert not reopened.bar_actions["chat"].isChecked()
        assert reopened.bar_actions["character"].isChecked()
    finally:
        reopened.close()


def test_with_no_characters_the_chat_says_so_and_cannot_send(qt, window):
    assert window.chat.character is None
    assert not window.chat.send_button.isEnabled()
    assert "Characters" in window.chat.status.text()


def test_a_chat_opens_on_the_greeting_with_your_name_in_it(qt, chat_window):
    from prompt_master.chat.characters import Persona

    page = chat_window.chat
    assert page.character.name == "Ada"
    assert [message.text for message in page.conversation.messages] == ["Hello You."]
    assert page.speaker_name("user") == "You"

    page.persona = Persona(name="Rashan", description="A director.")
    page.new_chat()
    assert page.conversation.messages[0].text == "Hello Rashan."
    assert page.speaker_name("user") == "Rashan"


def test_every_button_on_a_message_is_a_fingertip(qt, chat_window):
    """The per-message menu and the version pager are targets like any other."""
    page = chat_window.chat
    page.conversation.append("assistant", "one")
    page.conversation.messages[-1].add_version("two")
    page.render()

    buttons = [button for bubble in page.bubbles
               for button in bubble.findChildren(qt.QPushButton)]
    assert len(buttons) >= 4, "the message buttons were not found"
    assert [b.text() for b in buttons if b.sizeHint().height() < FINGERTIP] == []


def test_branching_leaves_the_chat_it_came_from_alone(qt, chat_window):
    page = chat_window.chat
    original = page.conversation.identifier
    page.conversation.append("user", "one")
    page.conversation.append("assistant", "two")
    page.save()

    page.branch(1)
    assert page.conversation.identifier != original
    assert [message.text for message in page.conversation.messages] == ["Hello You.", "one"]
    assert page.chat_box.count() == 2
    assert len(page.chats.load("Ada", original).messages) == 3


def test_editing_and_deleting_reach_the_saved_chat(qt, chat_window):
    page = chat_window.chat
    page.conversation.append("user", "one")
    page.conversation.append("assistant", "two")
    page.save()

    page.edit_message(1, "one, edited")
    page.delete_from(2)
    reloaded = page.chats.load("Ada", page.conversation.identifier)
    assert [message.text for message in reloaded.messages] == ["Hello You.", "one, edited"]


def test_paging_back_to_an_earlier_reply_is_saved(qt, chat_window):
    page = chat_window.chat
    page.conversation.append("user", "one")
    reply = page.conversation.append("assistant", "first try")
    reply.add_version("second try")
    page.save()

    page.show_version(2, 0)
    assert page.chats.load("Ada", page.conversation.identifier).messages[2].text == "first try"


def test_remove_last_puts_your_message_back_in_the_box(qt, chat_window):
    page = chat_window.chat
    page.conversation.append("user", "one")
    page.conversation.append("assistant", "two")
    page.render()

    page.remove_last()
    assert page.input.toPlainText() == "one"
    assert [message.text for message in page.conversation.messages] == ["Hello You."]


def test_the_settings_panel_is_saved_with_the_character(qt, chat_window):
    page = chat_window.chat
    page.settings_button.setChecked(True)
    assert page.settings_pane.isVisibleTo(page)

    page.temperature.setValue(0.42)
    page.max_tokens.setValue(128)
    page.system.setPlainText("Answer as {{char}}, tersely.")
    page.persist_settings()

    saved = page.characters.load("Ada")
    assert (saved.temperature, saved.max_reply_tokens) == (0.42, 128)
    assert saved.system == "Answer as {{char}}, tersely."
    # And the character's own fields were not lost on the way through.
    assert saved.context == "A runner." and saved.greeting == "Hello {{user}}."


def test_touching_a_setting_writes_it_without_anything_being_pressed(qt, chat_window):
    """Changing a value is what saves it. The panel schedules the write a
    moment after the control stops moving, so holding a stepper down is one
    write rather than forty."""
    from prompt_master.ui import chat_page

    page = chat_window.chat
    page.settings_button.setChecked(True)

    page.temperature.setValue(0.42)
    assert page._settings_timer.isActive(), "no write was scheduled"
    assert page.save_settings_button.isEnabled(), "the panel does not say it has anything to write"
    page._settings_timer.timeout.emit()          # what a moment of stillness does

    assert page.characters.load("Ada").temperature == 0.42
    assert not page.save_settings_button.isEnabled()
    assert page.settings_note.text() == chat_page.SETTINGS_SAVED


def test_the_panels_save_button_writes_now_and_names_what_it_wrote_to(qt, chat_window):
    """The button is not the only way to save; it is the way to be told that
    it saved, which is the half that was missing."""
    page = chat_window.chat
    page.settings_button.setChecked(True)
    page.top_p.setValue(0.5)
    page.max_tokens.setValue(256)
    page.system.setPlainText("Answer as {{char}}, tersely.")
    page._settings_timer.stop()                  # nothing has written it yet

    page.save_settings_button.click()

    saved = page.characters.load("Ada")
    assert (saved.top_p, saved.max_reply_tokens) == (0.5, 256)
    assert saved.system == "Answer as {{char}}, tersely."
    assert "Ada" in page.status.text() and "saved" in page.status.text()
    assert not page.save_settings_button.isEnabled()


def test_a_settings_write_that_fails_says_so_instead_of_going_quiet(qt, chat_window, monkeypatch):
    """A write that silently failed is exactly what "the temperature will not
    stick" looks like from the outside."""
    page = chat_window.chat
    page.settings_button.setChecked(True)

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(page.characters, "save", refuse)
    page.temperature.setValue(0.31)
    page.save_settings_button.click()

    assert "disk full" in page.settings_note.text()
    assert "disk full" in page.status.text()
    assert page.save_settings_button.isEnabled(), "there is still something to write"


def test_opening_another_character_does_not_look_like_an_unsaved_change(qt, chat_window):
    """Filling the panel in is not a change to it: the previous character's
    pending write is dropped, and the new one opens saved."""
    from prompt_master.chat.characters import Character

    page = chat_window.chat
    page.characters.save(Character(name="Chiharu", context="An engineer.", temperature=1.1))
    page.reload_characters(select="Chiharu")

    assert page.temperature.value() == 1.1
    assert not page._settings_timer.isActive()
    assert not page.save_settings_button.isEnabled()


class _ScriptedClient:
    """A llama-server that answers from a list instead of from a model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "seed": seed,
                           "temperature": temperature, "top_p": top_p})
        reply = self.replies.pop(0) if self.replies else "…"
        for piece in (reply[:3], reply[3:]):
            on_text(piece)
        return reply


class _ScriptedService:
    def __init__(self, replies):
        self.scripted = _ScriptedClient(replies)
        self.vision_asked = []

    def client(self, needs_vision=False):
        self.vision_asked.append(needs_vision)
        return self.scripted


def _finish(qt, page, timeout=15.0):
    """Run the generation the page just started to completion.

    The event loop is pumped rather than waited on: everything the worker sends
    back — the chunks, the finished reply, the thread's own quit — arrives as a
    queued signal delivered on the main thread, so blocking that thread on
    ``QThread.wait`` would stop the very messages being waited for.
    """
    import time

    assert page.thread is not None, "no generation was started"
    application = qt.QApplication.instance()
    deadline = time.monotonic() + timeout
    while page.thread is not None and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    assert page.thread is None, "the generation did not finish"
    for _ in range(5):
        application.processEvents()


def test_sending_a_message_streams_the_reply_into_the_chat_and_saves_it(qt, chat_window):
    page = chat_window.chat
    service = _ScriptedService(["Hello there."])
    page.service_provider = lambda: service
    page.temperature.setValue(0.5)
    page.seed.setValue(99)
    page.max_tokens.setValue(128)

    page.input.setPlainText("hello")
    page.send()
    _finish(qt, page)

    assert [message.text for message in page.conversation.messages] == [
        "Hello You.", "hello", "Hello there."]
    saved = page.chats.load("Ada", page.conversation.identifier)
    assert [message.text for message in saved.messages] == ["Hello You.", "hello", "Hello there."]
    assert page.input.toPlainText() == ""

    sent = service.scripted.calls[0]
    assert sent["messages"][0]["role"] == "system" and "You are Ada" in sent["messages"][0]["content"]
    assert sent["messages"][-1]["content"] == "hello"
    # The reply being written is not in the history it was written from.
    assert len(sent["messages"]) == 3
    assert (sent["seed"], sent["temperature"], sent["max_tokens"]) == (99, 0.5, 128)
    assert service.vision_asked == [False]


def test_regenerating_keeps_the_reply_it_replaced_and_pages_between_them(qt, chat_window):
    page = chat_window.chat
    service = _ScriptedService(["first try", "second try"])
    page.service_provider = lambda: service
    page.input.setPlainText("hello")
    page.send()
    _finish(qt, page)

    page.regenerate(page.conversation.last_index("assistant"))
    _finish(qt, page)

    reply = page.conversation.messages[-1]
    assert reply.versions == ["first try", "second try"] and reply.text == "second try"
    page.show_version(len(page.conversation.messages) - 1, 0)
    assert page.chats.load("Ada", page.conversation.identifier).messages[-1].text == "first try"


def test_impersonate_writes_into_the_box_rather_than_the_chat(qt, chat_window):
    page = chat_window.chat
    service = _ScriptedService(["You: I am ready to go."])
    page.service_provider = lambda: service
    before = len(page.conversation.messages)

    page.impersonate()
    _finish(qt, page)

    assert len(page.conversation.messages) == before          # nothing was said
    assert page.input.toPlainText() == "I am ready to go."    # the label was stripped
    assert "as You" in service.scripted.calls[0]["messages"][-1]["content"]


def test_an_attached_picture_is_sent_with_the_message_and_asks_for_vision(qt, chat_window, tmp_path):
    from PIL import Image

    page = chat_window.chat
    service = _ScriptedService(["A red square."])
    page.service_provider = lambda: service
    picture = tmp_path / "still.png"
    Image.new("RGB", (32, 32), "red").save(picture)

    page.attachment = picture
    page.input.setPlainText("what is this")
    page.send()
    _finish(qt, page)

    content = service.scripted.calls[0]["messages"][-1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"] == "what is this"
    assert service.vision_asked == [True]
    assert page.conversation.messages[1].image_name == "still.png"
    assert page.attachment is None                            # cleared after sending


# ── the start of the reply ───────────────────────────────────────────────────

def _holding_layout(root, widget):
    """The layout that directly holds ``widget``, anywhere under ``root``.

    ``QLayout.indexOf`` does not recurse, and the two buttons under test sit in
    a column nested two deep in the page's own.
    """
    pending = [root]
    while pending:
        layout = pending.pop()
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget() is widget:
                return layout
            if item.layout() is not None:
                pending.append(item.layout())
    return None


def test_the_response_button_sits_under_attach_and_folds_a_box_out(qt, chat_window):
    from prompt_master.ui import chat_page as module

    page = chat_window.chat
    assert page.prefix_button.text() == module.PREFIX_BUTTON
    assert not page.prefix_panel.isVisibleTo(page)

    # Directly under Attach…, in the column beside the input box.
    stack = _holding_layout(page.layout(), page.attach_button)
    assert stack is not None, "the Attach… button was not found in the layout"
    assert stack.indexOf(page.prefix_button) == stack.indexOf(page.attach_button) + 1

    page.prefix_button.setChecked(True)
    assert page.prefix_panel.isVisibleTo(page)
    page.prefix_button.setChecked(False)
    assert not page.prefix_panel.isVisibleTo(page)


def test_a_reply_begins_with_the_start_and_the_model_carries_on(qt, chat_window):
    """The start is not asked for, it is written in — so the reply begins with
    it whatever the model does, and what comes back is joined onto it."""
    page = chat_window.chat
    service = _ScriptedService(["and it always will be."])
    page.service_provider = lambda: service
    page.prefix_button.setChecked(True)
    page.prefix_edit.setPlainText("It's always been red")

    page.input.setPlainText("What colour is the sky")
    page.send()
    _finish(qt, page)

    assert page.conversation.messages[-1].text == "It's always been red and it always will be."
    saved = page.chats.load("Ada", page.conversation.identifier)
    assert saved.messages[-1].text == "It's always been red and it always will be."

    sent = service.scripted.calls[0]["messages"]
    assert sent[-2] == {"role": "assistant", "content": "It's always been red"}
    assert sent[-1]["role"] == "user" and "already been started" in sent[-1]["content"]


def test_the_start_survives_sending_regenerating_and_reopening_the_chat(qt, chat_window):
    """"Never auto-cleared" is the whole of the feature: it is used until it is
    taken away, not once."""
    from prompt_master.ui import chat_page as module

    page = chat_window.chat
    service = _ScriptedService(["one.", "two.", "three."])
    page.service_provider = lambda: service
    page.prefix_button.setChecked(True)
    page.prefix_edit.setPlainText("It's always been red")
    identifier = page.conversation.identifier

    page.input.setPlainText("What colour is the sky")
    page.send()
    _finish(qt, page)
    assert page.conversation.response_prefix == "It's always been red"

    page.regenerate(page._last("assistant"))            # and again on a regenerate
    _finish(qt, page)
    assert page.conversation.messages[-1].versions == ["It's always been red one.",
                                                       "It's always been red two."]

    page.input.setPlainText("and the sea")              # and on the message after that
    page.send()
    _finish(qt, page)
    assert page.conversation.messages[-1].text == "It's always been red three."

    # And on reopening the chat, from the file rather than from the box.
    page.new_chat()
    assert page.prefix_edit.toPlainText() == "", "a new chat starts with none"
    assert page.prefix_button.text() == module.PREFIX_BUTTON
    page.open_chat(identifier)
    assert page.prefix_edit.toPlainText() == "It's always been red"
    assert page.prefix_button.text() == module.PREFIX_BUTTON_SET


def test_the_start_is_the_chats_own_and_a_branch_takes_it_along(qt, chat_window):
    page = chat_window.chat
    page.prefix_button.setChecked(True)
    page.prefix_edit.setPlainText("It's always been red")
    first = page.conversation.identifier
    page.conversation.append("user", "one")
    page.save()

    page.branch(1)
    assert page.conversation.identifier != first
    assert page.prefix_edit.toPlainText() == "It's always been red"

    page.prefix_edit.setPlainText("It's always been blue")
    page.persist_prefix()
    assert page.chats.load("Ada", first).response_prefix == "It's always been red"


def test_clearing_is_the_one_thing_that_takes_the_start_away(qt, chat_window):
    from prompt_master.ui import chat_page as module

    page = chat_window.chat
    service = _ScriptedService(["The model's own opening."])
    page.service_provider = lambda: service
    page.prefix_button.setChecked(True)
    page.prefix_edit.setPlainText("It's always been red")
    assert page.prefix_button.text() == module.PREFIX_BUTTON_SET
    assert page.prefix_clear_button.isEnabled()

    page.prefix_clear_button.click()

    assert page.prefix_edit.toPlainText() == ""
    assert page.conversation.response_prefix == ""
    assert page.prefix_button.text() == module.PREFIX_BUTTON
    assert not page.prefix_clear_button.isEnabled()
    assert page.chats.load("Ada", page.conversation.identifier).response_prefix == ""

    page.input.setPlainText("What colour is the sky")
    page.send()
    _finish(qt, page)
    assert page.conversation.messages[-1].text == "The model's own opening."
    # Nothing is asked of the model beyond the conversation itself again.
    assert service.scripted.calls[0]["messages"][-1]["content"] == "What colour is the sky"


def test_the_start_has_its_placeholders_filled_in(qt, chat_window):
    from prompt_master.chat.characters import Persona

    page = chat_window.chat
    service = _ScriptedService([" — and I always have."])
    page.service_provider = lambda: service
    page.persona = Persona(name="Rashan", description="A director.")
    page.prefix_button.setChecked(True)
    page.prefix_edit.setPlainText("Listen, {{user}} — {{char}} is telling you")

    page.input.setPlainText("hello")
    page.send()
    _finish(qt, page)

    assert page.conversation.messages[-1].text.startswith(
        "Listen, Rashan — Ada is telling you")
    # The box keeps what was typed; only the wire sees it filled in.
    assert page.prefix_edit.toPlainText() == "Listen, {{user}} — {{char}} is telling you"


def test_a_failed_generation_does_not_leave_the_start_sitting_there(qt, chat_window):
    """A bubble holding nothing but the start it was handed is as empty as one
    holding nothing at all, and is no more use to look at."""
    class _Broken:
        def client(self, needs_vision=False):
            raise RuntimeError("llama-server is not running")

    page = chat_window.chat
    page.prefix_button.setChecked(True)
    page.prefix_edit.setPlainText("It's always been red")
    page.service_provider = _Broken

    page.input.setPlainText("What colour is the sky")
    page.send()
    _finish(qt, page)

    assert [message.text for message in page.conversation.messages] == [
        "Hello You.", "What colour is the sky"]
    assert "llama-server is not running" in page.status.text()
    assert page.conversation.response_prefix == "It's always been red"   # still in force


def test_a_failed_regenerate_leaves_the_reply_it_was_replacing(qt, chat_window):
    """The version a regenerate starts holds only the start, so dropping it has
    to put the reply that was showing back rather than delete the message."""
    page = chat_window.chat
    service = _ScriptedService(["first try"])
    page.service_provider = lambda: service
    page.input.setPlainText("hello")
    page.send()
    _finish(qt, page)

    page.prefix_edit.setPlainText("It's always been red")

    class _Broken:
        def client(self, needs_vision=False):
            raise RuntimeError("llama-server is not running")

    page.service_provider = _Broken
    page.regenerate(page._last("assistant"))
    _finish(qt, page)

    reply = page.conversation.messages[-1]
    assert reply.versions == ["first try"] and reply.text == "first try"


def test_a_generation_that_fails_does_not_leave_an_empty_reply_behind(qt, chat_window):
    class _Broken:
        def client(self, needs_vision=False):
            raise RuntimeError("llama-server is not running")

    page = chat_window.chat
    page.service_provider = _Broken
    page.input.setPlainText("hello")
    page.send()
    _finish(qt, page)

    assert [message.text for message in page.conversation.messages] == ["Hello You.", "hello"]
    assert "llama-server is not running" in page.status.text()
    assert page.send_button.isEnabled()


# ── the dialogs, which are built only when they are opened ───────────────────

@pytest.fixture
def character_dialog(qt, chat_window):
    from prompt_master.ui.character_editor import CharacterDialog

    page = chat_window.chat
    dialog = CharacterDialog(page.characters, page.metrics, chat_window)
    yield dialog
    dialog.close()


def test_the_character_dialog_writes_a_character_the_chat_can_open(qt, chat_window, character_dialog):
    character_dialog.new_character()
    character_dialog.name.setText("Chiharu")
    character_dialog.context.setPlainText("An engineer.")
    character_dialog.greeting.setPlainText("Hey!")
    assert character_dialog.save_character()

    saved = chat_window.chat.characters.load("Chiharu")
    assert (saved.context, saved.greeting) == ("An engineer.", "Hey!")
    assert character_dialog.chosen == "Chiharu"
    assert sorted(chat_window.chat.characters.names()) == ["Ada", "Chiharu"]


def test_the_character_dialog_will_not_write_a_nameless_or_duplicate_character(
        qt, character_dialog, monkeypatch):
    # The refusal is a modal box, which an offscreen run would wait on forever.
    said = []
    monkeypatch.setattr(qt.QMessageBox, "warning",
                        lambda *args, **kwargs: said.append(args[2]))

    character_dialog.new_character()
    character_dialog.name.setText("   ")
    assert not character_dialog.save_character()

    character_dialog.name.setText("Ada")            # already in the store
    assert not character_dialog.save_character()
    assert len(said) == 2 and "already exists" in said[1]


def test_editing_a_character_leaves_how_it_replies_alone(qt, chat_window, character_dialog):
    page = chat_window.chat
    page.temperature.setValue(0.33)
    page.persist_settings()

    character_dialog.refresh("Ada")
    character_dialog.context.setPlainText("A runner, and a swimmer.")
    assert character_dialog.save_character()

    saved = page.characters.load("Ada")
    assert saved.context == "A runner, and a swimmer."
    assert saved.temperature == 0.33


def test_the_dialogs_are_finger_sized_too(qt, character_dialog):
    small = [(type(widget).__name__, widget.sizeHint().height())
             for widget in targets(qt, character_dialog)
             if widget.sizeHint().height() < FINGERTIP]
    assert small == []


def test_the_persona_dialog_returns_what_was_typed(qt, chat_window):
    from prompt_master.chat.characters import Persona
    from prompt_master.ui.character_editor import PersonaDialog

    dialog = PersonaDialog(Persona(), chat_window)
    try:
        assert not dialog.persona().defined              # opens empty
        dialog.name.setText("Rashan")
        dialog.description.setPlainText("A director.")
        typed = dialog.persona()
        assert typed.defined and typed.display == "Rashan" and typed.description == "A director."
    finally:
        dialog.close()


# ── the transcript reads as a conversation ───────────────────────────────────

def _tap(widget, travel=0):
    """A press and release on ``widget``, ``travel`` pixels apart."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    for kind, offset in ((QMouseEvent.Type.MouseButtonPress, 0),
                         (QMouseEvent.Type.MouseButtonRelease, travel)):
        local = QPointF(4 + offset, 4)
        event = QMouseEvent(kind, local, QPointF(100 + offset, 100),
                            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, event)


def test_the_bubbles_sit_on_their_own_side_and_carry_no_name(qt, chat_window):
    """Which side a bubble is on is what says who said it."""
    page = chat_window.chat
    page.conversation.append("user", "one")
    page.render()
    theirs, mine = page.bubbles[0], page.bubbles[1]

    # Mine is pushed right by a stretch; theirs opens with the avatar holder.
    assert mine.layout().itemAt(0).spacerItem() is not None
    assert theirs.layout().itemAt(0).widget() is not None
    assert mine.mine and not theirs.mine

    labels = [label.text() for bubble in page.bubbles
              for label in bubble.findChildren(qt.QLabel)]
    assert "Ada" not in labels and "You" not in labels
    assert "one" in labels                                   # the words are there

    # And a bubble stops well short of the full width.
    assert 0 < mine.frame.maximumWidth() <= max(1, page.bubble_width())


def test_the_picture_appears_once_at_the_start_of_a_run(qt, chat_window, tmp_path):
    from PIL import Image

    page = chat_window.chat
    picture = tmp_path / "face.png"
    Image.new("RGB", (32, 32), "red").save(picture)
    page.characters.set_avatar("Ada", picture)
    page.open_character("Ada")

    page.conversation.append("assistant", "and another thing")
    page.render()
    faces = [bubble.layout().itemAt(0).widget() for bubble in page.bubbles]
    assert not faces[0].pixmap().isNull()                     # opens the run
    assert faces[1].pixmap().isNull()                         # follows it
    assert faces[0].size() == faces[1].size()                 # still in one column


def test_a_message_shows_its_menu_only_when_it_is_tapped(qt, chat_window):
    page = chat_window.chat
    page.conversation.append("user", "one")
    page.conversation.append("assistant", "two")
    page.render()
    assert all(not bubble.actions_button.isVisibleTo(bubble) for bubble in page.bubbles)

    _tap(page.bubbles[1].body)
    assert page.bubbles[1].actions_button.isVisibleTo(page.bubbles[1])
    assert not page.bubbles[2].actions_button.isVisibleTo(page.bubbles[2])
    assert page.revealed == 1

    _tap(page.bubbles[2].frame)                       # one at a time
    assert not page.bubbles[1].actions_button.isVisibleTo(page.bubbles[1])
    assert page.bubbles[2].actions_button.isVisibleTo(page.bubbles[2])

    _tap(page.bubbles[2].frame)                       # tapping again puts it away
    assert not page.bubbles[2].actions_button.isVisibleTo(page.bubbles[2])


def test_a_drag_through_the_transcript_opens_nothing(qt, chat_window):
    """A flick is how the transcript scrolls, and must not be read as a tap."""
    page = chat_window.chat
    page.conversation.append("user", "one")
    page.render()

    _tap(page.bubbles[1].body, travel=200)
    assert page.revealed == -1
    assert all(not bubble.actions_button.isVisibleTo(bubble) for bubble in page.bubbles)


def test_the_version_pager_shows_without_a_tap(qt, chat_window):
    """The ⋯ hides, but "2/3" is the only thing that says an older reply is
    still there."""
    page = chat_window.chat
    page.conversation.append("assistant", "first try")
    page.conversation.messages[-1].add_version("second try")
    page.render()

    regenerated, plain = page.bubbles[1], page.bubbles[0]
    assert regenerated.pager is not None and regenerated.actions_row.isVisibleTo(regenerated)
    assert not regenerated.actions_button.isVisibleTo(regenerated)
    assert plain.pager is None and not plain.actions_row.isVisibleTo(plain)


class _WindowWatcher:
    """Every top-level window that appears while it is installed.

    An application-wide event filter rather than a patched ``setVisible``,
    because the shows that matter here do not come from Python at all: Qt
    raises them from C++ on its way through the event loop, and a patched
    Python method never sees them.
    """

    def __init__(self, qt):
        from PySide6.QtCore import QEvent, QObject
        from PySide6.QtWidgets import QApplication, QWidget

        self.shown: list[str] = []
        watcher = self

        class Filter(QObject):
            def eventFilter(self, watched, event):
                if (event.type() == QEvent.Type.Show
                        and isinstance(watched, QWidget) and watched.isWindow()):
                    watcher.shown.append(type(watched).__name__)
                return False

        # Held on the instance: an event filter nothing refers to is collected,
        # and a collected filter silently watches nothing.
        self._filter = Filter()
        self._app = QApplication.instance()

    def __enter__(self):
        self._app.installEventFilter(self._filter)
        return self

    def __exit__(self, *_exc):
        self._app.processEvents()      # the show is deferred; give it its turn
        self._app.removeEventFilter(self._filter)
        return False


def test_sending_a_message_puts_no_window_on_the_screen(qt, chat_window):
    """Sending a message rebuilt the transcript, and the bubbles it replaced
    flashed up as blank windows of their own.

    Two things did it, and both come down to one Qt rule: a widget with no
    parent *is* a top-level window. The row under a bubble was made visible
    before the layout had claimed it; and the outgoing bubbles were unparented
    while still counting as visible, which Qt honours by showing each of them
    on its next pass through the event loop.

    Neither is visible to a test that watches the conversation, and neither
    comes from a Python call that could be patched, so what is watched here is
    the screen itself — any window appearing during a send is the bug.

    The send has to be a real one, and the window has to be on it: Qt marks a
    reparented widget hidden only when it was never created in the first place,
    so a synthetic ``render()`` on a window nobody showed cannot demonstrate
    this at all.
    """
    page = chat_window.chat
    service = _ScriptedService(["Hello there.", "A second try.", "And again."])
    page.service_provider = lambda: service
    application = qt.QApplication.instance()
    chat_window.show()
    for _ in range(5):
        application.processEvents()

    # A first exchange, and a regenerate to leave a pager behind, so the send
    # under test replaces a transcript that is on screen and carries both cases.
    page.input.setPlainText("hello")
    page.send()
    _finish(qt, page)
    page.regenerate(page._last("assistant"))
    _finish(qt, page)
    assert any(bubble.pager is not None for bubble in page.bubbles), "no pager to cover"

    with _WindowWatcher(qt) as watcher:
        page.input.setPlainText("and again")
        page.send()
        _finish(qt, page)

    assert watcher.shown == []
    chat_window.hide()


# ── it follows the newest message until you scroll away ──────────────────────

def test_the_transcript_follows_new_content_only_while_it_is_at_the_end(qt, chat_window):
    page = chat_window.chat
    bar = page.transcript_scroll.verticalScrollBar()

    bar.setRange(0, 400)
    bar.setValue(400)
    assert page.pinned
    bar.setRange(0, 800)                      # a reply arrives
    assert bar.value() == 800                 # and is followed

    bar.setValue(100)                         # scrolled up to read something
    assert not page.pinned
    bar.setRange(0, 1200)                     # more arrives
    assert bar.value() == 100                 # and is not chased

    bar.setValue(1200)                        # back to the bottom
    assert page.pinned
    bar.setRange(0, 1600)
    assert bar.value() == 1600


def test_sending_goes_back_to_the_newest_message(qt, chat_window):
    page = chat_window.chat
    service = _ScriptedService(["Hello there."])
    page.service_provider = lambda: service
    bar = page.transcript_scroll.verticalScrollBar()
    bar.setRange(0, 400)
    bar.setValue(0)
    assert not page.pinned

    page.input.setPlainText("hello")
    page.send()
    _finish(qt, page)
    assert page.pinned, "sending a message should put you back at the end"


def test_opening_a_chat_starts_at_its_newest_message(qt, chat_window):
    page = chat_window.chat
    identifier = page.conversation.identifier
    bar = page.transcript_scroll.verticalScrollBar()
    bar.setRange(0, 400)
    bar.setValue(0)
    assert not page.pinned

    page.open_chat(identifier)
    assert page.pinned


def test_a_bubble_is_as_wide_as_its_words_and_no_wider(qt, chat_window):
    """A word-wrapped label asks its layout for almost nothing, which collapses
    a bubble into a narrow column of two-word lines. The width is measured from
    the text instead, and capped."""
    page = chat_window.chat
    page.conversation.append("user", "ok")
    page.conversation.append("user", "a considerably longer message that has to wrap "
                                     "somewhere sensible rather than after every second word")
    page.render()
    short, long_message = page.bubbles[-2], page.bubbles[-1]

    assert short.body.minimumWidth() < long_message.body.minimumWidth()
    assert long_message.body.minimumWidth() <= page.bubble_width()
    assert short.frame.maximumWidth() == long_message.frame.maximumWidth() == page.bubble_width()


def test_rebuilding_the_transcript_leaves_no_ghosts_behind(qt, chat_window):
    """Taking a widget out of a layout leaves it parented and visible where it
    was, so a rebuild that only schedules deletion paints the old transcript
    under the new one."""
    from prompt_master.ui.chat_page import MessageBubble

    page = chat_window.chat
    page.conversation.append("user", "one")
    page.render()
    page.conversation.append("assistant", "two")
    page.render()

    still_there = [bubble for bubble in page.transcript.findChildren(MessageBubble)
                   if bubble.parent() is page.transcript]
    assert len(still_there) == len(page.conversation.messages) == len(page.bubbles)


# ── what runs the model, at runtime ──────────────────────────────────────────

def _three_devices():
    """A card offered both ways, and the processor — what setup offers."""
    import dataclasses

    from prompt_master.core.models import CPU_INDEX, GpuInfo

    card = GpuInfo(0, "GPU-0", "NVIDIA GeForce RTX 4090", 24564, 22000, "560.94", 8.9)
    processor = GpuInfo(CPU_INDEX, "CPU", "Test Processor", 65413, 40000, "AMD64", None)
    return [card, dataclasses.replace(card, mixed=True), processor]


@pytest.fixture
def device_window(qt, window, monkeypatch):
    """A window whose device scan is a fixture rather than this machine."""
    from prompt_master.core.config import atomic_write_json
    from prompt_master.ui import main_window as module

    monkeypatch.setattr(module, "detect_devices", lambda *a, **k: _three_devices())
    atomic_write_json(window.paths.state_file, {
        "runtime": "runtime/llama-server.exe", "runtime_id": "llama-runtime-cuda12",
        "model": "models/model.gguf", "mmproj": "models/mmproj.gguf",
        "mode": "gpu", "gpu_index": 0, "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_device": "CUDA0", "gpu_device_name": "RTX 4090",
        "quantization": "Q4_K_M", "context_size": 16384, "gpu_layers": "all",
    })
    monkeypatch.setattr(module.installer, "runtime_ready", lambda *a, **k: True)
    monkeypatch.setattr(module.installer, "runtime_downloaded", lambda *a, **k: True)
    return window


def test_the_device_menu_offers_every_way_to_run_the_model_with_the_current_one_ticked(
        qt, device_window):
    device_window.populate_devices()
    actions = device_window.device_actions.actions()

    assert len(actions) == 3
    assert "4090" in actions[0].text() and "mixed" in actions[1].text()
    assert "no GPU used" in actions[2].text()
    ticked = [action for action in actions if action.isChecked()]
    assert len(ticked) == 1 and ticked[0].data().mode == "gpu"
    # The same wording setup uses, so one device is not two things.
    from prompt_master.inference.device_detection import describe
    assert actions[1].text() == describe(_three_devices()[1])


def test_changing_the_device_warns_switches_and_unloads_the_model(qt, device_window, monkeypatch):
    from prompt_master.ui import main_window as module

    asked, switched, stopped = [], [], []
    monkeypatch.setattr(module.QMessageBox, "question",
                        lambda _self, _title, text, *a, **k: asked.append(text)
                        or module.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(module.installer, "switch_device",
                        lambda paths, device, **kwargs: switched.append(device))
    monkeypatch.setattr(device_window.service, "stop", lambda: stopped.append(True))

    processor = _three_devices()[2]
    device_window.choose_device(processor)

    assert switched == [processor]
    assert stopped, "the model has to be let go of, or the switch means nothing"
    warning = asked[0]
    assert "unloaded" in warning and "loads again on your next generation" in warning
    assert "16-27 GiB" in warning
    assert "loads again" in device_window.status.text()
    assert "loads again" in device_window.chat.status.text()


def test_declining_the_warning_changes_nothing(qt, device_window, monkeypatch):
    from prompt_master.ui import main_window as module

    monkeypatch.setattr(module.QMessageBox, "question",
                        lambda *a, **k: module.QMessageBox.StandardButton.No)
    monkeypatch.setattr(module.installer, "switch_device",
                        lambda *a, **k: pytest.fail("switched without being told to"))
    device_window.choose_device(_three_devices()[2])


def test_choosing_the_device_already_running_does_nothing_at_all(qt, device_window, monkeypatch):
    from prompt_master.ui import main_window as module

    monkeypatch.setattr(module.QMessageBox, "question",
                        lambda *a, **k: pytest.fail("asked about the device already in use"))
    monkeypatch.setattr(module.installer, "switch_device", lambda *a, **k: pytest.fail("switched"))
    device_window.choose_device(_three_devices()[0])


def test_the_device_cannot_be_changed_while_something_is_generating(qt, device_window, monkeypatch):
    from prompt_master.ui import main_window as module

    told = []
    monkeypatch.setattr(device_window.chat, "busy", lambda: True)
    monkeypatch.setattr(module.QMessageBox, "information",
                        lambda _self, _title, text, *a, **k: told.append(text))
    monkeypatch.setattr(module.installer, "switch_device",
                        lambda *a, **k: pytest.fail("switched mid-generation"))

    device_window.choose_device(_three_devices()[2])
    assert told and "Stop" in told[0]


def test_a_switch_that_fails_says_so_and_leaves_the_model_loaded(qt, device_window, monkeypatch):
    from prompt_master.ui import main_window as module

    complained, stopped = [], []
    monkeypatch.setattr(module.QMessageBox, "question",
                        lambda *a, **k: module.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(module.QMessageBox, "critical",
                        lambda _self, _title, text, *a, **k: complained.append(text))
    monkeypatch.setattr(module.installer, "switch_device",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no such runtime")))
    monkeypatch.setattr(device_window.service, "stop", lambda: stopped.append(True))

    device_window.choose_device(_three_devices()[2])
    assert complained == ["no such runtime"] and stopped == []


def test_a_switch_names_a_download_only_when_there_is_one(qt, device_window, monkeypatch):
    from prompt_master.ui import main_window as module

    state = device_window.setup_state()
    monkeypatch.setattr(module.installer, "runtime_ready", lambda *a, **k: True)
    assert "download" not in device_window.switch_warning(_three_devices()[2], state)

    monkeypatch.setattr(module.installer, "runtime_ready", lambda *a, **k: False)
    monkeypatch.setattr(module.installer, "runtime_downloaded", lambda *a, **k: False)
    assert "downloaded first" in device_window.switch_warning(_three_devices()[2], state)
    monkeypatch.setattr(module.installer, "runtime_downloaded", lambda *a, **k: True)
    assert "unpacked from the download cache" in device_window.switch_warning(
        _three_devices()[2], state)


def test_the_warning_repeats_the_vram_shortfall_setup_would_have_shown(qt, device_window):
    from prompt_master.core.config import atomic_write_json

    state = device_window.setup_state()
    state["quantization"] = "Q8_K_P"           # 40 GiB wanted, 24 GiB card
    atomic_write_json(device_window.paths.state_file, state)

    warning = device_window.switch_warning(_three_devices()[0], device_window.setup_state())
    assert "more VRAM than this card reports" in warning and "mixed mode" in warning
    # Mixed mode has nothing to fall short of: the weights are in system RAM.
    assert "more VRAM" not in device_window.switch_warning(_three_devices()[1],
                                                           device_window.setup_state())


# ── unloading the model ──────────────────────────────────────────────────────

def test_unloading_gives_the_memory_back_and_says_when_it_comes_again(qt, window, monkeypatch):
    stopped = []
    monkeypatch.setattr(window.service, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(type(window.service.process), "running", property(lambda _self: True))

    window.unload_model()

    assert stopped
    assert "unloaded" in window.status.text()
    assert "loads again on your next generation" in window.status.text()
    assert "loads again" in window.chat.status.text()


def test_unloading_with_nothing_loaded_says_so(qt, window, monkeypatch):
    monkeypatch.setattr(window.service, "stop", lambda: None)
    window.unload_model()
    assert "No model was loaded" in window.status.text()


def test_the_model_cannot_be_unloaded_mid_generation(qt, window, monkeypatch):
    from prompt_master.ui import main_window as module

    told, stopped = [], []
    monkeypatch.setattr(window.chat, "busy", lambda: True)
    monkeypatch.setattr(window.service, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(module.QMessageBox, "information",
                        lambda _self, _title, text, *a, **k: told.append(text))

    window.unload_model()
    assert stopped == [] and told and "Stop" in told[0]


# ── which model runs, as against what runs it ────────────────────────────────

@pytest.fixture
def model_window(qt, window):
    """A window with a finished install behind it and a second model on disk."""
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(window.paths.state_file, {
        "runtime": "runtime/llama-server.exe", "runtime_id": "llama-runtime-cuda12",
        "model": "models/model.gguf", "mmproj": "models/mmproj.gguf",
        "mode": "gpu", "gpu_index": 0, "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_device": "CUDA0", "gpu_device_name": "RTX 4090",
        "quantization": "Q4_K_M", "context_size": 16384, "gpu_layers": "all",
    })
    for relative in ("models/model.gguf", "models/mmproj.gguf",
                     "models/Other-Q6_K_P.gguf", "models/Other-mmproj-f16.gguf"):
        target = window.paths.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stand-in")
    return window


class _Picked:
    """The dialog, already answered. Standing in for one nobody can click."""

    def __init__(self, model, mmproj=None):
        self.model, self.mmproj = model, mmproj

    def __call__(self, _paths, _parent=None): return self
    def exec(self): return 1
    def model_path(self): return self.model
    def mmproj_path(self): return self.mmproj


def test_the_dialog_opens_on_the_model_that_is_running(qt, model_window):
    from prompt_master.ui.model_chooser import ModelDialog

    dialog = ModelDialog(model_window.paths, model_window)
    try:
        assert dialog.model.text() == str((model_window.paths.root / "models/model.gguf").resolve())
        assert dialog.mmproj.text() == str((model_window.paths.root / "models/mmproj.gguf").resolve())
        none = next(button for button in dialog.findChildren(qt.QPushButton)
                    if button.text() == "None")
        none.click()
        assert dialog.mmproj_path() is None, "None means run it without one"
        assert dialog.model_path() == (model_window.paths.root / "models/model.gguf").resolve()
    finally:
        dialog.close()


def test_choosing_another_model_records_it_and_unloads_the_one_running(qt, model_window, monkeypatch):
    import json
    from prompt_master.ui import main_window as module

    stopped = []
    chosen = model_window.paths.root / "models" / "Other-Q6_K_P.gguf"
    projector = model_window.paths.root / "models" / "Other-mmproj-f16.gguf"
    monkeypatch.setattr(module, "ModelDialog", _Picked(chosen, projector))
    monkeypatch.setattr(model_window.service, "stop", lambda: stopped.append(True))

    model_window.choose_model()

    state = json.loads(model_window.paths.state_file.read_text(encoding="utf-8"))
    assert state["model"] == "models/Other-Q6_K_P.gguf"
    assert state["mmproj"] == "models/Other-mmproj-f16.gguf"
    assert state["quantization"] == "Q6_K_P"
    # The device it runs on is not what this menu changes.
    assert state["gpu_device"] == "CUDA0" and state["runtime_id"] == "llama-runtime-cuda12"
    assert stopped, "the server still holds the old weights until it is stopped"
    assert "Other-Q6_K_P.gguf" in model_window.status.text()
    assert "loads on your next generation" in model_window.chat.status.text()


def test_a_model_chosen_without_a_projector_says_images_are_gone(qt, model_window, monkeypatch):
    import json
    from prompt_master.ui import main_window as module

    chosen = model_window.paths.root / "models" / "Other-Q6_K_P.gguf"
    monkeypatch.setattr(module, "ModelDialog", _Picked(chosen))
    monkeypatch.setattr(model_window.service, "stop", lambda: None)

    model_window.choose_model()

    assert json.loads(model_window.paths.state_file.read_text(encoding="utf-8"))["mmproj"] == ""
    assert "No vision projector" in model_window.status.text()
    # And the bar keeps saying it, not just once at the moment it changed.
    model_window.refresh_status()
    assert "no vision" in model_window.status.text()


def test_an_image_is_refused_before_a_generation_starts_when_nothing_can_see_it(
        qt, model_window, monkeypatch, tmp_path):
    from PIL import Image
    from prompt_master.ui import main_window as module

    refused = []
    monkeypatch.setattr(module.QMessageBox, "critical",
                        lambda _self, _title, text, *a, **k: refused.append(text))
    monkeypatch.setattr(module, "ModelDialog",
                        _Picked(model_window.paths.root / "models" / "Other-Q6_K_P.gguf"))
    monkeypatch.setattr(model_window.service, "stop", lambda: None)
    model_window.choose_model()

    picture = tmp_path / "still.png"
    Image.new("RGB", (32, 32), (10, 10, 10)).save(picture)
    model_window.image_path = picture
    model_window.select(model_window.mode, "i2v")
    model_window.intent.setPlainText("A red ball rolls across a table")

    model_window.generate()

    assert model_window.thread is None, "nothing was started"
    assert refused and "no vision projector" in refused[0]


def test_attaching_a_picture_in_a_chat_says_so_too(qt, model_window, monkeypatch):
    from prompt_master.ui import main_window as module
    from prompt_master.ui import chat_page as chat_module

    monkeypatch.setattr(module, "ModelDialog",
                        _Picked(model_window.paths.root / "models" / "Other-Q6_K_P.gguf"))
    monkeypatch.setattr(model_window.service, "stop", lambda: None)
    model_window.choose_model()

    picture = model_window.paths.root / "still.png"
    picture.write_bytes(b"not really a picture")
    monkeypatch.setattr(chat_module.QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(picture), ""))

    model_window.chat.attach_image()

    assert model_window.chat.attachment == picture, "it is still attached"
    assert "no vision projector" in model_window.chat.status.text()


def test_the_model_cannot_be_changed_mid_generation(qt, model_window, monkeypatch):
    from prompt_master.ui import main_window as module

    told = []
    monkeypatch.setattr(model_window.chat, "busy", lambda: True)
    monkeypatch.setattr(module, "ModelDialog",
                        lambda *a, **k: pytest.fail("the dialog must not open"))
    monkeypatch.setattr(module.QMessageBox, "information",
                        lambda _self, _title, text, *a, **k: told.append(text))

    model_window.choose_model()
    assert told and "Stop" in told[0]


def test_the_model_dialog_is_finger_sized_too(qt, model_window):
    from prompt_master.ui.model_chooser import ModelDialog

    dialog = ModelDialog(model_window.paths, model_window)
    try:
        small = [(type(widget).__name__, widget.sizeHint().height())
                 for widget in targets(qt, dialog)
                 if widget.sizeHint().height() < FINGERTIP]
        assert small == []
    finally:
        dialog.close()
