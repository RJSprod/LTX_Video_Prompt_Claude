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
    there is now one of those per page: Generate in prompt mode, Send in
    conversation mode. Each has to win on its own page rather than in the
    window, because only one page is ever on screen."""
    for page, primary in ((window.pages.widget(0), window.generate_button),
                          (window.chat, window.chat.send_button)):
        others = [b for b in page.findChildren(qt.QPushButton) if b is not primary]
        assert primary.objectName() == "primary"
        assert primary.sizeHint().height() > max(button.sizeHint().height() for button in others)
    assert window.generate_button.minimumWidth() >= 4 * FINGERTIP
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

    scrollable = [window.intent, window.lexicon, window.negative_extra,
                  window.positive, window.negative]
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


def test_metrics_grow_with_the_scale_and_never_go_under_a_fingertip():
    from prompt_master.ui import touch

    for name, scale in touch.SCALES.items():
        sizes = touch.metrics(scale)
        assert sizes["target"] >= FINGERTIP, name
        assert sizes["primary"] > sizes["action"] >= sizes["target"], name
    assert touch.metrics(1.35)["text"] > touch.metrics(1.0)["text"]


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


def test_the_mode_drop_down_switches_the_page_and_is_remembered(qt, chat_window, tmp_path):
    """One dropdown, two pages, and the one you left it on next time."""
    from prompt_master.ui.main_window import CONVERSATION_MODE, PROMPT_MODE, MainWindow

    assert chat_window.pages.currentWidget() is chat_window.chat
    chat_window.select_mode(PROMPT_MODE)
    assert chat_window.pages.currentWidget() is chat_window.pages.widget(0)
    assert chat_window.generate_button.isVisible() or not chat_window.isVisible()

    chat_window.select_mode(CONVERSATION_MODE)
    reopened = MainWindow(AppPaths(tmp_path))
    try:
        assert reopened.pages.currentWidget() is reopened.chat
        assert reopened.chosen(reopened.mode_selector) == CONVERSATION_MODE
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
