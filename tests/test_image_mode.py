"""Image mode, checked at the seam the application adds.

The engine's own behaviour is covered by ``test_image_engine.py`` and
``test_selection_quality.py``, which are the bundle's own 157 tests and import
no Qt. What is tested here is everything those cannot see: that the client
adapter hands llama-server what the engine asked for and pulls the answer back
in the fragments it arrived in, that the page's controls are the engine's own
banks rather than a second copy of them, and that the five things which have to
follow the model drop-down actually follow it.

The reactive tests are the ones worth keeping honest. A negative field on a
model with no negative conditioning, a JSON shape on a model that cannot parse
one, a fifth reference slot on a model that reads four — each of those is a
control that appears to work and silently does not, which is the failure mode
this page has most of.
"""

from __future__ import annotations

import base64
import io
import threading

import pytest

from prompt_master.core.paths import AppPaths
from prompt_master.image_engine import (AUTO, CHOOSE, FLUX_KLEIN_9B, KREA_2_RAW,
                                        KREA_2_TURBO, ImageControls, PromptShape,
                                        build_system, option_banks, profile_for)
from prompt_master.inference.image_client import ImageChatClient

WRITTEN_PROMPT = (
    "A shipwright in her fifties kneels on a half-planked hull, driving a caulking "
    "iron into a seam, documentary photograph, lit by soft daylight through the "
    "boatshed doors, weathered timber and oakum, visible skin pores and fine lines.")

WRITTEN_EDIT = (
    "Replace the wooden door on the left of the frame with a riveted steel door of "
    "the same size and position. Keep the surrounding brickwork, the light direction "
    "and the photograph's grain exactly as they are.")


# ── doubles ──────────────────────────────────────────────────────────────────

class FakeLlama:
    """``LlamaClient``'s shape: pushes fragments, returns the whole reply.

    Answers by pass, the way the real model would: the selection pass is asked
    for JSON and the writer pass for prose, and the two must not be confused
    with each other anywhere downstream.
    """

    SELECTION = '{"lighting": "Soft window light", "mood": "Weary"}'
    REFINE = "harsh on-camera flash, blown white shirt"

    def __init__(self, prose: str = WRITTEN_PROMPT):
        self.prose = prose
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95):
        system = messages[0]["content"]
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "seed": seed,
                           "temperature": temperature, "top_p": top_p})
        if "choose settings" in system.lower():
            reply = self.SELECTION
        elif "failure modes" in system.lower():
            reply = self.REFINE
        else:
            reply = self.prose
        pieces = []
        for word in reply.split(" "):
            if cancel is not None and cancel.is_set():
                break
            pieces.append(word + " ")
            on_text(word + " ")
        return "".join(pieces)


class FakeService:
    """The inference service, minus the process it manages."""

    def __init__(self, client=None, vision=True):
        self.client_object = client or FakeLlama()
        self.vision = vision
        self.asked_for_vision = None

    def client(self, needs_vision=False):
        self.asked_for_vision = needs_vision
        return self.client_object

    def vision_ready(self):
        return self.vision


def a_jpeg() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ── the adapter ──────────────────────────────────────────────────────────────

def test_the_adapter_yields_the_fragments_as_they_arrive():
    """The engine pulls and the client pushes. The whole point of the bridge is
    that what comes back out is still in pieces."""
    client = ImageChatClient(FakeLlama("one two three"))
    chunks = list(client.stream_chat(system="s", user="u", temperature=0.8, top_p=0.9, seed=1))
    assert chunks == ["one ", "two ", "three "]
    assert "".join(chunks) == "one two three "


def test_the_still_goes_ahead_of_the_words():
    """llama.cpp resolves the image part against the text after it, so a still
    that follows the question is a still the question could not be about."""
    llama = FakeLlama()
    list(ImageChatClient(llama).stream_chat(system="s", user="u", temperature=0.8,
                                            top_p=0.9, seed=1, image_jpeg_b64=a_jpeg()))
    content = llama.calls[0]["messages"][1]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[1]["text"] == "u"


def test_a_bare_base64_still_becomes_a_data_url():
    """The engine's protocol names the parameter ``image_jpeg_b64`` and this
    application produces a data URL. Both have to arrive somewhere usable."""
    llama = FakeLlama()
    list(ImageChatClient(llama).stream_chat(system="s", user="u", temperature=0.8,
                                            top_p=0.9, seed=1, image_jpeg_b64="QUJD"))
    url = llama.calls[0]["messages"][1]["content"][0]["image_url"]["url"]
    assert url == "data:image/jpeg;base64,QUJD"


def test_no_message_carries_an_image_part_when_there_is_no_image():
    llama = FakeLlama()
    list(ImageChatClient(llama).stream_chat(system="s", user="u", temperature=0.8,
                                            top_p=0.9, seed=1))
    assert llama.calls[0]["messages"][1]["content"] == "u"


def test_a_failure_on_the_request_thread_is_raised_on_the_asking_thread():
    """The engine's own recovery — the selection pass falls back to keyword
    matching, the refine pass keeps the banks — only runs if it sees the
    exception. An exception on a thread nobody joins is an exception nobody
    sees."""

    class Broken:
        def stream_chat(self, *args, **kwargs):
            raise RuntimeError("server went away")

    with pytest.raises(RuntimeError, match="server went away"):
        list(ImageChatClient(Broken()).stream_chat(system="s", user="u", temperature=0.8,
                                                   top_p=0.9, seed=1))


def test_cancelling_stops_the_request_rather_than_only_its_display():
    cancel = threading.Event()
    cancel.set()
    chunks = list(ImageChatClient(FakeLlama("one two three"), cancel).stream_chat(
        system="s", user="u", temperature=0.8, top_p=0.9, seed=1))
    assert chunks == []


def test_the_sampling_the_engine_asked_for_is_what_goes_on_the_wire():
    llama = FakeLlama()
    list(ImageChatClient(llama).stream_chat(system="s", user="u", temperature=0.31,
                                            top_p=0.77, seed=99))
    assert llama.calls[0]["temperature"] == 0.31
    assert llama.calls[0]["top_p"] == 0.77
    assert llama.calls[0]["seed"] == 99


# ── the page ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qt():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    application = widgets.QApplication.instance() or widgets.QApplication([])
    yield widgets
    application.processEvents()


@pytest.fixture
def page(qt, tmp_path):
    from PySide6 import QtCore

    from prompt_master.ui.image_page import ImagePage

    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    made = ImagePage(paths, lambda: FakeService())
    yield made
    made.shutdown()
    # Destroyed, not merely dropped. Qt's scroller keeps a reference to every
    # viewport ``touch.flickable`` grabbed, so a page that goes out of scope is
    # still alive — and setting a style sheet re-polishes every widget alive in
    # the process, which is what every window the rest of the suite builds
    # does. Forty pages left lying about here made those windows crawl.
    made.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def choose_model(page, key):
    page.model.setCurrentIndex(page.model.findData(key))


def choose_task(page, task):
    page.task.setCurrentIndex(page.task.findData(task))


def test_every_control_on_the_page_is_at_least_a_fingertip_tall(page, qt):
    """The same question ``test_touch_ui.py`` asks of the window, asked here
    because this page is built on demand and so is not in the window that suite
    measures. Not what the style sheet asks for — what the widgets report.

    The editor inside a spin box is skipped: it is not aimed at, it fills the
    spin box that is, and it reports the height of its text.
    """
    from PySide6.QtWidgets import QApplication

    from prompt_master.ui import touch

    # The style sheet lives on the application, so a page built outside a
    # window has to be given it before anything is measured.
    QApplication.instance().setStyleSheet(touch.stylesheet(1.0))
    kinds = (qt.QPushButton, qt.QComboBox, qt.QCheckBox, qt.QSpinBox, qt.QLineEdit)
    found = [widget for kind in kinds for widget in page.findChildren(kind)
             if not isinstance(widget.parentWidget(), qt.QAbstractSpinBox)]
    assert len(found) > 50, "the controls were not found, so nothing was checked"
    small = [(type(w).__name__, w.text() if hasattr(w, "text") else "", w.sizeHint().height())
             for w in found if w.sizeHint().height() < 44]
    assert small == []


def test_the_biggest_button_on_the_page_is_the_one_it_is_for(page, qt):
    """The same rule the other two pages are held to in ``test_touch_ui.py``:
    the button pressed most is the one that should never be hunted for."""
    from PySide6.QtWidgets import QApplication

    from prompt_master.ui import touch

    QApplication.instance().setStyleSheet(touch.stylesheet(1.0))
    # What the window does when the display size is chosen. A page built on its
    # own has not been told a size yet.
    page.apply_metrics(touch.metrics(1.0))
    primary = page.generate_button
    others = [b for b in page.findChildren(qt.QPushButton) if b is not primary]
    assert primary.objectName() == "primary"
    assert primary.sizeHint().height() > max(b.sizeHint().height() for b in others)
    assert primary.minimumWidth() >= 4 * 44


def test_every_long_drop_down_flicks_and_has_finger_sized_rows(page, qt):
    """Twenty operations, twenty lighting states, thirty-eight media. A popup
    that only scrolls by its scroll bar is the worst target on the page."""
    from PySide6.QtWidgets import QScroller, QStyledItemDelegate

    assert page.choices["visual_style"].count() > 40
    for name in ("visual_style", "lighting", "operation"):
        box = page.choices[name]
        assert QScroller.hasScroller(box.view().viewport()), name
        assert type(box.itemDelegate()) is QStyledItemDelegate, name


def test_the_areas_that_scroll_are_all_flickable(page, qt):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScroller

    scrollable = [page.intent, page.extra_negative, page.prompt_out, page.negative_out,
                  page.notes_out]
    scrollable += page.findChildren(qt.QScrollArea)
    for widget in scrollable:
        viewport = widget.viewport()
        assert QScroller.hasScroller(viewport), f"{type(widget).__name__} does not flick"
        assert viewport.testAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)


def test_every_drop_down_offers_the_three_sentinels_first(page):
    """Auto, Choose for me and Random are answers to a different question than
    any option is, so they are above the bank on every control that has one."""
    sentinels = list(option_banks()["sentinels"].values())
    for name, box in page.choices.items():
        assert [box.itemData(i) for i in range(3)] == sentinels, name


def test_the_option_lists_are_the_engines_own(page):
    """The rule ``prompt_engine/options.py`` already enforces for prompt mode: a
    control holding its own copy of a list is a control that will drift."""
    banks = option_banks()
    for name, key in (("shot_type", "shot_types"), ("lighting", "lighting"),
                      ("mood", "moods"), ("aspect_ratio", "aspect_ratios"),
                      ("operation", "edit_operations"), ("localization", "localizations")):
        box = page.choices[name]
        offered = [box.itemData(i) for i in range(box.count()) if box.itemData(i)]
        assert offered[3:] == list(banks[key]), name


def test_the_medium_keeps_the_engines_own_grouping(page):
    box = page.choices["visual_style"]
    offered = [box.itemData(i) for i in range(box.count()) if box.itemData(i)]
    every_style = [style for group in option_banks()["style_groups"].values() for style in group]
    assert offered[3:] == every_style
    # The headings are rows nobody can land on, exactly as in prompt mode.
    headings = [i for i in range(box.count())
                if box.itemText(i).startswith("— ") and box.itemData(i) is None]
    assert len(headings) == len(option_banks()["style_groups"])


def test_the_preservation_targets_are_all_offered(page):
    assert list(page.preservation) == list(option_banks()["preservation_targets"])
    assert [name for name, box in page.preservation.items() if box.isChecked()] == \
        list(page.edit_defaults.preservation)


# ── the five things that follow the model ────────────────────────────────────

def test_the_negative_controls_are_hidden_unless_the_model_has_one(page, qt):
    """Hidden, not disabled. A greyed-out field still says the model has one."""
    for key, wanted in ((KREA_2_TURBO.key, False), (FLUX_KLEIN_9B.key, False),
                        (KREA_2_RAW.key, True)):
        choose_model(page, key)
        assert page.negative_wanted() is wanted, key
        assert page.negative_box.isVisibleTo(page) is wanted, key
        assert page.negative_out.isVisibleTo(page) is wanted, key


def test_no_editing_path_offers_a_negative(page):
    """No supported editor takes one, including the profile that takes one when
    it is generating."""
    choose_model(page, KREA_2_RAW.key)
    page.allow_experimental.setChecked(True)
    choose_task(page, "edit")
    choose_model(page, KREA_2_RAW.key)
    assert page.negative_wanted() is False
    assert page.negative_box.isVisibleTo(page) is False


def test_hex_swatches_are_offered_only_where_a_hex_value_binds(page):
    for key, wanted in ((FLUX_KLEIN_9B.key, True), (KREA_2_TURBO.key, False),
                        (KREA_2_RAW.key, False)):
        choose_model(page, key)
        assert page.hex_swatches.isVisibleTo(page) is wanted, key


def test_the_json_shape_is_offered_only_where_it_is_honoured(page):
    """The engine downgrades JSON to prose on a model that gains nothing from
    it. Making that choice quietly is right; offering it is not."""
    choose_model(page, FLUX_KLEIN_9B.key)
    assert page.shape.findData(PromptShape.JSON.value) >= 0
    page.shape.setCurrentIndex(page.shape.findData(PromptShape.JSON.value))
    # A str enum survives a round trip through Qt's item data as the bare
    # string it also is, and the engine tests the member by identity.
    assert page.image_settings().prompt_shape is PromptShape.JSON
    choose_model(page, KREA_2_TURBO.key)
    assert page.shape.findData(PromptShape.JSON.value) < 0
    assert page.image_settings().prompt_shape is PromptShape.PROSE
    # And the field goes with it: a drop-down left holding one item is a
    # control that looks like a choice and is not.
    assert page.shape_field.isVisibleTo(page) is False


def test_edit_moves_off_a_model_that_cannot_edit(page):
    """Krea 2 does not ship with editing at all. The engine raises for it, and
    it raises after the user has waited for a server to start."""
    choose_model(page, KREA_2_TURBO.key)
    choose_task(page, "edit")
    assert page.profile().key == FLUX_KLEIN_9B.key
    assert page.model.model().item(page.model.findData(KREA_2_TURBO.key)).isEnabled() is False


def test_the_experimental_path_is_reachable_and_says_what_it_costs(page):
    choose_task(page, "edit")
    page.allow_experimental.setChecked(True)
    krea = page.model.findData(KREA_2_TURBO.key)
    assert page.model.model().item(krea).isEnabled() is True
    choose_model(page, KREA_2_TURBO.key)
    assert page.requirements.isVisibleTo(page)
    assert "third-party" in page.requirements.text()
    assert page.edit_settings().allow_experimental_edit is True


def test_reference_slots_are_capped_by_the_model(page):
    choose_task(page, "edit")
    for key, allowed in ((FLUX_KLEIN_9B.key, 4), (KREA_2_TURBO.key, 2)):
        page.allow_experimental.setChecked(True)
        choose_model(page, key)
        shown = [slot for slot in page.slots if slot.isVisibleTo(page)]
        assert len(shown) == allowed == profile_for(key).max_reference_images


def test_a_slot_the_model_cannot_read_is_emptied_as_well_as_hidden(page, tmp_path):
    """An attachment nothing will ever send is worse than no attachment."""
    from PIL import Image

    still = tmp_path / "still.png"
    Image.new("RGB", (8, 8), "white").save(still)
    choose_task(page, "edit")
    choose_model(page, FLUX_KLEIN_9B.key)
    for slot in page.slots:
        slot.path = still
    page.allow_experimental.setChecked(True)
    choose_model(page, KREA_2_TURBO.key)
    assert [slot.path for slot in page.slots] == [still, still, None, None]
    assert len(page.edit_settings().references) == 2


def test_the_numbers_the_image_model_needs_follow_the_profile(page):
    """Steps, guidance and the frame are not the writer's business and change
    with the model, so they are shown rather than left to be looked up."""
    choose_model(page, KREA_2_RAW.key)
    assert "52 steps" in page.facts.text() and "CFG 3.5" in page.facts.text()
    choose_model(page, FLUX_KLEIN_9B.key)
    assert "4 steps" in page.facts.text()
    choose_task(page, "edit")
    assert "denoise 0.5" in page.facts.text()
    page.choices["edit_strength"].setCurrentIndex(
        page.choices["edit_strength"].findData("Preserve — smallest change that works"))
    assert f"denoise {FLUX_KLEIN_9B.edit_denoise_preserve:g}" in page.facts.text()


def test_the_frame_follows_the_ratio_and_the_long_edge(page):
    choose_model(page, FLUX_KLEIN_9B.key)
    page.choices["aspect_ratio"].setCurrentIndex(
        page.choices["aspect_ratio"].findData("9:16 tall"))
    assert "576×1024" in page.facts.text()
    page.long_edge.setValue(1536)
    assert "864×1536" in page.facts.text()


def test_a_frame_that_is_not_settled_yet_says_so(page):
    """Choose for me and Random are resolved during the generation, so the size
    shown before one is the engine's fallback rather than what you will get."""
    page.choices["aspect_ratio"].setCurrentIndex(
        page.choices["aspect_ratio"].findData(CHOOSE))
    assert "until the ratio is settled" in page.facts.text()


# ── the roles the generation prompt knows about ──────────────────────────────

def test_every_generate_reference_role_changes_the_system_prompt():
    """``GENERATE_REFERENCE_ROLES`` is the one list this page holds, because the
    engine states these three as prose rather than as a bank. This is what stops
    it drifting into offering a role the writer is given nothing for."""
    from prompt_master.ui.image_page import GENERATE_REFERENCE_ROLES

    from prompt_master.image_engine import ImagePromptEngine

    written = {}
    for value, _label in GENERATE_REFERENCE_ROLES:
        controls = ImageControls(has_reference_image=True, reference_role=value)
        written[value] = build_system(ImagePromptEngine(None).plan("a street", controls))
    plain = build_system(ImagePromptEngine(None).plan(
        "a street", ImageControls(has_reference_image=False)))
    assert len(set(written.values())) == len(GENERATE_REFERENCE_ROLES)
    for value, text in written.items():
        assert text != plain, value


# ── settings the engine will accept ──────────────────────────────────────────

def test_the_page_builds_controls_the_engine_plans_from(page):
    from prompt_master.image_engine import ImagePromptEngine

    page.intent.setPlainText("a shipwright caulking a hull")
    page.choices["lighting"].setCurrentIndex(page.choices["lighting"].findData("Firelight"))
    brief = ImagePromptEngine(None).plan("a shipwright caulking a hull", page.image_settings())
    assert "firelight" in " ".join(brief.light).lower()
    assert brief.width and brief.height


def test_the_page_builds_edit_controls_the_engine_plans_from(page):
    from prompt_master.image_engine import ImagePromptEngine

    choose_task(page, "edit")
    page.target_element.setText("the wooden door")
    brief = ImagePromptEngine(None).plan_edit("swap the door for a steel one",
                                              page.edit_settings())
    assert brief.target_element == "the wooden door"
    assert brief.preservation == page.edit_defaults.preservation


def test_the_task_and_the_model_are_remembered(qt, tmp_path):
    from prompt_master.ui.image_page import ImagePage

    paths = AppPaths(tmp_path)
    paths.create_managed_dirs()
    service = FakeService()
    first = ImagePage(paths, lambda: service)
    choose_model(first, KREA_2_RAW.key)
    choose_task(first, "generate")
    first.shutdown()

    second = ImagePage(paths, lambda: service)
    assert second.model.currentData() == KREA_2_RAW.key
    assert second.current_task() == "generate"
    second.shutdown()


# ── one generation, end to end ───────────────────────────────────────────────

def run(worker):
    """Drive a worker on this thread and collect what it emitted.

    Called directly rather than moved onto a QThread: the worker is the thing
    under test, and a test that has to pump an event loop to see it finish
    proves less about the worker and more about the loop.
    """
    seen = {"chunks": [], "status": [], "result": None, "failed": None}
    worker.chunk.connect(seen["chunks"].append)
    worker.status.connect(seen["status"].append)
    worker.done.connect(lambda result: seen.update(result=result))
    worker.failed.connect(lambda message: seen.update(failed=message))
    worker.run()
    return seen


def test_a_generation_runs_end_to_end(page, qt):
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    llama = FakeLlama()
    worker = ImageWorker(FakeService(llama), GENERATE, "a shipwright caulking a hull",
                         page.image_settings(), None, False)
    seen = run(worker)
    assert seen["failed"] is None
    assert seen["result"].positive.startswith("A shipwright")
    assert "Writing…" in seen["status"]
    assert "".join(seen["chunks"]).strip() == WRITTEN_PROMPT.strip()


def test_an_edit_runs_end_to_end(page, qt):
    from prompt_master.ui.image_page import EDIT, ImageWorker

    choose_task(page, "edit")
    worker = ImageWorker(FakeService(FakeLlama(WRITTEN_EDIT)), EDIT,
                         "swap the wooden door for a steel one",
                         page.edit_settings(), None, False)
    seen = run(worker)
    assert seen["failed"] is None
    assert seen["result"].instruction.startswith("Replace the wooden door")
    assert seen["result"].denoise == FLUX_KLEIN_9B.edit_denoise_default


def test_the_selection_pass_never_reaches_the_output_pane(page, qt):
    """Choose for me and the smart-negative pass go out over the same client as
    the writer. Only the writer's fragments are yielded back, and only yielded
    fragments reach the pane — which is the whole reason the adapter pulls."""
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    controls = page.image_settings()
    controls.lighting = CHOOSE
    controls.mood = CHOOSE
    llama = FakeLlama()
    seen = run(ImageWorker(FakeService(llama), GENERATE, "a lone bartender closing up",
                           controls, None, False))
    written = "".join(seen["chunks"])
    assert FakeLlama.SELECTION not in written
    assert "{" not in written
    assert seen["result"].chosen["lighting"] == "Soft window light"
    assert seen["result"].chosen_source["mood"] == "model"


def test_the_smart_negative_pass_only_runs_where_a_negative_exists(page, qt):
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    for key, refined in ((KREA_2_RAW.key, True), (KREA_2_TURBO.key, False)):
        choose_model(page, key)
        controls = page.image_settings()
        controls.smart_negative = True
        llama = FakeLlama()
        seen = run(ImageWorker(FakeService(llama), GENERATE, "a lone bartender", controls,
                               None, False))
        asked = any("failure modes" in call["messages"][0]["content"].lower()
                    for call in llama.calls)
        assert asked is refined, key
        assert bool(seen["result"].negative) is refined, key


def test_a_still_is_only_asked_for_when_one_is_attached(page, qt):
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    service = FakeService()
    controls = page.image_settings()
    run(ImageWorker(service, GENERATE, "a hull", controls, None, False))
    assert service.asked_for_vision is False
    run(ImageWorker(service, GENERATE, "a hull", controls, a_jpeg(), True))
    assert service.asked_for_vision is True


def test_a_failed_generation_is_reported_rather_than_swallowed(page, qt):
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    class Refuses:
        def client(self, needs_vision=False):
            raise RuntimeError("Setup is incomplete")

    seen = run(ImageWorker(Refuses(), GENERATE, "a hull", page.image_settings(), None, False))
    assert seen["result"] is None
    assert "Setup is incomplete" in seen["failed"]


def test_what_choose_picked_is_shown_beside_the_control(page, qt):
    """A choice nobody can see is no better than Auto, and the value shown is
    what goes in the box next time."""
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    controls = page.image_settings()
    controls.lighting = CHOOSE
    controls.mood = CHOOSE
    seen = run(ImageWorker(FakeService(), GENERATE, "a lone bartender", controls, None, False))
    page.completed(seen["result"])
    assert "Soft window light" in page.resolved["lighting"].text()
    assert "model" in page.resolved["lighting"].text()
    assert page.resolved["lighting"].isVisibleTo(page)
    assert page.resolved["shot_type"].text() == ""       # never asked, so nothing said


def test_the_notes_the_engine_returns_are_surfaced(page, qt):
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    choose_model(page, FLUX_KLEIN_9B.key)
    seen = run(ImageWorker(FakeService(), GENERATE, "a hull", page.image_settings(),
                           None, False))
    page.completed(seen["result"])
    notes = page.notes_out.toPlainText()
    assert "no negative conditioning" in notes
    assert "licence" in notes.lower()


def test_the_saved_text_carries_the_settings_the_prompt_was_written_for(page, qt):
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    choose_model(page, KREA_2_RAW.key)
    seen = run(ImageWorker(FakeService(), GENERATE, "a hull", page.image_settings(),
                           None, False))
    page.completed(seen["result"])
    saved = page.as_text()
    assert "POSITIVE" in saved and "NEGATIVE" in saved
    assert "52 steps" in saved and f"seed {seen['result'].seed}" in saved


def test_clear_takes_the_page_back_to_empty(page, qt):
    from prompt_master.ui.image_page import GENERATE, ImageWorker

    page.intent.setPlainText("a hull")
    seen = run(ImageWorker(FakeService(), GENERATE, "a hull", page.image_settings(),
                           None, False))
    page.completed(seen["result"])
    page.clear()
    assert page.intent.toPlainText() == ""
    assert page.prompt_out.toPlainText() == ""
    assert all(note.text() == "" for note in page.resolved.values())


def test_a_control_left_alone_stays_the_engines_default(page):
    """Auto is the default on almost everything, and Auto means the facet is
    left out of the brief entirely rather than guessed at."""
    controls = page.image_settings()
    assert controls.shot_type == AUTO
    assert controls.render_detail == ImageControls().render_detail
    assert controls.aspect_ratio == ImageControls().aspect_ratio
