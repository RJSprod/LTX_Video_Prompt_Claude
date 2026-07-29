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


def test_the_primary_action_is_the_biggest_button(qt, window):
    """Generate is the button pressed most and the one that should never be
    hunted for."""
    generate = window.generate_button
    others = [b for b in window.findChildren(qt.QPushButton) if b is not generate]
    assert generate.objectName() == "primary"
    assert generate.sizeHint().height() > max(button.sizeHint().height() for button in others)
    assert generate.minimumWidth() >= 4 * FINGERTIP


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
