"""Making the window usable with a finger.

Two things, kept together because they are the same decision seen twice.

*Size.* Every interactive target is at least ``TARGET`` logical pixels tall.
That is the number the platform guidelines converge on for a fingertip, and it
is what the metrics below are derived from rather than chosen around: a combo
box, its drop-down arrow, the rows of its popup, a spin box's steppers, a
checkbox's indicator and a scroll bar's handle are all sized from it, because a
control is only as touchable as its smallest part. The display size multiplies
that number, and the two settings below 1.0 — see ``COMPACT`` — are the one
place it is allowed under the floor, by asking for it.

*Scrolling.* A touch screen scrolls by dragging the content, not by finding a
scroll bar. ``flickable`` grabs the touch gesture for a widget's viewport, which
is what turns a drag anywhere inside it into a kinetic scroll. It is grabbed for
touch only: a mouse drag keeps selecting text and moving sliders, so nothing a
desktop user does changes.

The style sheet sets geometry and type size, never colour. The platform palette
is what makes a window look native in light mode and in dark mode, and a
touch-friendly layout is no reason to take that over. The one exception is the
primary action, which is coloured because it is the one control that should be
findable without reading.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QAbstractSpinBox,
                               QComboBox, QHBoxLayout, QLabel, QPushButton, QScroller,
                               QScrollerProperties, QSlider, QStyle,
                               QStyledItemDelegate, QVBoxLayout, QWidget)

from prompt_master.core.config import atomic_write_json, read_json

# The fingertip. Everything else is a multiple of it.
TARGET = 48

# What the View menu offers, smallest first. A 10" tablet held at arm's length
# and a 27" panel on a desk want different numbers, and no single default is
# right for both.
SCALES: dict[str, float] = {"Smaller": 0.72, "Small": 0.85, "Comfortable": 1.0,
                            "Large": 1.15, "Larger": 1.35}
DEFAULT_SCALE = "Comfortable"

# The two sizes below Comfortable, which deliberately go under the fingertip
# floor the rest of this module is built around: 41 logical pixels at Small and
# 35 at Smaller, against the 48 a fingertip wants. They exist because the
# prompt-mode window has a great many controls and a large monitor with a mouse
# on it is a real way to use this application — but they are a choice made
# explicitly, never a default, and a finger will start to miss things at
# Smaller. Everything scales together, so nothing overlaps; it only gets small.
COMPACT = ("Smaller", "Small")

SETTINGS_FILE = "ui.json"


def metrics(scale: float) -> dict[str, int]:
    """Every number the style sheet uses, from one scale factor."""
    unit = round(TARGET * scale)
    return {
        "target": unit,
        "action": round(unit * 1.15),      # buttons: the things aimed at most
        "primary": round(unit * 1.3),      # Generate, findable without reading
        "text": round(15 * scale),
        "label": round(14 * scale),
        "heading": round(16 * scale),
        "output": round(16 * scale),       # prompts are read, not skimmed
        "pad": round(10 * scale),
        "gap": round(12 * scale),
        "indicator": round(unit * 0.62),   # checkbox box
        "stepper": round(unit * 0.9),      # spin box up/down
        "bubble": round(unit * 0.38),      # corner of a chat bubble
        "bar": round(unit * 0.42),         # scroll bar: wide enough to drag
        "grip": round(unit * 1.3),         # shortest a scroll handle may get
    }


def stylesheet(scale: float = 1.0) -> str:
    m = metrics(scale)
    return f"""
* {{ font-size: {m['text']}px; }}

QLabel#fieldLabel {{ font-size: {m['label']}px; font-weight: 600; }}
QLabel#sectionTitle {{ font-size: {m['heading']}px; font-weight: 700; }}
QLabel#status {{ font-size: {m['label']}px; }}

/* Rounding a button replaces the frame the platform would have drawn, so the
   frame has to be drawn here or the button reads as a line of text — the last
   thing a target being aimed at with a finger should look like. Grey at low
   alpha stands out from a light window and a dark one alike. */
QPushButton {{
    min-height: {m['action']}px; padding: 0 {m['gap'] * 2}px;
    border-radius: {m['pad']}px; font-size: {m['text']}px;
    border: 1px solid rgba(128, 128, 128, 0.55); background: rgba(128, 128, 128, 0.14);
}}
QPushButton:hover {{ background: rgba(128, 128, 128, 0.24); }}
QPushButton:pressed {{ background: rgba(128, 128, 128, 0.38); }}
QPushButton:disabled {{ color: rgba(128, 128, 128, 0.8); border-color: rgba(128, 128, 128, 0.3); background: transparent; }}
QPushButton#primary {{
    min-height: {m['primary']}px; font-size: {m['heading']}px; font-weight: 700;
    background: #2563eb; color: #ffffff; border: none;
}}
QPushButton#primary:disabled {{ background: rgba(128, 128, 128, 0.45); }}
QPushButton#primary:pressed {{ background: #1d4ed8; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    min-height: {m['target']}px; padding: 0 {m['pad']}px; border-radius: {m['pad']}px;
}}
QComboBox::drop-down {{ width: {m['target']}px; }}
QComboBox QAbstractItemView::item {{ min-height: {m['target']}px; }}

/* The − and + of a stepper: square, target-sized, and heavy enough to read at
   a glance. Qt's own stacked arrows are half a target tall each, which is the
   smallest thing in the window and the one aimed at to change a number by one. */
QPushButton#stepper {{
    min-width: {m['stepper']}px; max-width: {m['stepper']}px;
    min-height: {m['target']}px; padding: 0; font-size: {m['heading']}px; font-weight: 700;
}}

/* The buttons that belong to a message — its ⋯ and its version pager. Square
   and target-sized rather than the wide ones the rest of the window uses: they
   sit against a bubble, and a full-width button beside a short message is the
   loudest thing in the transcript. */
QPushButton#bubbleAction {{
    min-width: {m['stepper']}px; max-width: {m['stepper']}px;
    min-height: {m['target']}px; padding: 0; font-size: {m['label']}px;
}}

QCheckBox {{ min-height: {m['target']}px; spacing: {m['gap']}px; }}
QCheckBox::indicator {{ width: {m['indicator']}px; height: {m['indicator']}px; }}

/* A slider is dragged, so its handle is the target: a thumb, not a sliver.
   The filled side is the one colour in the window besides the primary action,
   because a slider with no fill does not say which way is more. */
QSlider {{ min-height: {m['target']}px; }}
QSlider::groove:horizontal {{
    height: {round(m['bar'] / 2)}px; border-radius: {round(m['bar'] / 4)}px;
    background: rgba(128, 128, 128, 0.35);
}}
QSlider::sub-page:horizontal {{ background: #2563eb; border-radius: {round(m['bar'] / 4)}px; }}
QSlider::handle:horizontal {{
    width: {round(m['target'] * 0.7)}px; border-radius: {round(m['target'] * 0.35)}px;
    margin: -{round(m['target'] * 0.33)}px 0;
    background: palette(button); border: 2px solid #2563eb;
}}
QSlider::handle:horizontal:pressed {{ background: #2563eb; }}

QPlainTextEdit, QTextEdit {{ padding: {m['pad']}px; border-radius: {m['pad']}px; font-size: {m['output']}px; }}

QGroupBox {{
    margin-top: {m['gap'] * 2}px; padding: {m['gap']}px; border-radius: {m['pad']}px;
    border: 1px solid rgba(128, 128, 128, 0.35);
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: {m['gap']}px; padding: 0 {m['pad']}px;
    font-size: {m['heading']}px; font-weight: 700;
}}

QMenuBar {{ font-size: {m['text']}px; min-height: {m['target']}px; }}
QMenuBar::item {{ padding: {round((m['target'] - m['text']) / 2)}px {m['gap'] * 2}px; }}
QMenu::item {{ min-height: {m['target']}px; padding: 0 {m['gap'] * 2}px; }}

/* Geometry only, and a handle tinted with grey that reads on either theme,
   so a styled scroll bar does not lose the platform's own colours. */
QScrollBar:vertical {{ width: {m['bar']}px; background: transparent; margin: 0; }}
QScrollBar:horizontal {{ height: {m['bar']}px; background: transparent; margin: 0; }}
QScrollBar::handle {{
    background: rgba(128, 128, 128, 0.6); border-radius: {round(m['bar'] / 2)}px;
}}
QScrollBar::handle:hover {{ background: rgba(128, 128, 128, 0.85); }}
QScrollBar::handle:vertical {{ min-height: {m['grip']}px; }}
QScrollBar::handle:horizontal {{ min-width: {m['grip']}px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* Chat bubbles. The two sides are told apart by the edge they sit on and by
   weight, never by hue: the one colour in this window is the primary action,
   and a transcript that competes with it makes that button harder to find
   rather than easier. Grey at two alphas reads on a light window and a dark
   one alike, and the corner nearest the speaker is squared off — the tail,
   without drawing one. */
QFrame#bubbleYou, QFrame#bubbleThem {{
    border-radius: {m['bubble']}px;
    border: 1px solid rgba(128, 128, 128, 0.30);
}}
QFrame#bubbleYou {{
    background: rgba(128, 128, 128, 0.26);
    border-bottom-right-radius: {round(m['bubble'] / 4)}px;
}}
QFrame#bubbleThem {{
    background: rgba(128, 128, 128, 0.11);
    border-bottom-left-radius: {round(m['bubble'] / 4)}px;
}}
/* The label inside a bubble draws no background of its own, or it paints a
   rectangle over the rounded corners it sits in. */
QFrame#bubbleYou QLabel, QFrame#bubbleThem QLabel {{ background: transparent; }}

QSplitter::handle {{ background: rgba(128, 128, 128, 0.35); }}
QSplitter::handle:horizontal {{ width: {m['pad']}px; }}
QSplitter::handle:vertical {{ height: {m['pad']}px; }}
"""


def flickable(widget: QAbstractScrollArea) -> QAbstractScrollArea:
    """Let a drag inside ``widget`` scroll it, the way a touch screen expects.

    The gesture is grabbed on the viewport rather than the widget so the scroll
    bars stay draggable, and it is the touch gesture rather than the mouse one
    so selecting text with a mouse still selects text.
    """
    viewport = widget.viewport()
    viewport.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
    QScroller.grabGesture(viewport, QScroller.ScrollerGestureType.TouchGesture)

    scroller = QScroller.scroller(viewport)
    properties = QScrollerProperties(scroller.scrollerProperties())
    metric = QScrollerProperties.ScrollMetric
    off = QScrollerProperties.OvershootPolicy.OvershootAlwaysOff
    for name, value in ((metric.DragStartDistance, 0.002),
                        (metric.DecelerationFactor, 0.15),
                        (metric.MaximumVelocity, 1.4),
                        (metric.VerticalOvershootPolicy, off),
                        (metric.HorizontalOvershootPolicy, off)):
        properties.setScrollMetric(name, value)
    scroller.setScrollerProperties(properties)

    # Pixel scrolling, where the widget has the notion: a flick that moves by
    # whole rows reads as a stutter and stops on a row boundary rather than
    # where the finger left off. Only item views — the combo popups — have it.
    if isinstance(widget, QAbstractItemView):
        widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        widget.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    return widget


def labelled(caption: str, widget: QWidget) -> QWidget:
    """A control with its caption above it rather than beside it.

    Above, because that leaves the control the whole width of its column, and
    the width of a control is half of what makes it hittable.
    """
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(2)
    label = QLabel(caption)
    label.setObjectName("fieldLabel")
    label.setBuddy(widget)
    column.addWidget(label)
    column.addWidget(widget)
    return holder


def stepper(box: QAbstractSpinBox) -> QWidget:
    """A spin box with its arrows replaced by a − and a + either side of it.

    Qt stacks its two arrows inside the field, so each is half a control tall
    and a few pixels wide — precisely the target a finger cannot hit. These are
    full height, and they repeat while held, which is how a value gets from 12
    to 30 without thirty taps.
    """
    box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(round(TARGET / 8))
    row.addWidget(_step_button("−", box.stepDown))
    row.addWidget(box, 1)
    row.addWidget(_step_button("+", box.stepUp))
    return holder


def _step_button(text: str, step) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("stepper")
    button.setAutoRepeat(True)
    button.setAutoRepeatDelay(400)
    button.setAutoRepeatInterval(90)
    button.clicked.connect(step)
    return button


class TouchSlider(QSlider):
    """A slider that goes where it is tapped.

    Qt's default is to page the handle one step toward the tap, which needs
    several taps to cross the track and a precise drag to land on a value. It
    also matters inside a flickable area: a tap is unambiguous where a drag has
    to be told apart from a scroll.
    """

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            span = self.width() if self.orientation() == Qt.Orientation.Horizontal else self.height()
            along = event.position().x() if self.orientation() == Qt.Orientation.Horizontal \
                else span - event.position().y()
            self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
                                                         int(along), span))
        super().mousePressEvent(event)


def touchable_popup(box: QComboBox) -> QComboBox:
    """A drop-down whose rows are finger-sized and whose list flicks.

    Both halves are needed: the accent list is 47 rows long, and the default
    popup delegate ignores the row height a style sheet asks for.
    """
    box.setItemDelegate(QStyledItemDelegate(box))
    flickable(box.view())
    return box


def load_scale(paths, default: str = DEFAULT_SCALE) -> str:
    """The display size chosen last time, if it is still one we offer."""
    try:
        name = read_json(paths.data / SETTINGS_FILE).get("display_size")
    except (OSError, ValueError):
        return default
    return name if name in SCALES else default


def save_scale(paths, name: str) -> None:
    if name not in SCALES:
        raise ValueError(f"Unknown display size {name}")
    settings = Path(paths.data) / SETTINGS_FILE
    try:
        current = read_json(settings)
    except (OSError, ValueError):
        current = {}
    atomic_write_json(settings, {**current, "display_size": name})
