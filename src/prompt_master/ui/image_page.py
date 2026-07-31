"""Image mode: one instant, described for a model that draws it in one pass.

The third page, beside prompt mode and conversation mode. It shares the window,
the display size and the one llama-server, and — like conversation mode — it
shares nothing else. In particular it never reaches ``prompt_engine``: a video
prompt describes change over time and an image prompt describes an instant, and
the two engines have no code in common. That separation is not tidiness, it is
what keeps the vendored LTX engine inside the surface ``PARITY_REPORT.md``
describes, and ``tests/test_image_engine.py`` fails the build if an import ever
crosses it.

Five things about this page are worth stating, because they are what image mode
adds that the other two pages have no equivalent of.

*Two tasks, one page.* Generate writes a text-to-image prompt; Edit writes an
instruction for changing an image that already exists. The task selector is the
first control on the page because it changes what almost every control under it
means, and the two control sets are shown and hidden rather than crammed
together.

*Three sentinels above every drop-down.* Auto omits the facet entirely, Choose
for me asks the model to pick the best fit for the intent, and Random draws
from the bank against the seed. Auto and Random differ in whether anything is
stated at all; Choose and Random differ in whether the intent is consulted.

*What Choose chose is shown.* Under every drop-down set to Choose or Random is
a line naming the value that was resolved and where it came from — the model,
the local keyword fallback, or a seeded draw. A choice nobody can see is no
better than Auto, and it is also the thing to copy into the box next time.

*The page follows the model.* The two supported families disagree, and the
disagreements are not cosmetic: only one of the three profiles has any
negative-conditioning path, only one edits natively, only one binds hex colour
and only one parses a JSON prompt. Controls a model would silently discard are
hidden rather than disabled, because a populated field that goes nowhere is
worse than no field.

*Every list comes from the engine.* ``option_banks()`` is the only source of
options here, exactly as ``prompt_engine/options.py`` is for prompt mode. A
control holding its own copy of a list is a control that will eventually differ
from the engine consuming its value.

*It is built the first time it is opened.* Nineteen drop-downs with their
sentinels, eleven preservation targets and four reference slots come to some
five hundred widgets, and setting a style sheet re-polishes every widget in the
application — so a session that only ever writes video prompts should not pay
for this page at all. ``MainWindow.image`` is what defers it, and nothing in
this module needs to know.
"""

from __future__ import annotations

import threading
from dataclasses import fields
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
                               QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget)

from prompt_master.core.config import atomic_write_json, read_json
from prompt_master.core.models import RANDOM_SEED
from prompt_master.image_engine import (AUTO, PROFILES, EditControls, EditGeneration,
                                        ImageControls, ImagePromptEngine, PromptShape,
                                        ReferenceImage, build_edit_brief, editing_profiles,
                                        is_set, option_banks, profile_for, resolve_dimensions)
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.inference.image_client import ImageChatClient
from prompt_master.ui import touch

# Where the page remembers the model and the task. Beside the display size and
# the chat view's own file, for the same reason: opening the app on the model
# you were using last is the whole of what remembering has to mean here.
STATE_FILE = "image-ui.json"

GENERATE, EDIT = "generate", "edit"
TASKS = ((GENERATE, "Generate — text to image"),
         (EDIT, "Edit — change an image that exists"))

# The one group of generation controls whose visibility is not simply "is this
# the task showing" — see ``react``.
NEGATIVE_GROUP = "Negative prompt"

# The reference roles the *generation* system prompt has guidance for. Not the
# same thing as the edit bank in ``option_banks()``: those roles address an
# attached image by number and role inside the instruction, while these three
# choose which paragraph the writer is given about a still it can see. Named
# here because the engine exposes them as prose rather than as a bank, and held
# honest by ``test_image_mode.py``, which asserts each one changes the system
# prompt it produces.
GENERATE_REFERENCE_ROLES = (("style", "Style — the treatment"),
                            ("subject", "Subject — who or what it is"),
                            ("composition", "Composition — the layout"))

# The most reference slots any profile accepts. Built rather than typed, so a
# fourth profile with a different limit needs no edit here.
MAX_REFERENCES = max(profile.max_reference_images for profile in PROFILES.values())

# Which controls carry the three sentinels, per task. Both are the engine's own
# answer — the fields of its controls dataclass that appear in its selectable
# banks — rather than a list this page keeps.
_SELECTABLE = set(option_banks()["selectable"])
GENERATE_CHOICES = tuple(f.name for f in fields(ImageControls) if f.name in _SELECTABLE)
EDIT_CHOICES = tuple(f.name for f in fields(EditControls) if f.name in _SELECTABLE)


class ImageWorker(QObject):
    """One prompt or one instruction, written off the UI thread.

    The engine pulls fragments through :class:`ImageChatClient`, so what this
    emits as ``chunk`` is the writer pass and only the writer pass. The
    Choose-for-me call and the smart-negative call go out over the same client
    and never reach the pane.
    """

    chunk = Signal(str)
    done = Signal(object)                 # Generation | EditGeneration
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service, task, text, controls, image, needs_vision):
        super().__init__()
        self.service, self.task, self.text, self.controls = service, task, text, controls
        self.image, self.needs_vision = image, needs_vision
        self.cancelled = threading.Event()

    @Slot()
    def cancel(self) -> None:
        self.cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("Starting llama-server…")
            # Raises when the still needs a projector this install has not got,
            # which is the same guard prompt mode reaches through.
            client = self.service.client(self.needs_vision)
            engine = ImagePromptEngine(ImageChatClient(client, self.cancelled))
            editing = self.task == EDIT
            self.status.emit("Resolving settings…")
            stream = (engine.stream_edit if editing else engine.stream)(
                self.text, self.controls, image_jpeg_b64=self.image)
            writing = False
            for chunk in stream:
                if self.cancelled.is_set():
                    self.status.emit("Generation cancelled")
                    return
                if not writing:
                    # Said on the first fragment rather than before the loop:
                    # everything up to here was the selection pass, which can
                    # take a moment of its own and is worth naming separately.
                    self.status.emit("Writing…")
                    writing = True
                self.chunk.emit(chunk)
            if self.cancelled.is_set():
                self.status.emit("Generation cancelled")
                return
            self.done.emit(engine.finish_edit() if editing else engine.finish())
        except Exception as exc:                       # surfaced, never swallowed
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ReferenceSlot(QWidget):
    """One attached image and what the prompt should take from it.

    The role is beside the image rather than implied by its position, because
    that is what lets an instruction say "the background from image 2" — the
    number alone tells the model nothing about what to take, and the role alone
    may not resolve to a particular attachment.
    """

    def __init__(self, caption: str, roles, parent=None):
        super().__init__(parent)
        self.path: Path | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(f"{caption} — none")
        self.name.setObjectName("fieldLabel")
        self.name.setWordWrap(True)
        self.caption = caption
        self.role = touch.touchable_popup(QComboBox())
        for value, label in roles:
            self.role.addItem(label, value)
        attach = QPushButton("Attach…")
        attach.clicked.connect(self.browse)
        self.drop = QPushButton("Remove")
        self.drop.setEnabled(False)
        self.drop.clicked.connect(self.clear)
        row.addWidget(self.name, 1)
        row.addWidget(self.role, 1)
        row.addWidget(attach)
        row.addWidget(self.drop)

    def browse(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Reference image", "",
                                                  "Images (*.png *.jpg *.jpeg *.webp)")
        if not filename:
            return
        self.path = Path(filename)
        # The name, not the path: a full path pushes the buttons beside it off
        # the pane, and the path is still there to hover or long-press for.
        self.name.setText(f"{self.caption} — {self.path.name}")
        self.name.setToolTip(filename)
        self.drop.setEnabled(True)

    def clear(self) -> None:
        self.path = None
        self.name.setText(f"{self.caption} — none")
        self.name.setToolTip("")
        self.drop.setEnabled(False)

    def chosen_role(self) -> str:
        value = self.role.currentData()
        return "" if value is None else value


class ImagePage(QWidget):
    """The whole of image mode: two panes, two tasks, and the bar under them."""

    def __init__(self, paths, service_provider, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.service_provider = service_provider
        self.banks = option_banks()
        # Defaults are read off the engine's own controls objects rather than
        # written out again here, so a default that changes there changes here.
        self.defaults = ImageControls()
        self.edit_defaults = EditControls()
        self.metrics = touch.metrics(1.0)
        self.thread: QThread | None = None
        self._worker: ImageWorker | None = None
        self.choices: dict[str, QComboBox] = {}
        self.resolved: dict[str, QLabel] = {}
        self.preservation: dict[str, QCheckBox] = {}

        column = QVBoxLayout(self)
        column.addLayout(self.task_row())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.compose_pane())
        splitter.addWidget(self.output_pane())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        column.addWidget(splitter, 1)
        column.addLayout(self.action_bar())

        self.wire()
        self.restore()
        self.react()

    # ── layout ───────────────────────────────────────────────────────────────

    def task_row(self) -> QHBoxLayout:
        """What is being written, and what it is being written for.

        Both at the top and both above everything else, because between them
        they decide which controls below are shown at all.
        """
        row = QHBoxLayout()
        self.task = touch.touchable_popup(QComboBox())
        for value, label in TASKS:
            self.task.addItem(label, value)
        self.model = touch.touchable_popup(QComboBox())
        for key, profile in PROFILES.items():
            self.model.addItem(profile.display_name, key)
        index = self.model.findData(self.defaults.profile_key)
        if index >= 0:
            self.model.setCurrentIndex(index)
        row.addWidget(touch.labelled("Task", self.task), 1)
        row.addWidget(touch.labelled("Image model", self.model), 1)
        return row

    def compose_pane(self) -> QWidget:
        """Intent, seed and references stay put; the settings under them scroll."""
        pane = QWidget()
        column = QVBoxLayout(pane)
        self.heading_intent = self.heading("Intent")
        column.addWidget(self.heading_intent)
        self.intent = QPlainTextEdit()
        self.intent.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        touch.flickable(self.intent)
        column.addWidget(self.intent)
        column.addLayout(self.seed_row())
        # Said where it is acted on: the requirements for a model that cannot
        # really edit belong beside the controls that would send it an edit.
        self.requirements = QLabel("")
        self.requirements.setObjectName("fieldLabel")
        self.requirements.setWordWrap(True)
        self.requirements.hide()
        column.addWidget(self.requirements)
        column.addWidget(self.reference_area())
        column.addWidget(self.settings_area(), 1)
        return pane

    def seed_row(self) -> QHBoxLayout:
        """The seed, and what the model this is written for will want set.

        The numbers beside it — steps, guidance, the frame, the denoise an edit
        resolves to — are not the writer's business at all; they are what has to
        be typed into the image model afterwards, and they change with the
        profile, so they are shown rather than left to be looked up.
        """
        row = QHBoxLayout()
        self.seed = QSpinBox()
        self.seed.setRange(RANDOM_SEED, 2 ** 31 - 1)
        self.seed.setValue(self.defaults.seed)
        self.seed.setSpecialValueText("Random each time (-1)")
        self.seed.setToolTip("Seeds the random draws, the Choose-for-me shuffle and the "
                             "image model's own sampling. A reported seed reproduces all three.")
        self.facts = QLabel("")
        self.facts.setObjectName("fieldLabel")
        self.facts.setWordWrap(True)
        row.addWidget(touch.labelled("Seed", touch.stepper(self.seed)), 1)
        row.addWidget(self.facts, 2)
        return row

    def reference_area(self) -> QWidget:
        """One attachment for generation, up to four for an edit.

        Generation shows the writer a still and asks it to describe what it can
        see; editing addresses the attachments from inside the instruction. The
        slots beyond what the chosen model accepts are hidden, so the prompt can
        never promise more references than the model will read.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        self.reference = ReferenceSlot("Reference image", GENERATE_REFERENCE_ROLES)
        column.addWidget(self.reference)
        roles = [(role, role) for role in self.banks["reference_roles"]]
        self.slots = [ReferenceSlot(f"Image {number}", roles)
                      for number in range(1, MAX_REFERENCES + 1)]
        for slot in self.slots:
            column.addWidget(slot)
        return holder

    def settings_area(self) -> QScrollArea:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        generate = {title: self.section(title, group)
                    for title, group in self.generate_controls()}
        self.generate_boxes = list(generate.values())
        self.edit_boxes = [self.section(title, group)
                           for title, group in self.edit_controls()]
        # Held by name rather than by position: it is the one group that is
        # hidden for a reason other than which task is showing.
        self.negative_box = generate[NEGATIVE_GROUP]
        for box in self.generate_boxes + self.edit_boxes:
            column.addWidget(box)
        column.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(page)
        area.setFrameShape(QFrame.Shape.NoFrame)
        touch.flickable(area)
        return area

    # ── the controls ─────────────────────────────────────────────────────────

    def generate_controls(self):
        """Every generation control, grouped by what it changes about the image."""
        b = self.banks
        self.hex_swatches = QCheckBox("Bind the palette to hex values")
        self.hex_swatches.setChecked(self.defaults.use_hex_swatches)
        self.hex_swatches.setToolTip("Only FLUX.2 binds a hex value to a named object.")
        self.concrete_identity = QCheckBox("Cast a specific person, not a type")
        self.concrete_identity.setChecked(self.defaults.concrete_identity)
        self.skin_texture = QCheckBox("Name skin texture")
        self.skin_texture.setChecked(self.defaults.skin_texture)
        self.render_text = QLineEdit()
        self.render_text.setPlaceholderText("Words to appear in the image")
        self.text_surface = QLineEdit()
        self.text_surface.setPlaceholderText("What they are printed on")
        self.long_edge = QSpinBox()
        self.long_edge.setRange(0, max(p.max_edge for p in PROFILES.values()))
        self.long_edge.setSingleStep(64)
        self.long_edge.setValue(self.defaults.long_edge)
        self.long_edge.setSpecialValueText("The model's own size")
        self.shape = touch.touchable_popup(QComboBox())
        # The value rather than the member: ``PromptShape`` is a str enum, and
        # Qt stores a str subclass as the plain str it also is. The engine tests
        # the member by identity, so a shape that came back through item data
        # would compare equal to JSON and still be written as prose.
        self.shape.addItem("A paragraph of prose", PromptShape.PROSE.value)
        # Held as its own widget so the whole field can go when the model gains
        # nothing from a JSON prompt: a drop-down with one item in it is a
        # control that looks like a choice and is not.
        self.shape_field = touch.labelled("Prompt shape", self.shape)
        self.word_target = QSpinBox()
        self.word_target.setRange(0, max(p.hard_word_ceiling for p in PROFILES.values()))
        self.word_target.setSingleStep(5)
        self.word_target.setValue(self.defaults.word_target)
        self.word_target.setSpecialValueText("The model's own target")
        self.smart_negative = QCheckBox("Smart negative — a second pass over the finished prompt")
        self.smart_negative.setChecked(self.defaults.smart_negative)
        self.extra_negative = QPlainTextEdit()
        self.extra_negative.setPlaceholderText("Extra terms to keep out of the image, comma separated.")
        touch.flickable(self.extra_negative)

        return [
            ("Framing and optics", [("Shot", self.choice("shot_type", b["shot_types"])),
                                    ("Angle", self.choice("camera_angle", b["camera_angles"])),
                                    ("Composition", self.choice("composition", b["composition"])),
                                    ("Lens", self.choice("lens", b["lenses"])),
                                    ("Aperture", self.choice("aperture", b["apertures"]))]),
            ("Light and air", [("Lighting", self.choice("lighting", b["lighting"])),
                               ("Time of day", self.choice("time_of_day", b["time_of_day"])),
                               ("Weather", self.choice("weather", b["weather"]))]),
            ("Look", [("Medium", self.choice("visual_style", groups=b["style_groups"])),
                      ("Film stock", self.choice("film_stock", b["film_stocks"])),
                      ("Palette", self.choice("colour_palette", b["colour_palettes"])),
                      (None, self.hex_swatches),
                      ("Material", self.choice("material_focus", b["materials"])),
                      ("Mood", self.choice("mood", b["moods"])),
                      ("Detail", self.choice("render_detail", b["render_detail"]))]),
            ("Figures", [("Wardrobe", self.choice("wardrobe", b["wardrobe"])),
                         (None, self.concrete_identity), (None, self.skin_texture)]),
            ("Text in the image", [("Words", self.render_text), ("Printed on", self.text_surface)]),
            ("Frame", [("Aspect ratio", self.choice("aspect_ratio", b["aspect_ratios"])),
                       ("Long edge", touch.stepper(self.long_edge))]),
            ("Wording", [(None, self.shape_field),
                         ("Word target", touch.stepper(self.word_target))]),
            (NEGATIVE_GROUP, [(None, self.smart_negative),
                              ("Extra negative terms", self.extra_negative)]),
        ]

    def edit_controls(self):
        """Every editing control. Three of them carry most of the quality."""
        b = self.banks
        self.target_element = QLineEdit()
        self.target_element.setPlaceholderText("Which element — \"the wooden door on the left\"")
        self.target_element.setToolTip("These models have no mask, so this description does the "
                                       "work a mask would. Left empty, the writer infers it.")
        self.match_lighting = QCheckBox("Match the scene's light and shadows")
        self.match_lighting.setChecked(self.edit_defaults.match_lighting)
        self.match_perspective = QCheckBox("Match the scene's scale and perspective")
        self.match_perspective.setChecked(self.edit_defaults.match_perspective)
        self.seamless_blend = QCheckBox("Integrate the edges — no cut-out look")
        self.seamless_blend.setChecked(self.edit_defaults.seamless_blend)
        self.iterative_hint = QCheckBox("Warn when the request is really several edits")
        self.iterative_hint.setChecked(self.edit_defaults.iterative_hint)
        self.allow_experimental = QCheckBox("Allow the experimental editing path")
        self.allow_experimental.setChecked(self.edit_defaults.allow_experimental_edit)
        self.edit_word_target = QSpinBox()
        self.edit_word_target.setRange(0, max(p.edit_max_words for p in PROFILES.values()))
        self.edit_word_target.setSingleStep(5)
        self.edit_word_target.setValue(self.edit_defaults.word_target)
        self.edit_word_target.setSpecialValueText("The model's own target")

        keep = QWidget()
        grid = QGridLayout(keep)
        grid.setContentsMargins(0, 0, 0, 0)
        for position, target in enumerate(b["preservation_targets"]):
            box = QCheckBox(target)
            box.setChecked(target in self.edit_defaults.preservation)
            self.preservation[target] = box
            grid.addWidget(box, position // 2, position % 2)

        return [
            ("The change", [("Operation", self.choice("operation", b["edit_operations"])),
                            ("Target element", self.target_element),
                            ("Where", self.choice("localization", b["localizations"])),
                            ("Strength", self.choice("edit_strength", b["edit_strengths"])),
                            ("Word target", touch.stepper(self.edit_word_target))]),
            ("What must survive", [(None, keep)]),
            ("Physical consistency", [(None, self.match_lighting), (None, self.match_perspective),
                                      (None, self.seamless_blend), (None, self.iterative_hint)]),
            ("Model support", [(None, self.allow_experimental)]),
        ]

    def choice(self, name: str, options=None, groups=None) -> QWidget:
        """One drop-down, its three sentinels, and the line under it.

        The sentinels come first and the bank follows, separated, because the
        three of them are answers to a different question than any option is.
        The line under the box is empty until a generation resolves the control,
        and then says what was picked and which of the three ways picked it.
        """
        box = touch.touchable_popup(QComboBox())
        for sentinel in self.banks["sentinels"].values():
            box.addItem(sentinel, sentinel)
        box.insertSeparator(box.count())
        if groups:
            for group, members in groups.items():
                box.addItem(f"— {group} —", None)
                item = box.model().item(box.count() - 1)
                if item is not None:
                    item.setEnabled(False)
                for value in members:
                    box.addItem(f"   {value}", value)
        else:
            for value in options or ():
                box.addItem(value, value)
        default = getattr(self.defaults, name, None)
        if default is None:
            default = getattr(self.edit_defaults, name, AUTO)
        index = box.findData(default)
        if index >= 0:
            box.setCurrentIndex(index)
        note = QLabel("")
        note.setObjectName("fieldLabel")
        note.setWordWrap(True)
        note.hide()
        self.choices[name] = box
        self.resolved[name] = note
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(box)
        column.addWidget(note)
        return holder

    def section(self, title, fields) -> QGroupBox:
        """One group, two columns wide, laid out as prompt mode lays its own out:
        a control with no caption of its own takes the full width."""
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        row = column = 0
        for caption, widget in fields:
            span = 2 if caption is None or isinstance(widget, QPlainTextEdit) else 1
            if span == 2 and column:
                row += 1
                column = 0
            grid.addWidget(widget if caption is None else touch.labelled(caption, widget),
                           row, column, 1, span)
            column += span
            if column >= 2:
                row += 1
                column = 0
        return box

    def output_pane(self) -> QWidget:
        """What was written, what it cost, and what the engine wants to say about it."""
        pane = QWidget()
        column = QVBoxLayout(pane)
        self.prompt_heading = self.heading("Prompt")
        self.prompt_out = QTextEdit()
        self.negative_heading = self.heading("Negative prompt")
        self.negative_out = QTextEdit()
        self.notes_out = QTextEdit()
        self.notes_out.setReadOnly(True)
        for edit in (self.prompt_out, self.negative_out, self.notes_out):
            edit.setAcceptRichText(False)
            touch.flickable(edit)
        column.addLayout(self.output_header(self.prompt_heading, self.prompt_out))
        column.addWidget(self.prompt_out, 4)
        self.negative_header = self.output_header(self.negative_heading, self.negative_out)
        column.addLayout(self.negative_header)
        column.addWidget(self.negative_out, 2)
        column.addWidget(self.heading("Notes"))
        column.addWidget(self.notes_out, 2)
        row = QHBoxLayout()
        both = QPushButton("Copy both")
        both.clicked.connect(self.copy_both)
        save = QPushButton("Save .txt…")
        save.clicked.connect(self.save)
        row.addWidget(both, 1)
        row.addWidget(save, 1)
        column.addLayout(row)
        return pane

    def output_header(self, heading: QLabel, source: QTextEdit) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(heading)
        row.addStretch(1)
        copy = QPushButton("Copy")
        copy.clicked.connect(lambda _=False, s=source: self.copy(s))
        row.addWidget(copy)
        source.copy_button = copy                      # so the pane can hide as one
        return row

    def action_bar(self) -> QHBoxLayout:
        """Always on screen: what the page is doing, and the button pressed most."""
        row = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self.generate)
        row.addWidget(self.status, 1)
        row.addWidget(clear)
        row.addWidget(self.cancel_button)
        row.addWidget(self.generate_button)
        return row

    @staticmethod
    def heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    # ── following the model and the task ─────────────────────────────────────

    def wire(self) -> None:
        """Connected after everything exists, because every one of these reads
        the whole page rather than the control that changed."""
        self.task.currentIndexChanged.connect(self.task_changed)
        self.model.currentIndexChanged.connect(self.model_changed)
        self.allow_experimental.toggled.connect(self.react)
        for control in (self.choices["aspect_ratio"], self.choices["edit_strength"]):
            control.currentIndexChanged.connect(self.describe_model)
        self.long_edge.valueChanged.connect(self.describe_model)

    def current_task(self) -> str:
        value = self.task.currentData()
        return value if value in (GENERATE, EDIT) else GENERATE

    def profile(self):
        return profile_for(self.model.currentData() or self.defaults.profile_key)

    def negative_wanted(self) -> bool:
        """Whether a negative prompt is a real thing here.

        One profile of three has classifier-free guidance, and no editing path
        on any of them takes a negative at all. Asked as a question rather than
        read off a widget's visibility, because the answer is needed before the
        window has ever been shown.
        """
        return self.profile().negative_supported and self.current_task() != EDIT

    def task_changed(self, _index: int = 0) -> None:
        self.react()
        self.remember(task=self.current_task())

    def model_changed(self, _index: int = 0) -> None:
        self.react()
        self.remember(profile=self.model.currentData())

    def react(self, *_args) -> None:
        """Everything on the page that follows the model and the task.

        Hidden rather than disabled, throughout. A greyed-out negative field
        still says the model has one, and the two distilled checkpoints do not:
        a populated field they discard would give a false sense of control,
        which is worse than the field not being there.
        """
        editing = self.current_task() == EDIT
        self.allow_editing_profiles(editing)
        profile = self.profile()

        for box in self.generate_boxes:
            box.setVisible(not editing)
        for box in self.edit_boxes:
            box.setVisible(editing)

        # 1. The negative prompt exists on one profile of three, and on no
        #    editing path at all — no supported editor takes one.
        negative = self.negative_wanted()
        self.negative_box.setVisible(negative)
        self.negative_heading.setVisible(negative)
        self.negative_out.setVisible(negative)
        self.negative_out.copy_button.setVisible(negative)

        # 2. Hex swatches bind to a named object on FLUX.2 and are ignored
        #    elsewhere. 3. So is a JSON prompt.
        self.hex_swatches.setVisible(profile.supports_hex_colour)
        self.shape_field.setVisible(profile.supports_json_prompt)
        self.offer_json(profile.supports_json_prompt)

        # 4. Editing, and what the experimental path costs to reach.
        experimental = editing and profile.edit_is_experimental
        self.requirements.setVisible(experimental)
        if experimental:
            self.requirements.setText(f"{profile.display_name}: {profile.edit_requirements}")

        # 5. Never offer a reference slot the model will not read. A slot the
        #    chosen model cannot take is emptied as well as hidden, because an
        #    attachment nothing will ever send is worse than no attachment;
        #    slots hidden only because generation is showing keep what is in
        #    them, so switching back to Edit finds them still attached.
        self.reference.setVisible(not editing)
        for number, slot in enumerate(self.slots, start=1):
            within = number <= profile.max_reference_images
            slot.setVisible(editing and within)
            if not within:
                slot.clear()

        self.heading_intent.setText("Edit request" if editing else "Intent")
        self.intent.setPlaceholderText(
            "Describe the change — \"swap the wooden door for a steel one\"…" if editing
            else "Describe the image you want to create…")
        self.prompt_heading.setText("Edit instruction" if editing else "Prompt")
        self.generate_button.setText("Write instruction" if editing else "Generate")
        self.describe_model()

    def allow_editing_profiles(self, editing: bool) -> None:
        """Grey out the models that cannot do the task, and leave one that can.

        ``editing_profiles()`` is the engine's own answer to which models may be
        offered, including what turning the experimental option on changes about
        it, so the rows follow it rather than a rule written again here.
        """
        offered = (editing_profiles(self.allow_experimental.isChecked()) if editing
                   else list(PROFILES.values()))
        allowed = {profile.key for profile in offered}
        for position in range(self.model.count()):
            item = self.model.model().item(position)
            if item is not None:
                item.setEnabled(self.model.itemData(position) in allowed)
        if offered and self.model.currentData() not in allowed:
            # Moved rather than left sitting on a model that will refuse: the
            # engine raises for this combination, and it raises after the user
            # has waited for a server to start.
            replacement = offered[0]
            self.model.blockSignals(True)
            self.model.setCurrentIndex(max(0, self.model.findData(replacement.key)))
            self.model.blockSignals(False)
            self.set_status(f"{replacement.display_name} is the model that edits natively; "
                            "switched to it.")

    def offer_json(self, supported: bool) -> None:
        """Offer the JSON shape only where it is honoured.

        The engine downgrades JSON to prose on a model that gains nothing from
        it, silently and correctly — but offering a choice that will not be
        kept is a different thing from making it quietly.
        """
        index = self.shape.findData(PromptShape.JSON.value)
        if supported and index < 0:
            self.shape.addItem("A JSON object", PromptShape.JSON.value)
        elif not supported and index >= 0:
            self.shape.removeItem(index)

    def describe_model(self, *_args) -> None:
        """The numbers the image model itself has to be set to.

        None of these are the writer's business; all of them change with the
        profile, and an edit's denoise changes with the strength as well. They
        are asked of the engine rather than worked out here — the denoise map
        lives in ``build_edit_brief`` and is not worth a second copy.
        """
        profile = self.profile()
        if self.current_task() == EDIT:
            bits = [profile.display_name, f"{profile.edit_steps} steps",
                    f"CFG {profile.edit_cfg:g}", f"denoise {self.denoise():g}",
                    f"{profile.edit_min_words}–{profile.edit_max_words} words",
                    f"up to {profile.max_reference_images} references"]
        else:
            aspect = self.chosen("aspect_ratio")
            width, height = resolve_dimensions(aspect, self.long_edge.value(), profile)
            frame = f"{width}×{height}"
            if not is_set(aspect):
                # The ratio is still a sentinel, so this is the fallback the
                # engine would use rather than the frame you will get. Saying
                # so beats printing a number that is about to change.
                frame += " until the ratio is settled"
            bits = [profile.display_name, frame,
                    f"{profile.default_steps} steps", f"CFG {profile.default_cfg:g}",
                    f"{profile.min_words}–{profile.max_words} words"]
            if not profile.negative_supported:
                bits.append("no negative")
        self.facts.setText(" · ".join(bits))

    def denoise(self) -> float:
        """What the strength control resolves to, from the engine's own map."""
        try:
            return build_edit_brief("preview", self.edit_settings(), 0).denoise
        except ValueError:
            # The experimental path without the opt-in. The strength still maps
            # the same way; there is simply nothing to run it on yet.
            return self.profile().edit_denoise_default

    # ── generating ───────────────────────────────────────────────────────────

    def chosen(self, name: str) -> str:
        """A drop-down's value, falling back to the engine's default for it."""
        box = self.choices.get(name)
        value = box.currentData() if box is not None else None
        if value is None:
            value = getattr(self.defaults, name, None)
        if value is None:
            value = getattr(self.edit_defaults, name, AUTO)
        return value

    def picked(self, names) -> dict[str, str]:
        return {name: self.chosen(name) for name in names}

    def image_settings(self) -> ImageControls:
        return ImageControls(
            profile_key=self.model.currentData() or self.defaults.profile_key,
            **self.picked(GENERATE_CHOICES),
            use_hex_swatches=self.hex_swatches.isChecked(),
            concrete_identity=self.concrete_identity.isChecked(),
            skin_texture=self.skin_texture.isChecked(),
            render_text=self.render_text.text(),
            text_surface=self.text_surface.text(),
            long_edge=self.long_edge.value(),
            prompt_shape=PromptShape(self.shape.currentData() or PromptShape.PROSE.value),
            word_target=self.word_target.value(),
            extra_negative=self.extra_negative.toPlainText(),
            smart_negative=self.smart_negative.isChecked(),
            has_reference_image=self.reference.path is not None,
            reference_role=self.reference.chosen_role(),
            seed=self.seed.value(),
        )

    def edit_settings(self) -> EditControls:
        attached = self.attached_slots()
        return EditControls(
            profile_key=self.model.currentData() or self.edit_defaults.profile_key,
            allow_experimental_edit=self.allow_experimental.isChecked(),
            **self.picked(EDIT_CHOICES),
            target_element=self.target_element.text(),
            preservation=tuple(name for name, box in self.preservation.items()
                               if box.isChecked()),
            match_lighting=self.match_lighting.isChecked(),
            match_perspective=self.match_perspective.isChecked(),
            seamless_blend=self.seamless_blend.isChecked(),
            references=tuple(ReferenceImage(index=number, role=slot.chosen_role())
                             for number, slot in enumerate(attached, start=1)),
            word_target=self.edit_word_target.value(),
            iterative_hint=self.iterative_hint.isChecked(),
            seed=self.seed.value(),
        )

    def attached_slots(self) -> list[ReferenceSlot]:
        """The slots this model can read that hold an image, in shown order.

        Order is the whole of it: the engine renumbers the references from one
        as it builds, and the instruction addresses them by that number.

        Bounded by the profile rather than by which slots are on screen —
        ``isVisible`` is false for every widget in a window that has not been
        shown yet, which is not what "the user cannot use this slot" means.
        """
        usable = self.slots[:self.profile().max_reference_images]
        return [slot for slot in usable if slot.path is not None]

    def generate(self) -> None:
        if self.busy():
            return
        editing = self.current_task() == EDIT
        text = self.intent.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Nothing to write from",
                                "Describe the change you want." if editing
                                else "Describe the image you want first.")
            return
        try:
            controls = self.edit_settings() if editing else self.image_settings()
        except Exception as exc:                       # a control in an impossible state
            QMessageBox.critical(self, "Could not read the settings", str(exc))
            return
        source = self.source_image(editing)
        try:
            image = image_data_url(source) if source else None
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Image error", str(exc))
            return
        # Asked here rather than left to the server: a model chosen by hand may
        # have no projector, and that is worth saying while the image is still
        # attached to be removed. Only on an install that finished — an
        # unconfigured one has a better answer of its own, from service.client.
        if image is not None and self.paths.configured and not self.vision_ready():
            QMessageBox.critical(self, "No vision projector",
                                 "The model running has no vision projector, so the attached "
                                 "image cannot be shown to it.\n\nChoose one under Settings → "
                                 "Which model runs, or remove the image.")
            return

        self.prompt_out.clear()
        self.negative_out.clear()
        self.notes_out.clear()
        for note in self.resolved.values():
            note.clear()
            note.hide()
        self.enable(False)
        worker = ImageWorker(self.service_provider(), self.current_task(), text, controls,
                             image, image is not None)
        self.thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.chunk.connect(self.prompt_out.insertPlainText)
        worker.status.connect(self.set_status)
        worker.done.connect(self.completed)
        worker.failed.connect(self.failed)
        worker.finished.connect(self.thread.quit)
        worker.finished.connect(worker.deleteLater)
        self.thread.finished.connect(self.generation_done)
        self.thread.start()

    def source_image(self, editing: bool) -> Path | None:
        """The one still the writing model is shown.

        Editing can carry up to four references, but only one of them can go on
        the wire to a text model with a vision projector, and the first is the
        one the instruction is mostly about.
        """
        if not editing:
            return self.reference.path
        attached = self.attached_slots()
        return attached[0].path if attached else None

    def vision_ready(self) -> bool:
        """Whether a picture can reach the model — false only when it is known to be."""
        try:
            return bool(self.service_provider().vision_ready())
        except Exception:
            return True

    def completed(self, result) -> None:
        """Fill the pane in from whichever of the two results came back."""
        editing = isinstance(result, EditGeneration)
        text = result.instruction if editing else result.positive
        self.prompt_out.setPlainText(text)
        self.negative_out.setPlainText("" if editing else result.negative)
        self.show_choices(result)
        notes = list(getattr(result, "warnings", [])) + list(result.notes)
        self.notes_out.setPlainText("\n\n".join(f"• {note}" for note in notes))
        self.set_status(result.status_line() if text
                        else "The model returned nothing — try again, or say more in the intent.")

    def show_choices(self, result) -> None:
        """Say what Choose for me and Random settled on, and which of them did.

        The point of Choose is that it is visible: the value is what goes in the
        box next time, and the source is what says whether the model actually
        picked it or the keyword fallback did.
        """
        for name, source in result.chosen_source.items():
            note = self.resolved.get(name)
            if note is None:
                continue
            value = result.chosen.get(name)
            note.setText(f"→ {value}  ({source})" if value
                         else "→ nothing fitted; left to the writer")
            note.setToolTip({"model": "The local model picked this for your intent.",
                             "heuristic": "Picked by keyword matching — the model was not "
                                          "available or did not answer usably.",
                             "random": "A seeded draw. The same seed draws it again.",
                             "none": "Nothing in the list fitted, so the facet was left out."}
                            .get(source, source))
            note.show()

    def failed(self, message: str) -> None:
        self.set_status("Generation failed")
        QMessageBox.critical(self, "Generation failed", message)

    def generation_done(self) -> None:
        self.enable(True)
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread, self._worker = None, None

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.set_status("Stopping…")

    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def enable(self, enabled: bool) -> None:
        self.generate_button.setEnabled(enabled)
        self.cancel_button.setEnabled(not enabled)

    # ── the finished text ────────────────────────────────────────────────────

    def copy(self, source: QTextEdit) -> None:
        source.selectAll()
        source.copy()

    def copy_both(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.as_text())

    def save(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save prompt", "image-prompt.txt",
                                                  "Text (*.txt)")
        if filename:
            Path(filename).write_text(self.as_text(), encoding="utf-8")

    def as_text(self) -> str:
        """Both boxes and the settings line, because the settings are half of it.

        A prompt without the steps, the guidance and the seed it was written for
        is a prompt that has to be set up again from memory.
        """
        editing = self.current_task() == EDIT
        parts = [self.status.text(),
                 "\nINSTRUCTION" if editing else "\nPOSITIVE",
                 self.prompt_out.toPlainText()]
        if self.negative_wanted() and self.negative_out.toPlainText().strip():
            parts += ["\nNEGATIVE", self.negative_out.toPlainText()]
        notes = self.notes_out.toPlainText().strip()
        if notes:
            parts += ["\nNOTES", notes]
        return "\n".join(parts) + "\n"

    def clear(self) -> None:
        self.intent.clear()
        self.prompt_out.clear()
        self.negative_out.clear()
        self.notes_out.clear()
        self.reference.clear()
        for slot in self.slots:
            slot.clear()
        for note in self.resolved.values():
            note.clear()
            note.hide()
        self.set_status("")

    # ── plumbing ─────────────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def apply_metrics(self, metrics: dict) -> None:
        """Follow the window's display size."""
        self.metrics = metrics
        self.intent.setMinimumHeight(metrics["target"] * 3)
        self.intent.setMaximumHeight(metrics["target"] * 4)
        self.extra_negative.setMinimumHeight(metrics["target"] * 2)
        self.extra_negative.setMaximumHeight(metrics["target"] * 3)
        for edit in (self.prompt_out, self.negative_out):
            edit.setMinimumHeight(metrics["target"] * 3)
        self.notes_out.setMinimumHeight(metrics["target"] * 2)
        self.notes_out.setMaximumHeight(metrics["target"] * 4)
        self.generate_button.setMinimumWidth(metrics["target"] * 4)

    def rebind(self, paths) -> None:
        """Follow a setup run that moved the installation root."""
        self.paths = paths
        self.restore()
        self.react()

    def restore(self) -> None:
        """Open on the model and the task last used."""
        state = self.state()
        profile = state.get("profile")
        if isinstance(profile, str) and profile in PROFILES:
            index = self.model.findData(profile)
            if index >= 0:
                self.model.blockSignals(True)
                self.model.setCurrentIndex(index)
                self.model.blockSignals(False)
        task = state.get("task")
        if task in (GENERATE, EDIT):
            index = self.task.findData(task)
            if index >= 0:
                self.task.blockSignals(True)
                self.task.setCurrentIndex(index)
                self.task.blockSignals(False)

    def remember(self, profile: str | None = None, task: str | None = None) -> None:
        state = self.state()
        if profile:
            state["profile"] = profile
        if task:
            state["task"] = task
        try:
            atomic_write_json(self.paths.data / STATE_FILE, state)
        except OSError:
            pass                                       # remembering is a convenience

    def state(self) -> dict:
        try:
            return read_json(self.paths.data / STATE_FILE)
        except (OSError, ValueError):
            return {}

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)
