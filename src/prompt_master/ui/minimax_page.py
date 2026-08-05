"""MiniMax H3 mode: a prompt in, an H3 prompt out.

The whole page is one question — what should the H3 prompt say — so it is laid
out as the two panes prompt mode uses and nothing else: what you want on the
left, what the model wrote on the right, and the bar along the bottom that never
scrolls away. Two of the controls between them are there because WanGP has to
know the same two things: which H3 model the prompt is for, and whether there is
a picture to write it around.

The picture is described before it is used, which is a step rather than a
setting: WanGP captions the image with a vision model and hands the enhancer the
paragraph, so this page does the same and shows the paragraph it got. A caption
is what the H3 image instructions are written against, and seeing it is what
explains a prompt that describes the wrong jacket.

The third control is the dialogue slider, and it is this application's own —
``minimax.dialogue`` holds the whole of what it does, and at its left-hand
position, which is where it starts, it does nothing at all. Above that a pass
runs between the caption and the prompt, and what it wrote is shown beside the
caption for the same reason the caption is shown: a prompt in the wrong voice is
a script in the wrong voice, and that is worth being able to read.

Everything else the model is told comes from ``minimax.enhancer``, which is
WanGP's convention around instructions vendored verbatim from its H3 module.
This file chooses nothing about the prompt; it collects the inputs, runs the
calls off the UI thread, and shows what came back.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QPushButton, QSlider, QSplitter,
                               QTextBrowser, QTextEdit, QVBoxLayout, QWidget)

from prompt_master.core.models import draw_seed
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.minimax import dialogue, enhancer
from prompt_master.ui import touch

NO_IMAGE = "No image — text only"


class EnhanceWorker(QObject):
    """One H3 prompt, written off the UI thread.

    Up to three calls, in the order each needs the one before it: the caption
    first, because everything after it is written about the picture; then the
    dialogue pass, which casts the people the caption found; then the enhancer,
    whose user turn is built from both.
    """

    chunk = Signal(str)
    captioned = Signal(str)
    spoken = Signal(str)
    ready = Signal(str)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service, prompt: str, variant: str, image: str | None, seed: int,
                 intensity: int = dialogue.OFF):
        super().__init__()
        self.service, self.prompt, self.variant = service, prompt, variant
        self.image, self.seed = image, seed
        self.intensity = dialogue.clamp(intensity)
        self.cancelled = threading.Event()

    @Slot()
    def cancel(self) -> None:
        self.cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("Starting llama-server…")
            # Raises when the still cannot go on the wire, which is the answer
            # to give rather than a prompt written about a picture nobody saw.
            client = self.service.client(self.image is not None)
            caption = None
            if self.image is not None:
                self.status.emit("Describing the image…")
                described = client.stream_chat(
                    enhancer.caption_messages(self.image), enhancer.CAPTION_MAX_TOKENS,
                    self.seed, lambda _text: None, self.cancelled,
                    temperature=enhancer.CAPTION_TEMPERATURE, top_p=enhancer.CAPTION_TOP_P)
                if self.cancelled.is_set():
                    self.status.emit("Cancelled")
                    return
                caption = enhancer.clean(described)
                if not caption:
                    raise RuntimeError("The model returned no description of the image.")
                self.captioned.emit(caption)
            roster, lines, note = [], [], ""
            if self.intensity > dialogue.OFF:
                # Before the H3 prompt, never after: the lines have to be part
                # of the request the enhancer reads, not bolted onto the
                # timeline it already wrote.
                self.status.emit("Writing the dialogue…")
                roster, lines, note = dialogue.write(
                    self.prompt, self.intensity, self._chat_stream(client),
                    image_caption=caption, seed=self.seed)
                if self.cancelled.is_set():
                    self.status.emit("Cancelled")
                    return
                if lines:
                    self.spoken.emit(dialogue.transcript(roster, lines))
                if note:
                    self.status.emit(f"Dialogue — {note}")
            self.status.emit(f"{enhancer.label(self.variant, caption is not None)}…")
            written = client.stream_chat(
                dialogue.request(self.prompt, variant=self.variant, image_caption=caption,
                                 intensity=self.intensity, roster=roster, lines=lines),
                dialogue.max_tokens(self.variant, len(lines), self.intensity),
                self.seed, self.chunk.emit, self.cancelled,
                temperature=enhancer.TEMPERATURE, top_p=enhancer.TOP_P)
            if self.cancelled.is_set():
                self.status.emit("Cancelled")
                return
            prompt = enhancer.clean(written)
            if not prompt:
                raise RuntimeError("The model returned an empty prompt.")
            self.ready.emit(prompt)
            # The dialogue note is carried into the final line rather than left
            # to flash past: "skipped" is the one thing about this generation
            # somebody would want to read after it has finished.
            spoke = f" · Dialogue: {note}" if note else ""
            self.status.emit(
                f"Server: running · H3 prompt: complete{spoke} · Seed: {self.seed}")
        except Exception as exc:                      # surfaced, never swallowed
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _chat_stream(self, client):
        """The shape ``dialogue.write`` calls, so its never-raises contract and
        its own sampling are what actually run."""
        def chat_stream(messages, *, temperature=0.9, top_p=0.95, max_tokens=600, seed=None):
            return [client.stream_chat(messages, max_tokens,
                                       seed if seed is not None else self.seed,
                                       lambda _text: None, self.cancelled,
                                       temperature=temperature, top_p=top_p)]
        return chat_stream


class StructureDialog(QDialog):
    """MiniMax's own guide to what an H3 prompt is made of.

    The same text WanGP shows beside its prompt box, behind a button rather than
    on the page: it is read once while learning the format and is in the way
    every time after that.
    """

    def __init__(self, variant: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("H3 prompt structure")
        column = QVBoxLayout(self)
        guide = QTextBrowser()
        guide.setOpenExternalLinks(True)
        guide.setMarkdown(enhancer.infos(variant))
        touch.flickable(guide)
        column.addWidget(guide, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        column.addLayout(row)
        if parent is not None:
            self.resize(int(parent.width() * 0.7), int(parent.height() * 0.8))


class MiniMaxPage(QWidget):
    """The whole of MiniMax H3 mode."""

    def __init__(self, paths, service_provider, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.service_provider = service_provider
        self.image_path: Path | None = None
        self.metrics = touch.metrics(1.0)
        self.thread: QThread | None = None
        self._worker: EnhanceWorker | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.compose_pane())
        splitter.addWidget(self.output_pane())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        column = QVBoxLayout(self)
        column.addWidget(splitter, 1)
        column.addLayout(self.action_bar())
        # After both panes exist: this one reaches across the splitter.
        self.intensity.valueChanged.connect(self.follow_intensity)
        self.describe_generation()
        self.set_status("Say what the video should be, attach a picture if there is one, "
                        f"then press {enhancer.BUTTON_LABEL}.")

    # ── layout ───────────────────────────────────────────────────────────────

    def compose_pane(self) -> QWidget:
        pane = QWidget()
        column = QVBoxLayout(pane)
        column.addWidget(self.heading("Prompt"))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Describe the video and its sound…")
        # WanGP's own escape hatch, which is nowhere in its interface either.
        self.prompt.setToolTip("Anything after @ is added to the H3 instructions for this\n"
                               "prompt; anything after @@ replaces them entirely.")
        touch.flickable(self.prompt)
        column.addWidget(self.prompt, 1)
        column.addLayout(self.image_row())
        column.addLayout(self.model_row())
        # What WanGP calls the generation these two choices add up to, under the
        # two choices that decide it.
        self.note = QLabel()
        self.note.setObjectName("fieldLabel")
        self.note.setWordWrap(True)
        column.addWidget(self.note)
        column.addWidget(touch.labelled("Dialogue", self.dialogue_field()))
        return pane

    def dialogue_field(self) -> QWidget:
        """The slider, and the sentence that says what its position means.

        Below the generation label rather than beside the H3 model, because it
        is the one control on the page that is not WanGP's: the two above decide
        which prompt is being written, and this one decides how much of the
        video is somebody talking.
        """
        self.intensity = touch.TouchSlider(Qt.Orientation.Horizontal)
        self.intensity.setRange(dialogue.OFF, dialogue.MOST)
        self.intensity.setValue(dialogue.OFF)
        self.intensity.setPageStep(1)
        self.intensity.setTickInterval(1)
        self.intensity.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.intensity.setToolTip(
            "Off leaves the H3 request exactly as WanGP would have built it.\n"
            "Above that, the speech is written first — for the people in the prompt,\n"
            "or for an unseen narrator when there are none — and the H3 instructions\n"
            "are told how much of the video has to carry it.")
        self.dialogue_note = QLabel()
        self.dialogue_note.setObjectName("fieldLabel")
        self.dialogue_note.setWordWrap(True)
        self.intensity.valueChanged.connect(self.describe_dialogue)
        self.describe_dialogue(self.intensity.value())
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(self.intensity)
        column.addWidget(self.dialogue_note)
        return holder

    def describe_dialogue(self, value: int) -> None:
        self.dialogue_note.setText(dialogue.describe(value))

    def follow_intensity(self, value: int) -> None:
        """Hide a script the next generation will not use, at Off.

        Connected from the constructor rather than from ``dialogue_field``,
        because the pane it hides belongs to the other half of the splitter and
        does not exist yet while the slider is being built.
        """
        self.show_spoken(value > dialogue.OFF and bool(self.spoken.toPlainText().strip()))

    def image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.image_label = QLabel(NO_IMAGE)
        self.image_label.setWordWrap(True)
        browse = QPushButton("Attach image…")
        browse.clicked.connect(self.browse_image)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self.remove_image)
        row.addWidget(self.image_label, 1)
        row.addWidget(browse)
        row.addWidget(self.remove_button)
        return row

    def model_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.variant = touch.touchable_popup(QComboBox())
        for value, label in enhancer.VARIANTS:
            self.variant.addItem(label, value)
        self.variant.setToolTip("FL2VA writes the three-field audiovisual prompt for text, a\n"
                                "first frame or a last frame. Ref2VA writes the six-section\n"
                                "reference prompt, where a picture is a subject rather than\n"
                                "a keyframe.")
        self.variant.currentIndexChanged.connect(lambda _index: self.describe_generation())
        guide = QPushButton("Prompt structure…")
        guide.setToolTip("What the fields of an H3 prompt are, and how shots connect")
        guide.clicked.connect(self.open_structure)
        row.addWidget(touch.labelled("H3 model", self.variant), 1)
        row.addWidget(guide, 0, Qt.AlignmentFlag.AlignBottom)
        return row

    def output_pane(self) -> QWidget:
        """The prompt, what the model saw in the picture, and what it wrote to say.

        The two boxes under the prompt are both there for the same reason and
        both hide when they have nothing in them: a prompt describing the wrong
        jacket is a caption that got the jacket wrong, and a prompt in the wrong
        voice is a script in the wrong voice.
        """
        pane = QWidget()
        column = QVBoxLayout(pane)
        self.output = QTextEdit()
        self.caption = QTextEdit()
        self.caption.setReadOnly(True)
        self.spoken = QTextEdit()
        self.spoken.setReadOnly(True)
        for edit in (self.output, self.caption, self.spoken):
            edit.setAcceptRichText(False)
            touch.flickable(edit)
        column.addWidget(self.output_header("H3 prompt", self.output))
        column.addWidget(self.output, 3)
        self.caption_header = self.output_header("Image caption", self.caption)
        column.addWidget(self.caption_header)
        column.addWidget(self.caption, 1)
        self.spoken_header = self.output_header("Dialogue added", self.spoken)
        column.addWidget(self.spoken_header)
        column.addWidget(self.spoken, 1)
        row = QHBoxLayout()
        save = QPushButton("Save .txt…")
        save.clicked.connect(self.save)
        row.addWidget(save, 1)
        column.addLayout(row)
        self.show_caption(False)
        self.show_spoken(False)
        return pane

    def output_header(self, title, source) -> QWidget:
        """A title with the Copy button for its box, as one widget.

        One widget rather than a row of two, because the caption's header is
        hidden along with the caption: a pane with nothing in it is a heading
        and a dead button.
        """
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        copy = QPushButton("Copy")
        copy.clicked.connect(lambda _checked=False, edit=source: self.copy(edit))
        row.addWidget(self.heading(title))
        row.addStretch(1)
        row.addWidget(copy)
        return holder

    def action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.write_button = QPushButton(enhancer.BUTTON_LABEL)
        self.write_button.setObjectName("primary")
        self.write_button.clicked.connect(self.write_prompt)
        bar.addWidget(self.status, 1)
        bar.addWidget(clear)
        bar.addWidget(self.cancel_button)
        bar.addWidget(self.write_button)
        return bar

    @staticmethod
    def heading(text) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    # ── the two choices ──────────────────────────────────────────────────────

    def chosen_variant(self) -> str:
        return enhancer.variant_of(self.variant.currentData())

    def describe_generation(self) -> None:
        """Name the generation the way WanGP names it."""
        self.note.setText(enhancer.label(self.chosen_variant(), self.image_path is not None))

    def open_structure(self) -> None:
        StructureDialog(self.chosen_variant(), self).exec()

    def browse_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Start or reference image", "",
                                                  "Images (*.png *.jpg *.jpeg *.webp)")
        if not filename:
            return
        self.image_path = Path(filename)
        # The name rather than the path, which would push the buttons beside it
        # off the pane; the path is still there to hover or long-press for.
        self.image_label.setText(f"Image: {self.image_path.name}")
        self.image_label.setToolTip(filename)
        self.remove_button.setEnabled(True)
        self.describe_generation()
        # Said now rather than when the generation fails: whether a picture can
        # be described at all depends on the model that is running.
        if not self.vision_ready():
            self.set_status(f"{self.image_path.name} is attached, but the model running has no "
                            "vision projector and cannot be shown it. Choose one under "
                            "Settings → Which model runs.")

    def remove_image(self) -> None:
        self.image_path = None
        self.image_label.setText(NO_IMAGE)
        self.image_label.setToolTip("")
        self.remove_button.setEnabled(False)
        self.describe_generation()

    def vision_ready(self) -> bool:
        """Whether a picture can reach the model — false only when it is known to be.

        An install that is not set up yet, or a state file that cannot be read,
        is setup's problem to describe rather than a reason to tell somebody
        their model cannot see.
        """
        if not self.paths.configured:
            return True
        try:
            return bool(self.service_provider().vision_ready())
        except Exception:
            return True

    def show_caption(self, shown: bool) -> None:
        for widget in (self.caption_header, self.caption):
            widget.setVisible(shown)

    def show_spoken(self, shown: bool) -> None:
        for widget in (self.spoken_header, self.spoken):
            widget.setVisible(shown)

    # ── writing one ──────────────────────────────────────────────────────────

    def write_prompt(self) -> None:
        if self.busy():
            return
        if not self.prompt.toPlainText().strip():
            QMessageBox.warning(self, "Missing prompt", "Enter what the video should be first.")
            return
        image = None
        if self.image_path is not None:
            if not self.vision_ready():
                QMessageBox.critical(self, "No vision projector",
                                     "The model running has no vision projector, so the attached "
                                     "image cannot be described.\n\nChoose one under Settings → "
                                     "Which model runs, or remove the image.")
                return
            try:
                image = image_data_url(self.image_path)
            except Exception as exc:
                QMessageBox.critical(self, "Image error", str(exc))
                return
        self.output.clear()
        self.caption.clear()
        self.spoken.clear()
        self.show_caption(image is not None)
        self.show_spoken(False)
        self.write_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        # A fresh seed per run, because WanGP randomizes the enhancer's seed by
        # default and an H3 prompt asked for twice should not come back twice
        # the same.
        worker = EnhanceWorker(self.service_provider(), self.prompt.toPlainText(),
                               self.chosen_variant(), image, draw_seed(),
                               self.intensity.value())
        self.thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.chunk.connect(self.output.insertPlainText)
        worker.captioned.connect(self.caption.setPlainText)
        worker.spoken.connect(self.show_spoken_script)
        worker.ready.connect(self.output.setPlainText)
        worker.status.connect(self.set_status)
        worker.failed.connect(self.generation_failed)
        worker.finished.connect(self.thread.quit)
        worker.finished.connect(worker.deleteLater)
        self.thread.finished.connect(self.generation_done)
        self.thread.start()

    def show_spoken_script(self, script: str) -> None:
        self.spoken.setPlainText(script)
        self.show_spoken(bool(script.strip()))

    def generation_failed(self, message: str) -> None:
        self.set_status("Could not write the prompt")
        QMessageBox.critical(self, "Could not write the prompt", message)

    def generation_done(self) -> None:
        self.write_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread, self._worker = None, None

    def cancel_generation(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.set_status("Cancelling…")

    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    # ── what comes out ───────────────────────────────────────────────────────

    @staticmethod
    def copy(source) -> None:
        source.selectAll()
        source.copy()

    def save(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save the H3 prompt", "h3-prompt.txt",
                                                  "Text (*.txt)")
        if filename:
            Path(filename).write_text(self.output.toPlainText() + "\n", encoding="utf-8")

    def clear(self) -> None:
        self.prompt.clear()
        self.output.clear()
        self.caption.clear()
        self.spoken.clear()
        self.show_caption(False)
        self.show_spoken(False)
        self.remove_image()

    # ── plumbing ─────────────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def apply_metrics(self, metrics: dict) -> None:
        """Follow the window's display size."""
        self.metrics = metrics
        self.prompt.setMinimumHeight(metrics["target"] * 3)
        self.output.setMinimumHeight(metrics["target"] * 3)
        for edit in (self.caption, self.spoken):
            edit.setMinimumHeight(metrics["target"] * 2)
            edit.setMaximumHeight(metrics["target"] * 3)
        self.write_button.setMinimumWidth(metrics["target"] * 4)

    def rebind(self, paths) -> None:
        """Follow a setup run that moved the installation root."""
        self.paths = paths

    def shutdown(self) -> None:
        if self.busy() and self.thread is not None:
            self.cancel_generation()
            self.thread.quit()
            self.thread.wait(3000)
