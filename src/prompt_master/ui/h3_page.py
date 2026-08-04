"""MiniMax-H3 mode: the same two panes, a different model's prompt.

Laid out like prompt mode on purpose — an intent on the left, the settings that
shape it under that, the finished prompt on the right, and a bar along the
bottom that never scrolls. Somebody who knows where Generate is in one mode
knows where it is in the other, and the differences between the two tabs should
be differences of substance rather than of furniture.

The substance is different in three ways, and each one shows on screen.

*The output has no negative half.* H3 takes one prompt. So the right-hand pane
is the brief and, under it, the checks — what ``minimax_h3.checks`` found wrong
with what came back, plus the character count against the 7,000 the API takes.
That pane is where a negative prompt would have been, and it earns the space:
an H3 brief is a specification, and a specification is worth proof-reading.

*The frames are anchors, not a hint.* Prompt mode attaches one optional still.
Here the H3 mode says which frames there are — none, a first, a first and a
last, or a last alone — and the rows for them appear and disappear with it,
because an attached last frame means nothing in a mode that has no last frame.

*The prompt is assembled, not just streamed.* The model writes three fields;
this page shows them arriving raw, then replaces them with the brief
``minimax_h3.assemble`` built — instruction line, blank line, fields in order.
The instruction cannot be written before the brief is: it names the real final
shot number, which is whatever the writer used. Watching it write and reading
what it wrote are different needs, and the second one is the one that gets
copied.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
                               QTextEdit, QVBoxLayout, QWidget)

from prompt_master.core.models import RANDOM_SEED, draw_seed
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.prompt_engine import minimax_h3 as h3
from prompt_master.ui import touch


class H3Worker(QObject):
    """One brief, written off the UI thread.

    Shorter than prompt mode's worker because there is less to do: H3 has no
    negative prompt and therefore no second pass. What it adds instead is the
    read-back — parsing the fields and assembling them — which happens here
    rather than in the slot so a slow regex never runs on the UI thread.
    """

    chunk = Signal(str)
    ready = Signal(str, dict)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service, request: h3.H3Request):
        super().__init__()
        self.service, self.request = service, request
        self.cancelled = threading.Event()

    @Slot()
    def cancel(self) -> None:
        self.cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            frames = h3.anchored(self.request.mode)
            self.status.emit("Starting llama-server…")
            client = self.service.client(frames)
            # After the client, so a projector that failed to load is reported
            # as the missing projector it is rather than as a refusal here.
            messages = h3.messages(self.request, vision_available=True)
            low, high = h3.word_budget(self.request.seconds)
            shots_low, shots_high = h3.shot_range(self.request)
            shots = f"{shots_low}" if shots_low == shots_high else f"{shots_low}-{shots_high}"
            self.status.emit(f"Writing the brief… ({shots} shots, {low}-{high} words)")
            temperature, top_p = h3.sampling()
            raw = client.stream_chat(messages, h3.max_tokens(self.request), self.request.seed,
                                     self.chunk.emit, self.cancelled,
                                     temperature=temperature, top_p=top_p)
            if self.cancelled.is_set():
                self.status.emit("Generation cancelled")
                return
            fields = h3.parse(raw)
            if not fields:
                raise RuntimeError("The model returned an empty brief.")
            brief = h3.assemble(fields, self.request)
            self.ready.emit(brief, fields)
            self.status.emit(f"Server: running · Generation: complete · Seed: {self.request.seed}")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class H3Page(QWidget):
    """Everything MiniMax-H3 mode is."""

    def __init__(self, paths, service_provider, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.service_provider = service_provider
        self.thread: QThread | None = None
        self._worker: H3Worker | None = None
        self.first_path: Path | None = None
        self.last_path: Path | None = None
        self.fields: dict[str, str] = {}
        # The request the brief on screen was written from. The checks are read
        # against it rather than against the controls, so editing a setting
        # after a generation does not silently re-judge the brief that is up.
        self.pending = h3.H3Request(intent="")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.compose_pane())
        splitter.addWidget(self.output_pane())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(splitter, 1)
        column.addLayout(self.action_bar())
        self.follow_mode()

    # ── layout ───────────────────────────────────────────────────────────────

    def compose_pane(self) -> QWidget:
        pane = QWidget()
        column = QVBoxLayout(pane)
        column.addWidget(self.heading("Intent"))
        self.intent = QPlainTextEdit()
        self.intent.setPlaceholderText("Describe the video you want H3 to make…")
        self.intent.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        touch.flickable(self.intent)
        column.addWidget(self.intent)
        self.first_row = self.frame_row("first")
        self.last_row = self.frame_row("last")
        column.addWidget(self.first_row)
        column.addWidget(self.last_row)
        column.addWidget(self.settings_area(), 1)
        return pane

    def frame_row(self, which: str) -> QWidget:
        """The attach/remove row for one anchor frame.

        A widget rather than a layout so the mode can hide the whole row: the
        label, the button and the Remove beside it are one thing that is either
        part of this mode or not part of it.
        """
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"No {which} frame")
        label.setWordWrap(True)
        browse = QPushButton(f"Attach {which} frame…")
        browse.clicked.connect(lambda _=False, w=which: self.browse_frame(w))
        remove = QPushButton("Remove")
        remove.setEnabled(False)
        remove.clicked.connect(lambda _=False, w=which: self.remove_frame(w))
        row.addWidget(label, 1)
        row.addWidget(browse)
        row.addWidget(remove)
        setattr(self, f"{which}_label", label)
        setattr(self, f"{which}_remove", remove)
        return holder

    def settings_area(self) -> QScrollArea:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        for title, fields in self.controls():
            column.addWidget(self.section(title, fields))
        column.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(page)
        area.setFrameShape(QFrame.Shape.NoFrame)
        touch.flickable(area)
        return area

    def controls(self):
        """Every control, grouped by what it changes about the brief.

        The groups follow the guide's own division rather than the window's
        convenience: the timeline, the look along it, the voices on it, and the
        two sound fields that sit beside it.
        """
        self.mode = touch.combo(h3.MODES, h3.T2VA)
        self.mode.currentIndexChanged.connect(lambda _index: self.follow_mode())
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(h3.MIN_SECONDS, h3.MAX_SECONDS)
        self.seconds.setSingleStep(0.5)
        self.seconds.setValue(10.0)
        self.seconds.setSuffix(" s")
        self.shots = touch.combo(h3.SHOTS, "auto")
        self.seed = QSpinBox()
        self.seed.setRange(RANDOM_SEED, 2 ** 31 - 1)
        self.seed.setValue(7)
        self.seed.setSpecialValueText("Random each time (-1)")

        self.style = touch.combo(h3.STYLES, "auto")
        self.camera = touch.combo(h3.CAMERAS, "auto")
        self.amplitude = touch.combo(h3.AMPLITUDES, "medium")
        self.speed = touch.combo(h3.SPEEDS, "normal")
        self.cut = touch.combo(h3.CUTS, "cut")

        self.speakers = QSpinBox()
        self.speakers.setRange(0, h3.MAX_SPEAKERS)
        self.speakers.setValue(0)
        self.talk = touch.combo(h3.TALK, "none")
        self.language = touch.combo(h3.LANGUAGES, "English")
        self.on_screen = QPlainTextEdit()
        self.on_screen.setPlaceholderText("Words that must appear on screen, reproduced exactly.")
        touch.flickable(self.on_screen)
        # The line count is worked out from the duration and the density, and
        # both of those move, so the sentence saying what they add up to is
        # rewritten whenever either does.
        self.talk_note = QLabel()
        self.talk_note.setObjectName("fieldLabel")
        self.talk_note.setWordWrap(True)
        for control in (self.speakers, self.seconds):
            control.valueChanged.connect(lambda _value: self.describe_talk())
        self.talk.currentIndexChanged.connect(lambda _index: self.describe_talk())
        self.describe_talk()

        self.soundscape_mode = touch.combo(h3.SOUNDSCAPES, "scene")
        self.music = touch.combo(h3.MUSIC_MODES, "off")
        self.music_brief = QPlainTextEdit()
        self.music_brief.setPlaceholderText("Instrumentation, tempo, rhythm, how it changes.")
        touch.flickable(self.music_brief)
        self.ambience = QPlainTextEdit()
        self.ambience.setPlaceholderText("Ambience to build from: rain, traffic, a room tone.")
        touch.flickable(self.ambience)
        for box in (self.music, self.soundscape_mode):
            box.currentIndexChanged.connect(lambda _index: self.follow_sound())
        self.follow_sound()

        self.cast = QPlainTextEdit()
        self.cast.setPlaceholderText("Name = description, one per line. Keeps a face the same across shots.")
        touch.flickable(self.cast)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Anything else the brief has to obey.")
        touch.flickable(self.notes)

        return [
            # "H3 mode" rather than "Mode": the window already has a Mode menu,
            # and this is the other kind — H3's own T2VA/I2VA/FL2VA.
            ("Timeline", [("H3 mode", self.mode), ("Duration", touch.stepper(self.seconds)),
                          ("Shots", self.shots), ("Seed", touch.stepper(self.seed))]),
            ("Look", [("Style", self.style), ("Camera move", self.camera),
                      ("Amplitude", self.amplitude), ("Speed", self.speed),
                      ("Transitions", self.cut)]),
            ("Voices", [("Speaking characters", touch.stepper(self.speakers)),
                        ("How much talking", self.talk), ("Language", self.language),
                        (None, self.talk_note, 2), ("On-screen text", self.on_screen)]),
            ("Sound", [("Soundscape", self.soundscape_mode),
                       ("Non-diegetic music", self.music),
                       ("Ambience", self.ambience), ("The score", self.music_brief)]),
            ("Continuity", [("Cast", self.cast), ("Notes", self.notes)]),
        ]

    def section(self, title, fields) -> QGroupBox:
        """One group, two columns wide — prompt mode's own arrangement."""
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        row = column = 0
        for caption, widget, *rest in fields:
            span = rest[0] if rest else (2 if caption is None or isinstance(widget, QPlainTextEdit) else 1)
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
        pane = QWidget()
        column = QVBoxLayout(pane)
        self.prompt = QTextEdit()
        self.prompt.setAcceptRichText(False)
        touch.flickable(self.prompt)
        header = QHBoxLayout()
        header.addWidget(self.heading("H3 prompt"))
        header.addStretch(1)
        self.count = QLabel("")
        self.count.setObjectName("fieldLabel")
        header.addWidget(self.count)
        copy = QPushButton("Copy")
        copy.clicked.connect(self.copy)
        header.addWidget(copy)
        column.addLayout(header)
        column.addWidget(self.prompt, 4)

        column.addWidget(self.heading("Checks"))
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setAcceptRichText(False)
        self.report.setPlaceholderText("What the guide's rules say about the brief above.")
        touch.flickable(self.report)
        column.addWidget(self.report, 1)

        row = QHBoxLayout()
        save = QPushButton("Save .txt…")
        save.clicked.connect(self.save)
        row.addWidget(save, 1)
        column.addLayout(row)
        return pane

    def action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.status = QLabel("Starting…")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self.generate)
        bar.addWidget(self.status, 1)
        bar.addWidget(clear)
        bar.addWidget(self.cancel_button)
        bar.addWidget(self.generate_button)
        return bar

    @staticmethod
    def heading(text) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    # ── controls that answer each other ──────────────────────────────────────

    def follow_mode(self) -> None:
        """Show only the frames this mode actually has."""
        mode = touch.chosen(self.mode, h3.T2VA)
        self.first_row.setVisible(h3.needs_first_frame(mode))
        self.last_row.setVisible(h3.needs_last_frame(mode))

    def follow_sound(self) -> None:
        """Either sound field can be N/A, and a box that feeds one that will be
        N/A is a box whose contents are going nowhere. So it says so."""
        self.music_brief.setEnabled(touch.chosen(self.music, "off") != "off")
        self.ambience.setEnabled(touch.chosen(self.soundscape_mode, "scene") != "silence")

    def describe_talk(self) -> None:
        # Built from the three controls it depends on rather than from the whole
        # panel: this runs on every tick of a spin box, and the panel it would
        # otherwise read is not finished being built the first time it is called.
        lines = h3.spoken_lines(h3.H3Request(
            intent="", seconds=self.seconds.value(), speakers=self.speakers.value(),
            talk=touch.chosen(self.talk, "none")))
        if not lines:
            self.talk_note.setText("Nobody speaks — no <d> tags, no speaker IDs")
        else:
            voices = self.speakers.value()
            self.talk_note.setText(
                f"About {lines} spoken line{'s' if lines != 1 else ''} across the clip, "
                f"across {voices} voice{'s' if voices != 1 else ''} — (S1)"
                + (f"…(S{voices})" if voices > 1 else ""))

    # ── frames ───────────────────────────────────────────────────────────────

    def browse_frame(self, which: str) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, f"{which.title()} frame", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if not filename:
            return
        path = Path(filename)
        setattr(self, f"{which}_path", path)
        label = getattr(self, f"{which}_label")
        label.setText(f"{which.title()} frame: {path.name}")
        label.setToolTip(filename)
        getattr(self, f"{which}_remove").setEnabled(True)

    def frame_wanted(self, mode: str) -> str:
        """Which frame this H3 mode needs and has not been given, or ``""``.

        The engine asks the same question of a built request, over encoded
        pictures. This asks it of the two file paths, before anything has been
        read off the disk — the point of the guard is to stop before the work.
        """
        if h3.needs_first_frame(mode) and self.first_path is None:
            return "first"
        if h3.needs_last_frame(mode) and self.last_path is None:
            return "last"
        return ""

    def remove_frame(self, which: str) -> None:
        setattr(self, f"{which}_path", None)
        label = getattr(self, f"{which}_label")
        label.setText(f"No {which} frame")
        label.setToolTip("")
        getattr(self, f"{which}_remove").setEnabled(False)

    # ── generating ───────────────────────────────────────────────────────────

    def request(self) -> h3.H3Request:
        """The settings as the engine's own request.

        A frame is read and re-encoded only when the mode it belongs to is the
        one selected: an attached last frame left over from FL2VA must not ride
        along into an I2VA brief that has nowhere to put it.
        """
        mode = touch.chosen(self.mode, h3.T2VA)
        first = last = None
        if self.first_path is not None and h3.needs_first_frame(mode):
            first = image_data_url(self.first_path)
        if self.last_path is not None and h3.needs_last_frame(mode):
            last = image_data_url(self.last_path)
        return h3.H3Request(
            intent=self.intent.toPlainText().strip(),
            mode=mode,
            seconds=self.seconds.value(),
            shots=touch.chosen(self.shots, "auto"),
            style=touch.chosen(self.style, "auto"),
            camera=touch.chosen(self.camera, "auto"),
            amplitude=touch.chosen(self.amplitude, "medium"),
            speed=touch.chosen(self.speed, "normal"),
            cut=touch.chosen(self.cut, "cut"),
            language=touch.chosen(self.language, "English"),
            speakers=self.speakers.value(),
            talk=touch.chosen(self.talk, "none"),
            on_screen_text=self.on_screen.toPlainText(),
            soundscape=self.ambience.toPlainText(),
            ambience=touch.chosen(self.soundscape_mode, "scene"),
            music=touch.chosen(self.music, "off"),
            music_brief=self.music_brief.toPlainText(),
            cast=self.cast.toPlainText(),
            notes=self.notes.toPlainText(),
            first_frame=first,
            last_frame=last,
            # -1 means a different one every time, resolved here so the number
            # in the status bar is the number that was actually sent.
            seed=self.seed.value() if self.seed.value() != RANDOM_SEED else draw_seed(),
        )

    def generate(self) -> None:
        if not self.intent.toPlainText().strip():
            QMessageBox.warning(self, "Missing intent", "Describe the video first.")
            return
        if self.thread is not None and self.thread.isRunning():
            return
        mode = touch.chosen(self.mode, h3.T2VA)
        wanted = self.frame_wanted(mode)
        if wanted:
            QMessageBox.warning(
                self, f"{wanted.title()} frame required",
                f"This H3 mode is anchored to a {wanted} frame. Attach one, or choose an H3 "
                "mode that does not need it.")
            return
        service = self.service_provider()
        # Asked while the frames are still attached and there is a choice to
        # make about them, rather than left to fail on the wire. Only on an
        # install that finished: an unconfigured one has a better answer of its
        # own, and service.client is what gives it.
        if h3.anchored(mode) and self.paths.configured and not service.vision_ready():
            QMessageBox.critical(self, "No vision projector",
                                 "The model running has no vision projector, so the anchor "
                                 "frames cannot be sent to it.\n\nChoose one under Settings → "
                                 "Which model runs, or switch to text to video.")
            return
        try:
            request = self.request()
        except Exception as exc:
            QMessageBox.critical(self, "Image error", str(exc))
            return

        self.pending = request
        self.prompt.clear()
        self.report.clear()
        self.count.setText("")
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.thread = QThread(self)
        worker = H3Worker(service, request)
        worker.moveToThread(self.thread)
        self._worker = worker
        self.thread.started.connect(worker.run)
        worker.chunk.connect(self.prompt.insertPlainText)
        worker.ready.connect(self.show_brief)
        worker.status.connect(self.status.setText)
        worker.failed.connect(self.generation_failed)
        worker.finished.connect(self.thread.quit)
        worker.finished.connect(worker.deleteLater)
        self.thread.finished.connect(self.generation_done)
        self.thread.start()

    @Slot(str, dict)
    def show_brief(self, brief: str, fields: dict) -> None:
        """Replace what was streamed with what was assembled, and check it."""
        self.fields = dict(fields)
        self.prompt.setPlainText(brief)
        self.count.setText(f"{len(brief)} / {h3.CHARACTER_LIMIT} characters")
        notes = h3.checks(brief, self.fields, self.pending)
        self.report.setPlainText("\n".join(f"• {note}" for note in notes) if notes
                                 else "Nothing to flag — the brief follows the guide's rules.")

    def generation_failed(self, message: str) -> None:
        self.status.setText("Generation failed")
        QMessageBox.critical(self, "Generation failed", message)

    def generation_done(self) -> None:
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self._worker = None

    def cancel_generation(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    # ── what comes out ───────────────────────────────────────────────────────

    def copy(self) -> None:
        QApplication.clipboard().setText(self.prompt.toPlainText())

    def save(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save the brief",
                                                  "minimax-h3-prompt.txt", "Text (*.txt)")
        if filename:
            Path(filename).write_text(self.prompt.toPlainText() + "\n", encoding="utf-8")

    def clear(self) -> None:
        self.intent.clear()
        self.prompt.clear()
        self.report.clear()
        self.count.setText("")
        self.fields = {}
        self.remove_frame("first")
        self.remove_frame("last")

    # ── the window's business ────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def rebind(self, paths) -> None:
        """Follow a setup run that moved the installation root."""
        self.paths = paths

    def apply_metrics(self, metrics: dict) -> None:
        """Follow the window's display size."""
        self.intent.setMinimumHeight(metrics["target"] * 3)
        self.intent.setMaximumHeight(metrics["target"] * 4)
        for edit in (self.on_screen, self.music_brief, self.ambience, self.cast, self.notes):
            edit.setMinimumHeight(metrics["target"] * 2)
            edit.setMaximumHeight(metrics["target"] * 3)
        self.prompt.setMinimumHeight(metrics["target"] * 3)
        self.report.setMinimumHeight(metrics["target"] * 2)
        self.generate_button.setMinimumWidth(metrics["target"] * 4)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)
