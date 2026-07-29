from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (QApplication,QCheckBox,QComboBox,QFileDialog,QFrame,QGridLayout,QGroupBox,QHBoxLayout,QLabel,QMainWindow,QMessageBox,QPlainTextEdit,QPushButton,QScrollArea,QSizePolicy,QSlider,QSpinBox,QDoubleSpinBox,QSplitter,QStackedWidget,QTextEdit,QVBoxLayout,QWidget)
import threading

from prompt_master.core.config import atomic_write_json, read_json
from prompt_master.core.models import RANDOM_SEED, PromptRequest, draw_seed
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.prompt_engine import motion
from prompt_master.prompt_engine import speech
from prompt_master.prompt_engine import options as opt
from prompt_master.prompt_engine.adapter import PromptEngine, VisionUnavailable
from prompt_master.core.paths import AppPaths
from prompt_master.inference.service import InferenceService
from prompt_master.ui import touch
from prompt_master.ui.chat_page import ChatPage
from prompt_master.ui.setup_wizard import SetupWizard

# The two things this application does, and the order they appear in the mode
# drop-down. Prompt mode is first because it is what the app was, and what an
# install that has no characters yet can do.
PROMPT_MODE = "prompt"
CONVERSATION_MODE = "conversation"
MODES = ((PROMPT_MODE, "Prompt mode"), (CONVERSATION_MODE, "Conversation mode"))


class GenerationWorker(QObject):
    positive_chunk = Signal(str)
    positive_ready = Signal(str)
    negative_ready = Signal(str)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service, engine, request):
        super().__init__(); self.service, self.engine, self.request = service, engine, request; self.cancelled = threading.Event()

    @Slot()
    def cancel(self): self.cancelled.set()

    @Slot()
    def run(self):
        try:
            needs_vision = self.request.image_data_url is not None and self.request.video_mode == "i2v"
            self.status.emit("Starting llama-server…")
            # service.client() raises when vision is needed and the projector is
            # missing, so reaching this line means the still can go on the wire.
            client = self.service.client(needs_vision)
            if self.request.speech > speech.NONE:
                # Before the brief, not after: the extra lines have to be in the
                # intent the engine reads, not bolted onto the shot it wrote.
                self.status.emit("Writing extra speech…")
                self.request, note = speech.expand(self.request, self._chat_stream(client),
                                                   seed=self.request.seed)
                if note: self.status.emit(f"Intent expanded — {note}")
            plan = self.engine.build(self.request, vision_available=True)
            self.status.emit(f"Generating positive prompt… ({plan.frames} frames, {plan.word_budget[0]}-{plan.word_budget[1]} words)")
            temperature, top_p = self.engine.sampling(self.request)
            raw = client.stream_chat(plan.messages, plan.max_tokens, self.request.seed, self.positive_chunk.emit,
                                     self.cancelled, temperature=temperature, top_p=top_p)
            if self.cancelled.is_set(): self.status.emit("Generation cancelled"); return
            positive = self.engine.clean_positive(raw)
            if not positive.strip(): raise RuntimeError("The model returned an empty script.")
            self.positive_ready.emit(positive)
            auto = ""
            if self.request.smart_negative:
                self.status.emit("Negative pass…")
                auto = self.engine.run_smart_negative(positive, self._chat_stream(client))
            self.negative_ready.emit(self.engine.merge_negative(self.request, auto))
            self.status.emit(f"Server: running · Generation: complete · Seed: {self.request.seed}")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _chat_stream(self, client):
        """Shim matching upstream backend.chat_stream so negative.run_auto — its
        temperature, its guards and its never-raises contract — is used verbatim."""
        def chat_stream(messages, *, temperature=0.85, top_p=0.95, max_tokens=900, seed=None):
            return [client.stream_chat(messages, max_tokens, seed if seed is not None else self.request.seed,
                                       lambda _: None, self.cancelled, temperature=temperature, top_p=top_p)]
        return chat_stream


class MainWindow(QMainWindow):
    """The window, laid out for a finger.

    Two panes side by side rather than one tall column: writing an intent and
    reading the prompt it produced are the two things done most, and neither
    should scroll the other off the screen. The settings between them are
    grouped and scroll on their own, and the bar along the bottom — status,
    Clear, Cancel, Generate — never scrolls at all, because the button pressed
    most often is the one that should never have to be found.

    Sizes come from ``ui.touch``: every target is at least a fingertip tall, and
    the scale that multiplies them is a menu item, since a tablet held at arm's
    length and a desk monitor do not agree on how big "big enough" is.

    The window holds two of these pages, chosen by the drop-down along the top:
    prompt mode, described above, and conversation mode. They share the window,
    the display size and the one llama-server the application runs, and nothing
    else — a chat cannot reach the prompt engine, which is what keeps the engine
    the byte-for-byte copy of upstream that ``PARITY_REPORT.md`` says it is.
    """

    def __init__(self, paths: AppPaths | None = None):
        super().__init__(); self.paths = paths or AppPaths.discover(); self.service = InferenceService(self.paths); self.thread = None; self.setWindowTitle("Prompt Master Standalone"); self.image_path: Path | None = None; self.engine = PromptEngine()
        self.build_menus()
        self.pages=QStackedWidget(); self.pages.addWidget(self.prompt_page())
        # The service is handed over as a callable rather than as itself: re-running
        # setup replaces it, and the chat page must talk to the one running now.
        self.chat=ChatPage(self.paths, lambda: self.service, self); self.pages.addWidget(self.chat)
        central=QWidget(); page=QVBoxLayout(central); page.addLayout(self.mode_row()); page.addWidget(self.pages,1)
        self.setCentralWidget(central)
        self.apply_scale(touch.load_scale(self.paths), remember=False)
        self.select_mode(self.remembered_mode(), remember=False)
        self.fill_screen(); self.refresh_status()

    # ── layout ───────────────────────────────────────────────────────────────

    def prompt_page(self) -> QWidget:
        """Everything prompt mode is: the two panes, and the bar under them."""
        splitter=QSplitter(Qt.Orientation.Horizontal); splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.compose_pane()); splitter.addWidget(self.output_pane())
        splitter.setStretchFactor(0,4); splitter.setStretchFactor(1,5)
        page=QWidget(); column=QVBoxLayout(page); column.setContentsMargins(0,0,0,0)
        column.addWidget(splitter,1); column.addLayout(self.action_bar())
        return page

    def mode_row(self) -> QHBoxLayout:
        """The one control that is above both pages rather than on one of them."""
        row=QHBoxLayout(); label=QLabel("Mode"); label.setObjectName("fieldLabel")
        self.mode_selector=self.combo(MODES,PROMPT_MODE)
        self.mode_selector.currentIndexChanged.connect(lambda _index: self.select_mode(self.chosen(self.mode_selector,PROMPT_MODE)))
        row.addWidget(label); row.addWidget(self.mode_selector); row.addStretch(1)
        return row

    def select_mode(self, mode: str, remember: bool = True):
        self.pages.setCurrentIndex(1 if mode == CONVERSATION_MODE else 0)
        self.select(self.mode_selector,mode)
        if remember:
            settings=self.paths.data/touch.SETTINGS_FILE
            try: current=read_json(settings)
            except (OSError,ValueError): current={}
            try: atomic_write_json(settings,{**current,"mode":mode})
            except OSError: pass                      # remembering is a convenience

    def remembered_mode(self) -> str:
        try: mode=read_json(self.paths.data/touch.SETTINGS_FILE).get("mode")
        except (OSError,ValueError): return PROMPT_MODE
        return mode if mode in (PROMPT_MODE,CONVERSATION_MODE) else PROMPT_MODE

    def build_menus(self):
        settings_menu=self.menuBar().addMenu("Settings"); settings_menu.addAction("Models and Hardware…").triggered.connect(self.open_setup)
        view_menu=self.menuBar().addMenu("View"); sizes=view_menu.addMenu("Display size"); self.size_actions=QActionGroup(self); self.size_actions.setExclusive(True)
        for name in touch.SCALES:
            action=sizes.addAction(name); action.setCheckable(True); action.setData(name); self.size_actions.addAction(action)
            action.triggered.connect(lambda _checked=False,chosen=name: self.apply_scale(chosen))

    def compose_pane(self) -> QWidget:
        """Intent and image stay put; the settings under them scroll."""
        pane=QWidget(); column=QVBoxLayout(pane)
        column.addWidget(self.heading("Intent"))
        self.intent=QPlainTextEdit(); self.intent.setPlaceholderText("Describe the video you want to create…")
        self.intent.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed); touch.flickable(self.intent)
        column.addWidget(self.intent)
        column.addLayout(self.image_row())
        column.addWidget(self.settings_area(),1)
        return pane

    def image_row(self) -> QHBoxLayout:
        row=QHBoxLayout(); self.image_label=QLabel("No image — text to video"); self.image_label.setWordWrap(True)
        browse=QPushButton("Attach image…"); browse.clicked.connect(self.browse_image)
        self.remove_button=QPushButton("Remove"); self.remove_button.setEnabled(False); self.remove_button.clicked.connect(self.remove_image)
        row.addWidget(self.image_label,1); row.addWidget(browse); row.addWidget(self.remove_button)
        return row

    def settings_area(self) -> QScrollArea:
        page=QWidget(); column=QVBoxLayout(page); column.setContentsMargins(0,0,0,0)
        for title,fields in self.controls(): column.addWidget(self.section(title,fields))
        column.addStretch(1)
        area=QScrollArea(); area.setWidgetResizable(True); area.setWidget(page); area.setFrameShape(QFrame.Shape.NoFrame)
        touch.flickable(area)
        return area

    def controls(self):
        """Every control, grouped by what it changes about the shot."""
        d=opt.DEFAULTS
        self.mode=self.combo(opt.VIDEO_MODES,d["video_mode"])
        self.seconds=QDoubleSpinBox(); self.seconds.setRange(1,60); self.seconds.setSingleStep(0.5); self.seconds.setValue(d["seconds"]); self.seconds.setSuffix(" s")
        self.fps=QSpinBox(); self.fps.setRange(8,60); self.fps.setValue(d["fps"])
        self.dimensions=self.combo([("704x1216","704 × 1216 (portrait)"),("1216x704","1216 × 704 (landscape)"),("768x768","768 × 768 (square)"),("1920x1080","1920 × 1080"),("1080x1920","1080 × 1920")],f"{d['output_width']}x{d['output_height']}")
        self.seed=QSpinBox(); self.seed.setRange(RANDOM_SEED,2**31-1); self.seed.setValue(d["seed"])
        # Qt shows the special text in place of the minimum, which is what -1 is.
        self.seed.setSpecialValueText("Random each time (-1)")
        self.style=self.grouped_combo(opt.STYLES_GROUPED,d["style"])
        self.motion=self.combo(motion.OPTIONS,motion.DEFAULT)
        self.camera=self.combo(opt.CAMERAS,d["camera"])
        self.transition=self.combo(opt.TRANSITIONS,d["transition"])
        self.pov=self.combo(opt.POV,d["pov"])
        self.wardrobe=self.combo(opt.WARDROBE,d["wardrobe"])
        self.undress=QCheckBox("Undress sequence"); self.undress.setChecked(d["undress"])
        self.accent=self.combo(opt.ACCENTS,d["accent"])
        self.accent_strength=self.combo(opt.ACCENT_STRENGTHS,d["accent_strength"])
        self.dialogue=QSpinBox(); self.dialogue.setRange(0,100); self.dialogue.setValue(d["dialogue"]); self.dialogue.setSuffix("%")
        self.music=self.combo(opt.MUSIC,d["music"])
        self.music_bg=QCheckBox("Music plays low under the scene"); self.music_bg.setChecked(d["music_bg"])
        self.speech_slider=touch.TouchSlider(Qt.Orientation.Horizontal)
        self.speech_slider.setRange(speech.NONE,speech.MOST); self.speech_slider.setValue(speech.NONE)
        self.speech_slider.setPageStep(1); self.speech_slider.setTickInterval(1); self.speech_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speech_slider.setToolTip("Extra lines are written in the voice the intent already quotes, mixed back\ninto the intent, and the dialogue budget above is lifted to match so they survive\ninto the finished shot.")
        self.speech_note=QLabel(); self.speech_note.setObjectName("fieldLabel"); self.speech_note.setWordWrap(True)
        self.speech_slider.valueChanged.connect(self.describe_speech); self.describe_speech(self.speech_slider.value())
        # The sentence names a budget worked out from the dial above it, so it
        # goes stale the moment that dial moves.
        self.dialogue.valueChanged.connect(lambda _value: self.describe_speech(self.speech_slider.value()))
        self.output_format=self.combo(opt.OUTPUT_FORMATS,d["fmt"])
        self.smart=QCheckBox("Smart negative — a second pass over the finished script"); self.smart.setChecked(d["smart_negative"])
        self.lexicon=QPlainTextEdit(); self.lexicon.setPlaceholderText("Name = description, one per line. Only names present in the intent are used."); touch.flickable(self.lexicon)
        self.negative_extra=QPlainTextEdit(); self.negative_extra.setPlaceholderText("Extra terms to keep out of the shot, comma separated."); touch.flickable(self.negative_extra)
        return [
            ("Shot", [("Video mode",self.mode),("Duration",touch.stepper(self.seconds)),
                      ("FPS",touch.stepper(self.fps)),("Dimensions",self.dimensions),
                      ("Seed",touch.stepper(self.seed))]),
            ("Look", [("Style",self.style),("Motion",self.motion),("Camera",self.camera),
                      ("Transition",self.transition),("First person",self.pov),
                      ("Wardrobe",self.wardrobe),(None,self.undress)]),
            ("Voice and music", [("Accent",self.accent),("Accent strength",self.accent_strength),
                                 ("Dialogue / talk",touch.stepper(self.dialogue)),("Music",self.music),
                                 (None,self.music_bg),("Extra speech",self.speech_field(),2)]),
            ("Wording", [("Output format",self.output_format),(None,self.smart),
                         ("Lexicon",self.lexicon),("Extra negative terms",self.negative_extra)]),
        ]

    def section(self, title, fields) -> QGroupBox:
        """One group, two columns wide. A control with no caption of its own —
        a check box says what it is — takes the full width."""
        box=QGroupBox(title); grid=QGridLayout(box); grid.setColumnStretch(0,1); grid.setColumnStretch(1,1)
        row=column=0
        for caption,widget,*rest in fields:
            span=rest[0] if rest else (2 if caption is None or isinstance(widget,QPlainTextEdit) else 1)
            if span == 2 and column: row+=1; column=0
            grid.addWidget(self.field(caption,widget),row,column,1,span)
            column+=span
            if column >= 2: row+=1; column=0
        return box

    def speech_field(self) -> QWidget:
        """The slider with the sentence that says what its position means. Ten
        positions of a bare track say nothing; "3× the lines" says all of it."""
        holder=QWidget(); column=QVBoxLayout(holder); column.setContentsMargins(0,0,0,0); column.setSpacing(2)
        column.addWidget(self.speech_slider); column.addWidget(self.speech_note)
        return holder

    def describe_speech(self, value):
        if value <= speech.NONE:
            self.speech_note.setText("Speech exactly as the intent quotes it")
        else:
            self.speech_note.setText(
                f"{value}× the lines — extra speech written to match, and the dialogue "
                f"budget raised to {speech.dialogue_floor(value, self.dialogue.value())}%")

    @staticmethod
    def field(caption, widget) -> QWidget:
        """Caption above its control, not beside it: the control gets the whole
        column width, which is what makes it wide enough to hit."""
        return widget if caption is None else touch.labelled(caption,widget)

    def output_pane(self) -> QWidget:
        """The finished prompts. QTextEdit rather than QPlainTextEdit for these
        two alone: it scrolls by pixel, so a flick through a long script glides
        instead of stepping a line at a time."""
        pane=QWidget(); column=QVBoxLayout(pane)
        self.positive=QTextEdit(); self.negative=QTextEdit()
        for edit in (self.positive,self.negative): edit.setAcceptRichText(False); touch.flickable(edit)
        column.addLayout(self.output_header("Positive prompt",self.positive)); column.addWidget(self.positive,3)
        column.addLayout(self.output_header("Negative prompt",self.negative)); column.addWidget(self.negative,2)
        row=QHBoxLayout()
        both=QPushButton("Copy both"); both.clicked.connect(self.copy_both)
        save=QPushButton("Save .txt…"); save.clicked.connect(self.save)
        row.addWidget(both,1); row.addWidget(save,1); column.addLayout(row)
        return pane

    def output_header(self, title, source) -> QHBoxLayout:
        row=QHBoxLayout(); row.addWidget(self.heading(title)); row.addStretch(1)
        copy=QPushButton("Copy"); copy.clicked.connect(lambda _=False,s=source: self.copy(s)); row.addWidget(copy)
        return row

    def action_bar(self) -> QHBoxLayout:
        """Always on screen: what the app is doing, and the button pressed most."""
        bar=QHBoxLayout(); self.status=QLabel("Starting…"); self.status.setObjectName("status"); self.status.setWordWrap(True)
        clear=QPushButton("Clear"); clear.clicked.connect(self.clear)
        self.cancel_button=QPushButton("Cancel"); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel_generation)
        self.generate_button=QPushButton("Generate"); self.generate_button.setObjectName("primary"); self.generate_button.clicked.connect(self.generate)
        bar.addWidget(self.status,1); bar.addWidget(clear); bar.addWidget(self.cancel_button); bar.addWidget(self.generate_button)
        return bar

    @staticmethod
    def heading(text) -> QLabel:
        label=QLabel(text); label.setObjectName("sectionTitle"); return label

    # ── size ─────────────────────────────────────────────────────────────────

    def apply_scale(self, name: str, remember: bool = True):
        """Restyle everything for the chosen display size, live.

        The style sheet goes on the application rather than this window so the
        setup wizard is sized by the same choice.
        """
        self.scale_name=name; scale=touch.SCALES[name]; m=touch.metrics(scale)
        application=QApplication.instance()
        (application or self).setStyleSheet(touch.stylesheet(scale))
        for action in self.size_actions.actions(): action.setChecked(action.data() == name)
        self.intent.setMinimumHeight(m["target"]*3)
        self.intent.setMaximumHeight(m["target"]*4)
        for edit in (self.lexicon,self.negative_extra): edit.setMinimumHeight(m["target"]*2); edit.setMaximumHeight(m["target"]*3)
        for edit in (self.positive,self.negative): edit.setMinimumHeight(m["target"]*3)
        self.generate_button.setMinimumWidth(m["target"]*4)
        for box in self.findChildren(QComboBox): box.setMaxVisibleItems(8)
        # Conversation mode is sized by the same choice; it holds the numbers
        # itself because a bubble's picture and avatar are measured from them.
        if hasattr(self,"chat"): self.chat.apply_metrics(m)
        if remember: touch.save_scale(self.paths,name)

    def fill_screen(self):
        """Open on most of the screen rather than a fixed 1050×760, which on a
        touch panel is a window in the corner of a display it could have used."""
        screen=self.screen() or QApplication.primaryScreen()
        if screen is None: self.resize(1280,860); return
        available=screen.availableGeometry()
        self.resize(min(1600,int(available.width()*0.92)),min(1050,int(available.height()*0.92)))
        self.setMinimumSize(min(880,available.width()),min(620,available.height()))
        self.move(available.center()-self.rect().center())

    @staticmethod
    def combo(options, default=None):
        """Label is shown, upstream key is carried as item data — never the label."""
        box=QComboBox()
        for value,label in options: box.addItem(label,value)
        if default is not None:
            index=box.findData(default)
            if index >= 0: box.setCurrentIndex(index)
        return touch.touchable_popup(box)

    @staticmethod
    def grouped_combo(groups, default=None):
        """Styles arrive grouped by upstream; the headings are inserted as
        disabled rows so the engine's own grouping survives in the UI."""
        box=QComboBox()
        for heading,options in groups:
            box.addItem(f"— {heading} —",None)
            item=box.model().item(box.count()-1)
            if item is not None: item.setEnabled(False)
            for value,label in options: box.addItem(f"   {label}",value)
        if default is not None:
            index=box.findData(default)
            if index >= 0: box.setCurrentIndex(index)
        return touch.touchable_popup(box)

    @staticmethod
    def chosen(box, fallback=""):
        value=box.currentData()
        return fallback if value is None else value

    def browse_image(self):
        filename,_=QFileDialog.getOpenFileName(self,"Reference image","","Images (*.png *.jpg *.jpeg *.webp)")
        if not filename: return
        self.image_path=Path(filename)
        # The name, not the path: a full path pushes the buttons beside it off
        # the pane, and the path is still there to hover or long-press for.
        self.image_label.setText(f"Image: {self.image_path.name}"); self.image_label.setToolTip(filename)
        self.remove_button.setEnabled(True); self.select(self.mode,"i2v")
    def remove_image(self):
        self.image_path=None; self.image_label.setText("No image — text to video"); self.image_label.setToolTip("")
        self.remove_button.setEnabled(False); self.select(self.mode,"t2v")
    @staticmethod
    def select(box,value):
        index=box.findData(value)
        if index >= 0: box.setCurrentIndex(index)
    def request(self):
        data=image_data_url(self.image_path) if self.image_path else None
        width,height=map(int,self.chosen(self.dimensions,"704x1216").split("x"))
        return PromptRequest(
            intent=self.intent.toPlainText().strip(),
            image_data_url=data,
            image_name=self.image_path.name if self.image_path else "",
            video_mode=self.chosen(self.mode,"i2v"),
            seconds=self.seconds.value(),
            fps=self.fps.value(),
            style=self.chosen(self.style,"off"),
            motion=self.chosen(self.motion,motion.DEFAULT),
            speech=self.speech_slider.value(),
            camera=self.chosen(self.camera,"off"),
            transition=self.chosen(self.transition,"off"),
            pov=self.chosen(self.pov,"off"),
            accent=self.chosen(self.accent,"off"),
            accent_strength=self.chosen(self.accent_strength,"natural"),
            dialogue=self.dialogue.value(),
            music=self.chosen(self.music,"off"),
            music_bg=self.music_bg.isChecked(),
            wardrobe=self.chosen(self.wardrobe,"auto"),
            undress=self.undress.isChecked(),
            lexicon=self.lexicon.toPlainText(),
            fmt=self.chosen(self.output_format,"flowing"),
            negative_extra=self.negative_extra.toPlainText(),
            # -1 means "a different one every time": resolved here, so the
            # casting upstream seeds and the sampler both get the same number.
            seed=self.seed.value() if self.seed.value() != RANDOM_SEED else draw_seed(),
            smart_negative=self.smart.isChecked(),
            output_width=width,
            output_height=height,
        )
    def generate(self):
        if not self.intent.toPlainText().strip(): QMessageBox.warning(self,"Missing intent","Enter a video intent first."); return
        if self.thread and self.thread.isRunning(): return
        try: request = self.request()
        except Exception as exc: QMessageBox.critical(self,"Image error",str(exc)); return
        if request.video_mode == "i2v" and request.image_data_url is None:
            QMessageBox.warning(self,"Image required","Image to video needs an attached image. Attach one, or switch to text to video."); return
        try: self.negative.setPlainText(self.engine.base_negative(request))
        except VisionUnavailable as exc: QMessageBox.critical(self,"Vision unavailable",str(exc)); return
        self.positive.clear(); self.generate_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.thread=QThread(self); worker=GenerationWorker(self.service,self.engine,request); worker.moveToThread(self.thread); self._worker=worker
        self.thread.started.connect(worker.run); worker.positive_chunk.connect(self.positive.insertPlainText); worker.positive_ready.connect(self.positive.setPlainText); worker.negative_ready.connect(self.negative.setPlainText); worker.status.connect(self.status.setText); worker.failed.connect(self.generation_failed); worker.finished.connect(self.thread.quit); worker.finished.connect(worker.deleteLater); self.thread.finished.connect(self.generation_done); self.thread.start()
    def generation_failed(self, message): self.status.setText("Generation failed"); QMessageBox.critical(self,"Generation failed",message)
    def generation_done(self): self.generate_button.setEnabled(True); self.cancel_button.setEnabled(False); self.thread.deleteLater(); self.thread=None; self._worker=None
    def cancel_generation(self):
        if getattr(self,"_worker",None): self._worker.cancel()
    def copy(self, source): source.selectAll(); source.copy()
    def copy_both(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(f"POSITIVE\n{self.positive.toPlainText()}\n\nNEGATIVE\n{self.negative.toPlainText()}")
    def save(self):
        filename,_=QFileDialog.getSaveFileName(self,"Save prompts","prompts.txt","Text (*.txt)")
        if filename: Path(filename).write_text(f"POSITIVE\n{self.positive.toPlainText()}\n\nNEGATIVE\n{self.negative.toPlainText()}\n",encoding="utf-8")
    def clear(self): self.intent.clear(); self.positive.clear(); self.negative.clear(); self.remove_image()
    def refresh_status(self):
        from prompt_master.core.config import read_json
        from prompt_master.core.models import GPU_MODE
        state=read_json(self.paths.data/"setup-state.json")
        # The mode is named unless the card simply holds the model, which is
        # what a device name on its own has always meant. An install predating
        # the setting records none, and reads as that same default.
        device=state.get('gpu_device_name',state.get('gpu_name','not configured'))
        mode=state.get('mode',GPU_MODE)
        if mode != GPU_MODE: device=f"{device} ({mode})"
        self.status.setText(f"Device: {device} · Model: {state.get('quantization','not configured')} · Server: {'running' if self.service.process.running else 'stopped'} · Generation: idle")
    def open_setup(self):
        self.service.stop(); wizard=SetupWizard(self.paths,self)
        if wizard.exec() and wizard.completed:
            self.paths=wizard.paths; self.service=InferenceService(self.paths); self.chat.rebind(self.paths); self.refresh_status()
    def closeEvent(self,event):
        if self.thread and self.thread.isRunning(): self.thread.quit(); self.thread.wait(3000)
        self.chat.shutdown(); self.service.stop(); event.accept()
