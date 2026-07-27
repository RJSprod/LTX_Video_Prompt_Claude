from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (QCheckBox,QComboBox,QFileDialog,QFormLayout,QHBoxLayout,QLabel,QMainWindow,QMessageBox,QPlainTextEdit,QPushButton,QSpinBox,QDoubleSpinBox,QVBoxLayout,QWidget,QScrollArea)
import threading

from prompt_master.core.models import PromptRequest
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.prompt_engine import options as opt
from prompt_master.prompt_engine.adapter import PromptEngine, VisionUnavailable
from prompt_master.core.paths import AppPaths
from prompt_master.inference.service import InferenceService
from prompt_master.ui.setup_wizard import SetupWizard


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
            plan = self.engine.build(self.request, vision_available=True)
            self.status.emit(f"Generating positive prompt… ({plan.frames} frames, {plan.word_budget[0]}-{plan.word_budget[1]} words)")
            raw = client.stream_chat(plan.messages, plan.max_tokens, self.request.seed, self.positive_chunk.emit, self.cancelled)
            if self.cancelled.is_set(): self.status.emit("Generation cancelled"); return
            positive = self.engine.clean_positive(raw)
            if not positive.strip(): raise RuntimeError("The model returned an empty script.")
            self.positive_ready.emit(positive)
            auto = ""
            if self.request.smart_negative:
                self.status.emit("Negative pass…")
                auto = self.engine.run_smart_negative(positive, self._chat_stream(client))
            self.negative_ready.emit(self.engine.merge_negative(self.request, auto))
            self.status.emit("Server: running · Generation: complete")
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
    def __init__(self, paths: AppPaths | None = None):
        super().__init__(); self.paths = paths or AppPaths.discover(); self.service = InferenceService(self.paths); self.thread = None; self.setWindowTitle("Prompt Master Standalone"); self.resize(1050, 760); self.image_path: Path | None = None; self.engine = PromptEngine()
        settings_menu=self.menuBar().addMenu("Settings"); models_action=settings_menu.addAction("Models and Hardware…"); models_action.triggered.connect(self.open_setup)
        root=QWidget(); layout=QVBoxLayout(root); self.intent=QPlainTextEdit(); self.intent.setPlaceholderText("Describe the video you want to create…"); layout.addWidget(QLabel("Intent")); layout.addWidget(self.intent)
        image_row=QHBoxLayout(); self.image_label=QLabel("Text only"); browse=QPushButton("Browse image…"); remove=QPushButton("Remove image"); browse.clicked.connect(self.browse_image); remove.clicked.connect(self.remove_image); image_row.addWidget(self.image_label,1); image_row.addWidget(browse); image_row.addWidget(remove); layout.addLayout(image_row)
        d=opt.DEFAULTS
        form=QFormLayout()
        self.mode=self.combo(opt.VIDEO_MODES,d["video_mode"])
        self.seconds=QDoubleSpinBox(); self.seconds.setRange(1,60); self.seconds.setSingleStep(0.5); self.seconds.setValue(d["seconds"])
        self.fps=QSpinBox(); self.fps.setRange(8,60); self.fps.setValue(d["fps"])
        self.style=self.grouped_combo(opt.STYLES_GROUPED,d["style"])
        self.camera=self.combo(opt.CAMERAS,d["camera"])
        self.transition=self.combo(opt.TRANSITIONS,d["transition"])
        self.pov=self.combo(opt.POV,d["pov"])
        self.accent=self.combo(opt.ACCENTS,d["accent"])
        self.accent_strength=self.combo(opt.ACCENT_STRENGTHS,d["accent_strength"])
        self.dialogue=QSpinBox(); self.dialogue.setRange(0,100); self.dialogue.setValue(d["dialogue"]); self.dialogue.setSuffix("%")
        self.music=self.combo(opt.MUSIC,d["music"])
        self.music_bg=QCheckBox("Play low under the scene"); self.music_bg.setChecked(d["music_bg"])
        self.wardrobe=self.combo(opt.WARDROBE,d["wardrobe"])
        self.undress=QCheckBox("Undress sequence"); self.undress.setChecked(d["undress"])
        self.lexicon=QPlainTextEdit(); self.lexicon.setMaximumHeight(70); self.lexicon.setPlaceholderText("Name = description, one per line. Only names present in the intent are used.")
        self.output_format=self.combo(opt.OUTPUT_FORMATS,d["fmt"])
        self.dimensions=self.combo([("704x1216","704 × 1216 (portrait)"),("1216x704","1216 × 704 (landscape)"),("768x768","768 × 768 (square)"),("1920x1080","1920 × 1080"),("1080x1920","1080 × 1920")],f"{d['output_width']}x{d['output_height']}")
        self.seed=QSpinBox(); self.seed.setRange(0,2**31-1); self.seed.setValue(d["seed"])
        self.negative_extra=QPlainTextEdit(); self.negative_extra.setMaximumHeight(50)
        self.smart=QCheckBox("Second pass over the finished script"); self.smart.setChecked(d["smart_negative"])
        for label,widget in [("Video mode",self.mode),("Duration (seconds)",self.seconds),("FPS",self.fps),("Dimensions",self.dimensions),("Seed",self.seed),("Style",self.style),("Camera",self.camera),("Transition",self.transition),("First person",self.pov),("Accent",self.accent),("Accent strength",self.accent_strength),("Dialogue / talk",self.dialogue),("Music",self.music),("Music background",self.music_bg),("Wardrobe",self.wardrobe),("Undress",self.undress),("Lexicon",self.lexicon),("Output format",self.output_format),("Extra negative terms",self.negative_extra),("Smart negative",self.smart)]: form.addRow(label,widget)
        layout.addLayout(form); actions=QHBoxLayout(); self.generate_button=QPushButton("Generate"); self.generate_button.clicked.connect(self.generate); self.cancel_button=QPushButton("Cancel"); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel_generation); clear=QPushButton("Clear"); clear.clicked.connect(self.clear); actions.addWidget(self.generate_button); actions.addWidget(self.cancel_button); actions.addWidget(clear); layout.addLayout(actions)
        self.positive=QPlainTextEdit(); self.negative=QPlainTextEdit(); layout.addWidget(QLabel("Positive prompt")); layout.addWidget(self.positive); layout.addWidget(QLabel("Negative prompt")); layout.addWidget(self.negative)
        copies=QHBoxLayout()
        for label,source in [("Copy Positive",self.positive),("Copy Negative",self.negative)]: button=QPushButton(label); button.clicked.connect(lambda _=False,s=source: self.copy(s)); copies.addWidget(button)
        both=QPushButton("Copy Both"); both.clicked.connect(self.copy_both); copies.addWidget(both)
        save=QPushButton("Save .txt…"); save.clicked.connect(self.save); copies.addWidget(save); layout.addLayout(copies); self.status=QLabel("GPU: not configured · Model: not configured · Server: stopped · Generation: idle"); layout.addWidget(self.status)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(root); self.setCentralWidget(scroll); self.refresh_status()

    @staticmethod
    def combo(options, default=None):
        """Label is shown, upstream key is carried as item data — never the label."""
        box=QComboBox()
        for value,label in options: box.addItem(label,value)
        if default is not None:
            index=box.findData(default)
            if index >= 0: box.setCurrentIndex(index)
        return box

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
        return box

    @staticmethod
    def chosen(box, fallback=""):
        value=box.currentData()
        return fallback if value is None else value

    def browse_image(self):
        filename,_=QFileDialog.getOpenFileName(self,"Reference image","","Images (*.png *.jpg *.jpeg *.webp)")
        if filename: self.image_path=Path(filename); self.image_label.setText(filename); self.select(self.mode,"i2v")
    def remove_image(self): self.image_path=None; self.image_label.setText("Text only"); self.select(self.mode,"t2v")
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
            seed=self.seed.value(),
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
        state=read_json(self.paths.data/"setup-state.json")
        self.status.setText(f"GPU: {state.get('gpu_device_name',state.get('gpu_name','not configured'))} · Model: {state.get('quantization','not configured')} · Server: {'running' if self.service.process.running else 'stopped'} · Generation: idle")
    def open_setup(self):
        self.service.stop(); wizard=SetupWizard(self.paths,self)
        if wizard.exec() and wizard.completed:
            self.paths=wizard.paths; self.service=InferenceService(self.paths); self.refresh_status()
    def closeEvent(self,event):
        if self.thread and self.thread.isRunning(): self.thread.quit(); self.thread.wait(3000)
        self.service.stop(); event.accept()
