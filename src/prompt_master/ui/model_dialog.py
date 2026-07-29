"""Choosing a model of your own, and what it sees with.

One dialog behind one menu item. It is deliberately not a wizard: the questions
are a file, a projector, and which of the two modes should use it, and every one
of them has an answer already filled in.

Two things it says out loud, because they are the difference between this and
the model setup installs. Nothing here is hash-verified — there is no pinned
SHA-256 for a file this build has never heard of — and a model with no projector
is a model that cannot be sent images, which is a normal thing for a model to be
and worth knowing before the first attached still is refused.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressDialog,
                               QPushButton, QVBoxLayout, QWidget)

from prompt_master.core.library import CONVERSATION, MODEL_SUFFIX, PROMPT, ModelEntry
from prompt_master.provisioning import importer
from prompt_master.ui import touch

# What the projector box offers besides the files already in the models folder.
NO_PROJECTOR = "__none__"
BROWSE_PROJECTOR = "__browse__"

MODE_LABELS = ((PROMPT, "Prompt mode"), (CONVERSATION, "Conversation mode"))


class ModelDialog(QDialog):
    """Point a mode at a ``.gguf``, moving it into the models folder if needed."""

    def __init__(self, paths, library, metrics: dict, current_mode: str = PROMPT, parent=None):
        super().__init__(parent)
        self.paths, self.library, self.metrics = paths, library, metrics
        self.setWindowTitle("Model")
        self.chosen: ModelEntry | None = None      # what was installed, once accepted
        self.modes: list[str] = []
        self.browsed_projector: Path | None = None

        entry = library.entry_for(current_mode)
        column = QVBoxLayout(self)
        column.addWidget(self._intro())
        column.addWidget(touch.labelled("Model file", self._model_row(entry)))
        column.addWidget(touch.labelled("Vision projector", self._projector_box(entry)))
        self.keep = QCheckBox("Keep the original file where it is (copy instead of moving)")
        column.addWidget(self.keep)
        column.addWidget(self._modes_row(current_mode))
        column.addStretch(1)
        column.addLayout(self._buttons())
        self.resize(metrics["target"] * 16, metrics["target"] * 9)

    # ── layout ───────────────────────────────────────────────────────────────

    def _intro(self) -> QLabel:
        note = QLabel("A model of your own is used as it is: kept under its own name, and not "
                      "checked against anything, because this build pins no hash for it. The "
                      "model setup installed stays where it is and is what a mode falls back to.")
        # Plain rather than a field label: it is three lines of prose to read
        # once, not a heading over a control.
        note.setWordWrap(True)
        return note

    def _model_row(self, entry) -> QWidget:
        self.model_path = QLineEdit(str(self.paths.root / entry.model) if entry else "")
        self.model_path.setPlaceholderText(f"Path to a {MODEL_SUFFIX} model")
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_model)
        row.addWidget(self.model_path, 1)
        row.addWidget(browse)
        return holder

    def _projector_box(self, entry) -> QComboBox:
        box = touch.touchable_popup(QComboBox())
        box.addItem("None — this model cannot be sent images", NO_PROJECTOR)
        for relative in self.library.projectors():
            box.addItem(Path(relative).name, relative)
        for relative in self.library.models():
            # Any .gguf can be a projector; the ones named like one come first.
            box.addItem(f"{Path(relative).name} (in the models folder)", relative)
        box.addItem("Choose a file…", BROWSE_PROJECTOR)
        if entry is not None and entry.mmproj:
            index = box.findData(entry.mmproj)
            if index >= 0:
                box.setCurrentIndex(index)
        box.activated.connect(self._projector_chosen)
        self.projector = box
        return box

    def _modes_row(self, current_mode: str) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Use it for"))
        self.mode_boxes = {}
        for mode, label in MODE_LABELS:
            box = QCheckBox(label)
            box.setChecked(mode == current_mode)
            self.mode_boxes[mode] = box
            row.addWidget(box, 1)
        return holder

    def _buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        use = QPushButton("Use this model")
        use.setObjectName("primary")
        use.clicked.connect(self.accept)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(use)
        return row

    # ── choosing ─────────────────────────────────────────────────────────────

    def _browse_model(self):
        chosen, _ = QFileDialog.getOpenFileName(self, "Model file", self.model_path.text(),
                                                f"GGUF models (*{MODEL_SUFFIX})")
        if chosen:
            self.model_path.setText(chosen)

    def _projector_chosen(self, _index: int):
        if self.projector.currentData() != BROWSE_PROJECTOR:
            return
        chosen, _ = QFileDialog.getOpenFileName(self, "Vision projector", "",
                                                f"GGUF projectors (*{MODEL_SUFFIX})")
        if not chosen:
            self.projector.setCurrentIndex(0)
            return
        self.browsed_projector = Path(chosen)
        # Shown as itself, and carried as a marker until it is installed.
        self.projector.insertItem(1, f"{self.browsed_projector.name} (new)", BROWSE_PROJECTOR)
        self.projector.setCurrentIndex(1)

    # ── applying ─────────────────────────────────────────────────────────────

    def accept(self):
        given = self.model_path.text().strip().strip('"')
        if not given:
            QMessageBox.warning(self, "No model", "Choose a model file first."); return
        source = Path(given).expanduser()
        if source.suffix.casefold() != MODEL_SUFFIX:
            QMessageBox.warning(self, "Not a model", f"A model file ends in {MODEL_SUFFIX}."); return
        if not source.is_file():
            QMessageBox.warning(self, "Not found", f"{source} is not a file."); return
        modes = [mode for mode, _ in MODE_LABELS if self.mode_boxes[mode].isChecked()]
        if not modes:
            QMessageBox.warning(self, "No mode", "Tick at least one mode to use this model for."); return
        try:
            entry = self._install(source)
        except Exception as exc:
            QMessageBox.critical(self, "Could not use that model", str(exc)); return
        self.library.remember_projector(entry.model, entry.mmproj)
        for mode in modes:
            self.library.assign(mode, entry.model)
        self.chosen, self.modes = entry, modes
        super().accept()

    def _install(self, source: Path) -> ModelEntry:
        """Move or copy what was chosen into the models folder, with a progress bar."""
        models = self.paths.root / "models"
        dialog = QProgressDialog("Installing…", "", 0, 100, self)
        dialog.setWindowTitle("Model")
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setValue(0)

        def progress(done, total):
            dialog.setValue(int(done / max(total, 1) * 100)); QApplication.processEvents()
        try:
            dialog.setLabelText(f"Installing {source.name}…")
            model = importer.install_unverified(source, models, move=not self.keep.isChecked(),
                                                progress=progress)
            mmproj = ""
            data = self.projector.currentData()
            if data == BROWSE_PROJECTOR and self.browsed_projector is not None:
                dialog.setLabelText(f"Installing {self.browsed_projector.name}…")
                dialog.setValue(0)
                installed = importer.install_unverified(self.browsed_projector, models,
                                                        move=not self.keep.isChecked(),
                                                        progress=progress)
                mmproj = self._relative(installed)
            elif data not in (NO_PROJECTOR, BROWSE_PROJECTOR):
                mmproj = str(data)
        finally:
            dialog.close()
        return ModelEntry(self._relative(model), mmproj)

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.paths.root.resolve()).as_posix()
