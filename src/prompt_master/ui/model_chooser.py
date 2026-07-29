"""Choosing which GGUF runs, without re-running setup.

Two paths and one sentence about what the second one is for. The dialog is
deliberately not a wizard: nothing here downloads, verifies or moves anything,
so there are no steps to walk through — see ``inference.model_choice``, which
is where the rule about what may be recorded lives.

The projector box is the part worth laying out carefully. It is optional, and a
box that is optional and empty looks identical to one that is optional and
forgotten, so it says what leaving it empty costs rather than only that it may
be left empty. When a file that looks like a projector is sitting beside the
chosen model — which is where it usually is — it is offered into the box as a
suggestion, and the box stays editable so that suggestion can be emptied.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QVBoxLayout)

from prompt_master.core.paths import AppPaths
from prompt_master.inference import model_choice

# What the file dialogs filter on. Both halves of a multimodal pair are GGUF.
GGUF_FILTER = f"GGUF models (*{model_choice.MODEL_SUFFIX});;All files (*)"

PROJECTOR_NOTE = (
    "Optional. A vision projector is what lets a model be shown a picture — it is what "
    "image-to-video prompts and pictures in a chat go through. Leave it empty and the model "
    "still runs, answering text only, and anything carrying an image is refused rather than "
    "quietly answered blind.")

MODEL_NOTE = (
    "The file is read where it is. Nothing is copied, moved or downloaded, and the runtime and "
    "the device this install is set up with do not change. The model loads on your next "
    "generation.")


class ModelDialog(QDialog):
    """Which weights to run, and which projector — if any — to run them with."""

    def __init__(self, paths: AppPaths, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.setWindowTitle("Which model runs")

        column = QVBoxLayout(self)
        self.model = QLineEdit()
        self.model.setPlaceholderText(f"Full path to a {model_choice.MODEL_SUFFIX} model file")
        self.mmproj = QLineEdit()
        self.mmproj.setPlaceholderText("Full path to an mmproj file — optional")
        column.addLayout(self._row("Model file", self.model, self._browse_model))
        column.addWidget(_note(MODEL_NOTE))
        column.addLayout(self._row("Vision projector (mmproj)", self.mmproj, self._browse_mmproj,
                                   clear=True))
        column.addWidget(_note(PROJECTOR_NOTE))
        column.addStretch(1)
        column.addLayout(self._buttons())
        self._show_current()
        # Wide enough for a path: these are long, and a box that shows the last
        # thirty characters of one is a box nobody can check.
        self.resize(max(720, self.sizeHint().width()), self.sizeHint().height())

    # ── layout ───────────────────────────────────────────────────────────────

    def _row(self, caption: str, field: QLineEdit, browse, clear: bool = False) -> QVBoxLayout:
        column = QVBoxLayout()
        label = QLabel(caption)
        label.setObjectName("fieldLabel")
        label.setBuddy(field)
        column.addWidget(label)
        row = QHBoxLayout()
        row.addWidget(field, 1)
        chooser = QPushButton("Browse…")
        chooser.clicked.connect(browse)
        row.addWidget(chooser)
        if clear:
            empty = QPushButton("None")
            empty.setToolTip("Run this model with no vision projector")
            empty.clicked.connect(self.mmproj.clear)
            row.addWidget(empty)
        column.addLayout(row)
        return column

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

    def _show_current(self) -> None:
        current = model_choice.recorded(self.paths)
        if current is None:
            return
        self.model.setText(str(current.model))
        if current.mmproj is not None:
            self.mmproj.setText(str(current.mmproj))

    # ── choosing ─────────────────────────────────────────────────────────────

    def _browse_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Model file", self._start_in(self.model),
                                                  GGUF_FILTER)
        if not filename:
            return
        self.model.setText(filename)
        if not self.mmproj.text().strip():
            beside = model_choice.projector_beside(Path(filename))
            if beside is not None:
                self.mmproj.setText(str(beside))

    def _browse_mmproj(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Vision projector",
                                                  self._start_in(self.mmproj) or
                                                  self._start_in(self.model), GGUF_FILTER)
        if filename:
            self.mmproj.setText(filename)

    def _start_in(self, field: QLineEdit) -> str:
        """The folder the file dialog opens on: the one already named, if any."""
        typed = field.text().strip().strip('"')
        if not typed:
            return ""
        path = Path(typed).expanduser()
        return str(path.parent if path.parent.is_dir() else "")

    # ── what the caller reads ────────────────────────────────────────────────

    def model_path(self) -> Path:
        return Path(self.model.text().strip().strip('"')).expanduser()

    def mmproj_path(self) -> Path | None:
        typed = self.mmproj.text().strip().strip('"')
        return Path(typed).expanduser() if typed else None

    def accept(self) -> None:
        """Check both paths here, so a refusal lands on the box that caused it."""
        typed = self.model.text().strip().strip('"')
        if not typed:
            QMessageBox.warning(self, "No model", "Choose a model file to run.")
            return
        model = self.model_path()
        if not model.is_file():
            QMessageBox.warning(self, "No such file", f"There is no file at {model}")
            return
        mmproj = self.mmproj_path()
        if mmproj is not None and not mmproj.is_file():
            QMessageBox.warning(self, "No such file",
                                f"There is no file at {mmproj}\n\nLeave the projector empty to "
                                "run this model without one.")
            return
        if mmproj is None and QMessageBox.question(
                self, "No vision projector",
                f"{model.name} will run with no vision projector, so it cannot be shown a "
                "picture: image-to-video prompts and pictures in a chat will be refused.\n\n"
                "Run it anyway?") != QMessageBox.StandardButton.Yes:
            return
        super().accept()


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    label.setWordWrap(True)
    return label
