"""Making a character, and saying who you are.

Two dialogs, both opened from the chat view and both built only when they are
opened — the window they belong to is measured by the touch tests, and a dialog
that exists before it is asked for is a dialog whose controls are counted as
part of a window nobody can see them in.

The character editor is oobabooga's, laid out for a finger: the list of
characters down one side, the fields that make one down the other, and the four
things done to a character — new, import, delete, save — on a row of their own
along the bottom. The fields are exactly the ones that survive a round trip
through oobabooga's own editor: name, picture, context and greeting. How a
character is sampled is not here but in the chat view's settings panel, beside
the conversation it changes, and this dialog carries those values through
untouched so that editing a name cannot reset a temperature.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QPushButton, QVBoxLayout)

from prompt_master.chat.characters import (AVATAR_SUFFIXES, IMPORTABLE_SUFFIXES, Character,
                                           CharacterStore, Persona)
from prompt_master.ui import touch


class CharacterDialog(QDialog):
    """Create, edit, import and delete the characters in the store."""

    def __init__(self, store: CharacterStore, metrics: dict, parent=None,
                 selected: str | None = None):
        super().__init__(parent)
        self.store, self.metrics = store, metrics
        self.setWindowTitle("Characters")
        self.current: str | None = None      # the name the open character is saved under
        self.avatar_path: Path | None = None
        self.dirty = False
        self.chosen: str | None = None       # what the chat view should select on close

        body = QHBoxLayout(self)
        body.addLayout(self._list_column(), 2)
        body.addLayout(self._form_column(), 5)
        self.refresh(selected)
        self.resize(metrics["target"] * 20, metrics["target"] * 14)

    # ── layout ───────────────────────────────────────────────────────────────

    def _list_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.addWidget(_heading("Characters"))
        self.list = QListWidget()
        self.list.setUniformItemSizes(True)
        touch.flickable(self.list)
        self.list.currentItemChanged.connect(self._selection_changed)
        column.addWidget(self.list, 1)
        row = QHBoxLayout()
        new = QPushButton("New"); new.clicked.connect(self.new_character)
        importer = QPushButton("Import…"); importer.clicked.connect(self.import_character)
        row.addWidget(new, 1); row.addWidget(importer, 1)
        column.addLayout(row)
        return column

    def _form_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        self.name = QLineEdit(); self.name.setPlaceholderText("Character name")
        self.context = QPlainTextEdit()
        self.context.setPlaceholderText(
            "Who they are — personality, background, how they speak. {{char}} and {{user}} "
            "are replaced with the character's name and yours.")
        self.greeting = QPlainTextEdit()
        self.greeting.setPlaceholderText("The first thing they say when a chat starts.")
        for editor in (self.context, self.greeting):
            touch.flickable(editor)
        for widget in (self.name, self.context, self.greeting):
            signal = widget.textChanged
            signal.connect(self._touched)

        column.addWidget(touch.labelled("Name", self.name))
        column.addLayout(self._picture_row())
        column.addWidget(touch.labelled("Context", self.context), 3)
        column.addWidget(touch.labelled("Greeting", self.greeting), 2)
        column.addLayout(self._buttons())
        return column

    def _picture_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.avatar = QLabel()
        self.avatar.setFixedSize(QSize(self.metrics["target"] * 2, self.metrics["target"] * 2))
        self.avatar.setFrameShape(QFrame.Shape.StyledPanel)
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        choose = QPushButton("Picture…"); choose.clicked.connect(self.browse_avatar)
        clear = QPushButton("Remove"); clear.clicked.connect(self.clear_avatar)
        row.addWidget(self.avatar)
        row.addWidget(choose, 1)
        row.addWidget(clear, 1)
        return row

    def _buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        delete = QPushButton("Delete"); delete.clicked.connect(self.delete_character)
        save = QPushButton("Save"); save.setObjectName("primary"); save.clicked.connect(self.save_character)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        row.addWidget(delete); row.addStretch(1); row.addWidget(close); row.addWidget(save)
        return row

    # ── the list ─────────────────────────────────────────────────────────────

    def refresh(self, selected: str | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for name in self.store.names():
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(0, self.metrics["target"]))
            avatar = self.store.avatar_for(name)
            if avatar is not None:
                item.setIcon(QPixmap(str(avatar)).scaled(
                    self.metrics["target"], self.metrics["target"],
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.list.addItem(item)
        self.list.blockSignals(False)
        if selected:
            self.select(selected)
        elif self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.new_character()

    def select(self, name: str) -> None:
        matches = self.list.findItems(name, Qt.MatchFlag.MatchExactly)
        if matches:
            self.list.setCurrentItem(matches[0])

    def _selection_changed(self, current, previous) -> None:
        if previous is not None and not self._resolve_unsaved(previous.text()):
            self.list.blockSignals(True)
            self.list.setCurrentItem(previous)
            self.list.blockSignals(False)
            return
        if current is None:
            return
        try:
            self.show_character(self.store.load(current.text()))
        except (OSError, ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "Character", f"Could not open that character: {exc}")

    def _resolve_unsaved(self, name: str) -> bool:
        """True to leave the open character; False to stay on it."""
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", f"Save your changes to {name or 'this character'}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_character()
        self.dirty = False
        return True

    # ── the form ─────────────────────────────────────────────────────────────

    def show_character(self, character: Character) -> None:
        # The whole character is held, not just the fields with boxes: how it
        # replies is tuned in the chat view's own panel, and editing a name here
        # must not reset a temperature set there.
        self.loaded = character
        self.current = character.name
        self.name.setText(character.name)
        self.context.setPlainText(character.context)
        self.greeting.setPlainText(character.greeting)
        self.avatar_path = self.store.avatar_for(character.name)
        self._show_avatar()
        self.dirty = False

    def collect(self) -> Character:
        base = getattr(self, "loaded", Character(name=""))
        return Character(
            name=self.name.text().strip(),
            context=self.context.toPlainText().strip(),
            greeting=self.greeting.toPlainText().strip(),
            temperature=base.temperature,
            top_p=base.top_p,
            max_reply_tokens=base.max_reply_tokens,
            seed=base.seed,
            system=base.system,
        )

    def new_character(self) -> None:
        if not self._resolve_unsaved(self.current or ""):
            return
        self.list.blockSignals(True)
        self.list.setCurrentItem(None)
        self.list.blockSignals(False)
        self.show_character(Character(name=""))
        self.current = None
        self.name.setFocus()

    def save_character(self) -> bool:
        character = self.collect()
        if not character.name:
            QMessageBox.warning(self, "Name needed", "Give the character a name first.")
            return False
        if self.current is None and self.store.exists(character.name):
            QMessageBox.warning(self, "Name taken", f"A character called {character.name} already exists.")
            return False
        try:
            self.store.save(character, previous_name=self.current)
            if self.avatar_path is not None and self.avatar_path.parent != self.store.directory:
                self.store.set_avatar(character.name, self.avatar_path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save", str(exc))
            return False
        self.dirty = False
        self.current = self.chosen = character.name
        self.refresh(character.name)
        return True

    def delete_character(self) -> None:
        if self.current is None:
            return
        if QMessageBox.question(self, "Delete character",
                                f"Delete {self.current}? Chats with them are kept.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.store.delete(self.current)
        self.current, self.dirty = None, False
        if self.chosen == self.current:
            self.chosen = None
        self.refresh()

    def import_character(self) -> None:
        pattern = " ".join(f"*{suffix}" for suffix in IMPORTABLE_SUFFIXES)
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import a character", "", f"Characters ({pattern})")
        if not filename:
            return
        try:
            character = self.store.import_file(Path(filename))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not import", str(exc))
            return
        self.dirty = False
        self.chosen = character.name
        self.refresh(character.name)

    def browse_avatar(self) -> None:
        pattern = " ".join(f"*{suffix}" for suffix in AVATAR_SUFFIXES)
        filename, _ = QFileDialog.getOpenFileName(self, "Character picture", "", f"Images ({pattern})")
        if not filename:
            return
        self.avatar_path = Path(filename)
        self._show_avatar()
        self._touched()

    def clear_avatar(self) -> None:
        if self.current:
            self.store.set_avatar(self.current, None)
        self.avatar_path = None
        self._show_avatar()

    def _show_avatar(self) -> None:
        if self.avatar_path is None or not self.avatar_path.is_file():
            self.avatar.setPixmap(QPixmap())
            self.avatar.setText("No\npicture")
            return
        self.avatar.setText("")
        self.avatar.setPixmap(QPixmap(str(self.avatar_path)).scaled(
            self.avatar.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _touched(self, *_args) -> None:
        self.dirty = True

    def accept(self) -> None:
        if not self._resolve_unsaved(self.current or ""):
            return
        super().accept()


class PersonaDialog(QDialog):
    """Who the character is replying to — optional, and empty by default."""

    def __init__(self, persona: Persona, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About you")
        column = QVBoxLayout(self)
        note = QLabel("Leave this empty and characters reply to an unnamed \"you\". Fill it in "
                      "and they know your name and can refer to what you tell them here.")
        note.setWordWrap(True)
        column.addWidget(note)
        self.name = QLineEdit(persona.name); self.name.setPlaceholderText("Your name — optional")
        self.description = QPlainTextEdit(persona.description)
        self.description.setPlaceholderText(
            "Anything the character should know about you — optional.")
        touch.flickable(self.description)
        column.addWidget(touch.labelled("Name", self.name))
        column.addWidget(touch.labelled("About you", self.description), 1)
        row = QHBoxLayout()
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        save = QPushButton("Save"); save.setObjectName("primary"); save.clicked.connect(self.accept)
        row.addStretch(1); row.addWidget(cancel); row.addWidget(save)
        column.addLayout(row)

    def persona(self) -> Persona:
        return Persona(name=self.name.text().strip(),
                       description=self.description.toPlainText().strip())


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label

