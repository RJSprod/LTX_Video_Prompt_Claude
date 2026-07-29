"""Conversation mode: the chat view, laid out for a finger.

The shape is oobabooga's — a character, a transcript, and a box to type in —
with its hover menus turned into things a finger can actually hit. Five
decisions are worth stating, because they are what a touch screen changes:

*The transcript reads as a conversation, not as a log.* Bubbles held to their
own side, held to a fraction of the width, and carrying no name label at all:
the side a bubble sits on is what says who said it, and the character's picture
appears once at the start of a run of their replies rather than beside every
one of them.

*Every message has its own menu, and a tap is what reveals it.* oobabooga puts
the message actions in one hover menu that applies to the last reply. A finger
cannot hover, and half of these actions — edit, branch, delete from here — are
about a message somewhere up the transcript rather than the last one. So a tap
on a bubble shows a ``⋯`` on that message alone; a drag still scrolls the
transcript, and a drag across words still selects them.

*Regenerating pages rather than replaces.* The versions of a reply live on the
message (see ``chat.history``), so a regenerate that came back worse is undone
by tapping ``◀`` instead of by regenerating until luck returns. That pager is
the one thing shown without a tap, because "2/3" is what says an earlier
attempt is still there.

*The newest message stays in view until you leave it.* The transcript follows
what arrives only while it is already at the end. Scroll up to read something
and it stops chasing; scroll back to the bottom and it resumes. Sending a
message, opening a chat and starting one all count as going back to the end.

*The transcript is rebuilt, not patched.* Editing, deleting, branching and
paging all change what the transcript is, and one path that renders the
conversation from scratch cannot disagree with the conversation the way a dozen
in-place mutations eventually would. Streaming is the single exception: the
reply being written updates its own bubble, because rebuilding sixty widgets per
token is not a thing to do to a tablet.
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QKeySequence, QPainter, QPainterPath, QPixmap, QShortcut
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

from prompt_master.chat import prompt
from prompt_master.chat.characters import (Character, CharacterStore, Persona, load_persona,
                                           save_persona)
from prompt_master.chat.history import ASSISTANT, USER, ChatStore, Conversation, Message
from prompt_master.core.config import atomic_write_json, read_json
from prompt_master.core.models import RANDOM_SEED, draw_seed
from prompt_master.imaging.preprocess import image_data_url
from prompt_master.ui import touch
from prompt_master.ui.character_editor import CharacterDialog, PersonaDialog

# Where the chat view remembers what was open. Beside the display size, and for
# the same reason: reopening the app on the chat you were having is the whole
# of what "remembered" has to mean here.
STATE_FILE = "chat-ui.json"

# The context window llama-server is started with when setup recorded none.
DEFAULT_CONTEXT = 8192

# How near the end counts as being at it. A scroll bar rarely lands exactly on
# its maximum, and a few pixels short of the bottom is still the bottom to the
# person looking at it.
STICKY_MARGIN = 24

# How long after the last change to the settings panel it is written to the
# character. Long enough that holding + on a stepper is one write rather than
# forty, and short enough that letting go of it and looking at the panel shows
# the change already saved.
SETTINGS_WRITE_DELAY = 600

# What the settings panel says about itself. It is the panel's own line rather
# than the window's status bar because it is about the panel, and because the
# status bar is busy saying what the model is doing.
SETTINGS_SAVED = "Saved with the character."
SETTINGS_UNSAVED = "Not saved yet — saving…"


class ChatWorker(QObject):
    """One reply, streamed off the UI thread."""

    chunk = Signal(str)
    completed = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service, messages, needs_vision, temperature, top_p, max_tokens, seed):
        super().__init__()
        self.service, self.messages, self.needs_vision = service, messages, needs_vision
        self.temperature, self.top_p = temperature, top_p
        self.max_tokens, self.seed = max_tokens, seed
        self.cancelled = threading.Event()

    @Slot()
    def cancel(self) -> None:
        self.cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            client = self.service.client(self.needs_vision)
            text = client.stream_chat(self.messages, self.max_tokens, self.seed,
                                      self.chunk.emit, self.cancelled,
                                      temperature=self.temperature, top_p=self.top_p)
            self.completed.emit(text)
        except Exception as exc:                      # surfaced, never swallowed
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MessageBubble(QWidget):
    """One turn: a bubble on its own side of the transcript.

    The widget is the whole row rather than the bubble itself — the bubble sits
    inside it, held against one edge and stopped from growing past ``WIDEST`` of
    the width, because a line of text that runs the full width of a tablet is a
    line that has to be tracked back across the screen to read the next one.

    What is *not* on the row is as deliberate as what is. No name over every
    message: the side it sits on says who said it, and a name repeated down the
    whole transcript is the thing that made the first version look like a log
    file rather than a conversation. The picture appears once at the start of a
    run of replies, and the ``⋯`` appears when the bubble is tapped.
    """

    # The fraction of the transcript a bubble may fill.
    WIDEST = 0.72
    # How far a finger may travel and still be a tap rather than a drag or a
    # text selection. A flick through the transcript must not open a menu.
    TAP_SLOP = 12

    def __init__(self, page: "ChatPage", message: Message, index: int, opens_run: bool = True):
        super().__init__()
        self.page, self.message, self.index = page, message, index
        self.mine = message.role == USER
        self.editor: QPlainTextEdit | None = None
        self._press_at = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, page.metrics["gap"] if opens_run else 2, 0, 0)
        row.setSpacing(round(page.metrics["gap"] / 2))
        # ``stack`` has no widget of its own until it joins ``row`` below, so
        # anything put in it stays unparented until then. Everything built here
        # is therefore parented to the bubble explicitly, and nothing is made
        # visible before this constructor has finished assembling the row: a
        # widget shown while it has no parent *is* a top-level window, and Qt
        # duly puts one on the screen — a bare white rectangle that flashes up
        # and vanishes the instant the layout claims it.
        stack = QVBoxLayout()
        stack.setSpacing(2)
        stack.addWidget(self.frame_for(message))
        stack.addWidget(self.actions_for(message))
        if self.mine:
            row.addStretch(1)
            row.addLayout(stack)
        else:
            # Level with the first line of the bubble rather than centred on
            # the whole message, which on a long reply floats it in mid-air.
            row.addWidget(self.face(opens_run), 0, Qt.AlignmentFlag.AlignTop)
            row.addLayout(stack)
            row.addStretch(1)
        # Now that everything is in the tree: the ⋯ waits for a tap, and the row
        # under the bubble is there only for a pager.
        self.set_revealed(False)
        self.set_max_width(page.bubble_width())

    def frame_for(self, message: Message) -> QFrame:
        """The bubble: the picture if there is one, then the words."""
        self.frame = QFrame(self)
        self.frame.setObjectName("bubbleYou" if self.mine else "bubbleThem")
        pad = self.page.metrics["pad"]
        column = QVBoxLayout(self.frame)
        column.setContentsMargins(pad, round(pad * 0.7), pad, round(pad * 0.7))
        column.setSpacing(round(pad / 2))
        if message.image:
            column.addWidget(self.picture())
        self.body = QLabel(message.text or "…")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.body.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        column.addWidget(self.body)
        self.column = column
        # Tapping anywhere on the bubble reveals its menu, so both the frame
        # and the label it is filled with have to be watched.
        for widget in (self.frame, self.body):
            widget.installEventFilter(self)
        return self.frame

    def face(self, opens_run: bool) -> QWidget:
        """The character's picture, once per run of their messages.

        A holder of the same width is left behind on the messages that follow,
        so a run of replies stays in one column instead of stepping sideways.
        """
        side = self.page.metrics["target"]
        holder = QLabel()
        holder.setFixedSize(QSize(side, side))
        avatar = self.page.avatar_pixmap() if opens_run else None
        if avatar is not None and not avatar.isNull():
            holder.setPixmap(avatar)
            holder.setToolTip(self.page.speaker_name(self.message.role))
        return holder

    def actions_for(self, message: Message) -> QWidget:
        """The row under the bubble: the version pager, and the ``⋯``.

        The pager is shown whenever a reply has more than one version, because
        "2/3" is the only thing that says an earlier attempt is still there.
        The menu button is not: it appears on the message being tapped.

        Neither is decided here. Both are ``set_revealed``'s to say, and it is
        called once the row exists — see the constructor.
        """
        holder = QWidget(self)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(round(self.page.metrics["gap"] / 2))
        if self.mine:
            row.addStretch(1)
        self.pager = self.versions() if len(message.versions) > 1 else None
        if self.pager is not None:
            row.addWidget(self.pager)
        self.actions_button = QPushButton("⋯")
        self.actions_button.setObjectName("bubbleAction")
        self.actions_button.setToolTip("What to do with this message")
        self.actions_button.clicked.connect(self.open_menu)
        row.addWidget(self.actions_button)
        if not self.mine:
            row.addStretch(1)
        self.actions_row = holder
        return holder

    def versions(self) -> QWidget:
        """The pager a regenerate leaves behind."""
        holder = QWidget(self)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        back, forward = QPushButton("◀"), QPushButton("▶")
        for button in (back, forward):
            button.setObjectName("bubbleAction")
        count = QLabel(f"{self.message.active + 1}/{len(self.message.versions)}")
        count.setObjectName("fieldLabel")
        back.setEnabled(self.message.active > 0)
        forward.setEnabled(self.message.active < len(self.message.versions) - 1)
        back.clicked.connect(lambda: self.page.show_version(self.index, self.message.active - 1))
        forward.clicked.connect(lambda: self.page.show_version(self.index, self.message.active + 1))
        for widget in (back, count, forward):
            row.addWidget(widget)
        return holder

    def picture(self) -> QLabel:
        label = QLabel()
        pixmap = _pixmap_from_data_url(self.message.image)
        if pixmap is None:
            label.setText(f"[{self.message.image_name or 'image'}]")
            return label
        side = self.page.metrics["target"] * 4
        label.setPixmap(pixmap.scaled(QSize(side, side), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        label.setToolTip(self.message.image_name)
        label.installEventFilter(self)
        return label

    def set_text(self, text: str) -> None:
        """Used while a reply streams, where a rebuild per token is not on."""
        self.body.setText(text or "…")
        self.fit_body()

    def set_max_width(self, width: int) -> None:
        self.frame.setMaximumWidth(max(self.page.metrics["target"] * 4, width))
        self.fit_body()

    def fit_body(self) -> None:
        """Make the bubble as wide as its longest line, and no wider.

        A word-wrapped label asks its layout for very little, so a bubble left
        to its own size hint collapses into a narrow column of two-word lines
        with the rest of the row empty beside it. The width the text actually
        wants is measured here — the longest line it has, capped at the widest
        a bubble may be — and asked for as a minimum, which is what lets a short
        message stay short and a long one fill the space it is allowed.
        """
        inside = self.frame.maximumWidth() - 2 * self.page.metrics["pad"] - 4
        metrics = self.body.fontMetrics()
        longest = max((metrics.horizontalAdvance(line)
                       for line in self.body.text().split("\n")), default=0)
        self.body.setMinimumWidth(max(0, min(longest + 2, inside)))

    def set_revealed(self, revealed: bool) -> None:
        """Show or hide this message's menu button."""
        self.actions_button.setVisible(revealed)
        self.actions_row.setVisible(revealed or self.pager is not None)

    def eventFilter(self, watched, event) -> bool:
        """A tap on the bubble reveals its menu; a drag still scrolls or selects."""
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            self._press_at = event.globalPosition().toPoint()
        elif kind == QEvent.Type.MouseButtonRelease and self._press_at is not None:
            travelled = (event.globalPosition().toPoint() - self._press_at).manhattanLength()
            self._press_at = None
            if travelled <= self.TAP_SLOP and not self.body.selectedText():
                # Which message is open is the page's to know: a widget's own
                # visibility is false for every widget in a window that has not
                # been shown, which is not what "already open" means.
                self.page.reveal(-1 if self.page.revealed == self.index else self.index)
        return False                      # watched, never swallowed

    # ── the menu ─────────────────────────────────────────────────────────────

    def open_menu(self) -> None:
        menu = QMenu(self)
        page, index = self.page, self.index
        last = index == len(page.conversation.messages) - 1 if page.conversation else False
        menu.addAction("Edit").triggered.connect(self.start_edit)
        menu.addAction("Copy").triggered.connect(lambda: page.copy_text(self.message.text))
        if self.message.role == ASSISTANT:
            menu.addAction("Regenerate").triggered.connect(lambda: page.regenerate(index))
            if last:
                menu.addAction("Continue").triggered.connect(lambda: page.continue_reply(index))
            if len(self.message.versions) > 1:
                menu.addAction("Delete this version").triggered.connect(
                    lambda: page.drop_version(index))
        else:
            menu.addAction("Send again from here").triggered.connect(lambda: page.resend(index))
        menu.addSeparator()
        menu.addAction("Branch from here").triggered.connect(lambda: page.branch(index))
        menu.addAction("Delete message").triggered.connect(lambda: page.delete_message(index))
        menu.addAction("Delete from here").triggered.connect(lambda: page.delete_from(index))
        # Under the button that opened it, which is where the finger already is.
        menu.exec(self.actions_button.mapToGlobal(self.actions_button.rect().bottomLeft()))

    def start_edit(self) -> None:
        if self.editor is not None:
            return
        self.body.hide()
        self.editor = QPlainTextEdit(self.message.text)
        self.editor.setMinimumHeight(self.page.metrics["target"] * 3)
        touch.flickable(self.editor)
        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        save = QPushButton("Save")
        cancel.clicked.connect(self.cancel_edit)
        save.clicked.connect(self.commit_edit)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(save)
        self.edit_row = row
        self.column.addWidget(self.editor)
        self.column.addLayout(row)
        self.editor.setFocus()

    def commit_edit(self) -> None:
        if self.editor is None:
            return
        self.page.edit_message(self.index, self.editor.toPlainText())

    def cancel_edit(self) -> None:
        self.page.render()


class ChatPage(QWidget):
    """The whole of conversation mode.

    Three of the rows here are optional, and the window's View menu is what
    turns them off: on a small screen, a chat that has been set up is mostly
    transcript, and the two drop-downs that chose the character and the chat are
    a row each that only matter when they are being changed.
    """

    # The rows the View menu can hide, and what it calls them.
    BARS = (("character", "“Talking to” bar"), ("chat", "Chat bar"),
            ("actions", "Quick actions"))

    def __init__(self, paths, service_provider, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.service_provider = service_provider
        self.characters = CharacterStore.from_paths(paths)
        self.chats = ChatStore.from_paths(paths)
        self.persona: Persona = load_persona(paths)
        self.character: Character | None = None
        self.conversation: Conversation | None = None
        self.attachment: Path | None = None
        self.metrics = touch.metrics(1.0)
        self.thread: QThread | None = None
        self._worker: ChatWorker | None = None
        self._streaming_index = -1
        self._streaming_prefix = ""
        self._into_input = False
        self._join_space = False
        self.bubbles: list[MessageBubble] = []
        self._avatar: QPixmap | None = None
        # Which message is showing its menu, and whether the transcript is
        # following what arrives. Both are about where you are looking, so both
        # are reset by anything that changes what you are looking at.
        self.revealed = -1
        self.pinned = True
        # The settings panel writes itself back a moment after it stops being
        # touched, so a value that was changed is a value that is saved without
        # anything having to be pressed. Built before the panel it belongs to.
        self._settings_problem = ""
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(SETTINGS_WRITE_DELAY)
        self._settings_timer.timeout.connect(self.persist_settings)

        column = QVBoxLayout(self)
        self.bars = {"character": _bar(self.character_row()),
                     "chat": _bar(self.chat_row())}
        column.addWidget(self.bars["character"])
        column.addWidget(self.bars["chat"])
        column.addWidget(self.middle(), 1)
        self.bars["actions"] = _bar(self.quick_actions())
        column.addWidget(self.bars["actions"])
        column.addLayout(self.input_row())
        column.addWidget(self.status_label())
        for key, _label in self.BARS:
            self.bars[key].setVisible(self.bar_visible(key))

        send = QShortcut(QKeySequence("Ctrl+Return"), self)
        # Scoped to this page: the same window holds prompt mode, and a chord
        # pressed there must not send a half-written chat message.
        send.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        send.activated.connect(self.send)
        self.reload_characters()

    # ── layout ───────────────────────────────────────────────────────────────

    def character_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel("Talking to")
        label.setObjectName("fieldLabel")
        self.character_box = touch.touchable_popup(QComboBox())
        self.character_box.currentIndexChanged.connect(self._character_chosen)
        edit = QPushButton("Characters…")
        edit.clicked.connect(self.open_characters)
        you = QPushButton("You…")
        you.setToolTip("Give yourself a name and a description the character can use")
        you.clicked.connect(self.open_persona)
        row.addWidget(label)
        row.addWidget(self.character_box, 1)
        row.addWidget(edit)
        row.addWidget(you)
        return row

    def chat_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel("Chat")
        label.setObjectName("fieldLabel")
        self.chat_box = touch.touchable_popup(QComboBox())
        self.chat_box.currentIndexChanged.connect(self._chat_chosen)
        new = QPushButton("New")
        new.clicked.connect(self.new_chat)
        rename = QPushButton("Rename")
        rename.clicked.connect(self.rename_chat)
        delete = QPushButton("Delete")
        delete.clicked.connect(self.delete_chat)
        self.settings_button = QPushButton("Settings")
        self.settings_button.setCheckable(True)
        self.settings_button.toggled.connect(self._toggle_settings)
        row.addWidget(label)
        row.addWidget(self.chat_box, 1)
        for button in (new, rename, delete, self.settings_button):
            row.addWidget(button)
        return row

    def middle(self) -> QSplitter:
        """The transcript, and the settings pane that folds away beside it."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.transcript_area())
        self.settings_pane = self.settings_area()
        splitter.addWidget(self.settings_pane)
        self.settings_pane.hide()          # after the splitter has taken it
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        return splitter

    def transcript_area(self) -> QScrollArea:
        self.transcript = QWidget()
        self.transcript_column = QVBoxLayout(self.transcript)
        self.transcript_column.setContentsMargins(0, 0, 0, 0)
        self.transcript_column.setSpacing(0)
        self.transcript_column.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.transcript)
        area.setFrameShape(QFrame.Shape.NoFrame)
        touch.flickable(area)
        self.transcript_scroll = area
        # Sticky bottom: the bar says where you are looking, and its range
        # changing is content arriving. Following one and reacting to the other
        # is the whole of it — scrolling away stops the transcript chasing the
        # newest message, and scrolling back to the end starts it again.
        bar = area.verticalScrollBar()
        bar.valueChanged.connect(self._note_scroll)
        bar.rangeChanged.connect(self._follow_if_pinned)
        return area

    def settings_area(self) -> QScrollArea:
        """Temperature and its neighbours, saved with the character.

        Everything in here belongs to the character rather than to the app, so
        a change to it is a change to a file on disk. That happens on its own a
        moment after the control stops moving — see ``SETTINGS_WRITE_DELAY`` —
        and Save is what makes it happen now and say so. The button is not the
        only way to save; it is the way to be told that it saved.
        """
        box = QGroupBox("How this character replies")
        column = QVBoxLayout(box)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.05, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setDecimals(2)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.05, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setDecimals(2)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(64, 4096)
        self.max_tokens.setSingleStep(64)
        self.seed = QSpinBox()
        self.seed.setRange(RANDOM_SEED, 2 ** 31 - 1)
        self.seed.setSpecialValueText("Random each reply (-1)")
        for caption, widget in (("Temperature", self.temperature), ("Top-p", self.top_p),
                                ("Reply length (tokens)", self.max_tokens), ("Seed", self.seed)):
            widget.valueChanged.connect(self._settings_touched)
            column.addWidget(touch.labelled(caption, touch.stepper(widget)))
        self.system = QPlainTextEdit()
        self.system.setPlaceholderText(
            "Leave empty to use the character's context as written. Anything here replaces the "
            "whole system message — {{char}} and {{user}} still work.")
        self.system.textChanged.connect(self._settings_touched)
        touch.flickable(self.system)
        column.addWidget(touch.labelled("Custom system message", self.system), 1)
        self.settings_note = QLabel(SETTINGS_SAVED)
        self.settings_note.setObjectName("fieldLabel")
        self.settings_note.setWordWrap(True)
        column.addWidget(self.settings_note)
        self.save_settings_button = QPushButton("Save")
        self.save_settings_button.setToolTip(
            "Write these to the character now. They apply to the next reply either way.")
        self.save_settings_button.setEnabled(False)   # a character just opened is saved
        self.save_settings_button.clicked.connect(self.save_settings)
        column.addWidget(self.save_settings_button)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(box)
        area.setFrameShape(QFrame.Shape.NoFrame)
        touch.flickable(area)
        return area

    def quick_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.regenerate_button = QPushButton("Regenerate")
        self.regenerate_button.setToolTip("Write the last reply again, keeping the one it had")
        self.regenerate_button.clicked.connect(lambda: self.regenerate(self._last(ASSISTANT)))
        self.continue_button = QPushButton("Continue")
        self.continue_button.setToolTip("Carry on from where the last reply stopped")
        self.continue_button.clicked.connect(lambda: self.continue_reply(self._last(ASSISTANT)))
        self.impersonate_button = QPushButton("Impersonate")
        self.impersonate_button.setToolTip("Have the model write your next message for you")
        self.impersonate_button.clicked.connect(self.impersonate)
        self.undo_button = QPushButton("Remove last")
        self.undo_button.setToolTip("Take back the last exchange")
        self.undo_button.clicked.connect(self.remove_last)
        for button in (self.regenerate_button, self.continue_button,
                       self.impersonate_button, self.undo_button):
            row.addWidget(button, 1)
        return row

    def input_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.attach_button = QPushButton("Attach…")
        self.attach_button.setToolTip("Send a picture with your message")
        self.attach_button.clicked.connect(self.attach_image)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Say something…   (Ctrl+Enter sends)")
        touch.flickable(self.input)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self.send)
        buttons = QVBoxLayout()
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.send_button)
        row.addWidget(self.attach_button)
        row.addWidget(self.input, 1)
        row.addLayout(buttons)
        return row

    def status_label(self) -> QLabel:
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        return self.status

    # ── characters ───────────────────────────────────────────────────────────

    def reload_characters(self, select: str | None = None) -> None:
        names = self.characters.names()
        wanted = select or (self.character.name if self.character else None) or self._remembered("character")
        self.character_box.blockSignals(True)
        self.character_box.clear()
        for name in names:
            self.character_box.addItem(name, name)
        self.character_box.blockSignals(False)
        if not names:
            self.character = None
            self.conversation = None
            self.render()
            self.set_status("No characters yet — press Characters… to make one, or to import "
                            "one you already have.")
            self._enable(False)
            return
        index = self.character_box.findData(wanted)
        # Signals stay blocked through the selection so the character is loaded
        # once, here, rather than once by the signal and once by this call.
        self.character_box.blockSignals(True)
        self.character_box.setCurrentIndex(index if index >= 0 else 0)
        self.character_box.blockSignals(False)
        self.open_character(self.character_box.currentData())

    def _character_chosen(self, _index: int) -> None:
        name = self.character_box.currentData()
        if name:
            self.open_character(name)

    def open_character(self, name: str) -> None:
        self.persist_settings()
        try:
            self.character = self.characters.load(name)
        except (OSError, ValueError, FileNotFoundError) as exc:
            self.set_status(f"Could not open {name}: {exc}")
            return
        self._avatar = None
        self.show_settings_for(self.character)
        self.remember(character=name)
        self.reload_chats()

    def show_settings_for(self, character: Character) -> None:
        for widget, value in ((self.temperature, character.temperature),
                              (self.top_p, character.top_p),
                              (self.max_tokens, character.max_reply_tokens),
                              (self.seed, character.seed)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.system.blockSignals(True)
        self.system.setPlainText(character.system)
        self.system.blockSignals(False)
        # Filling the panel in is not a change to it, and the write the timer
        # was holding was for the character that is no longer open.
        self._settings_timer.stop()
        self._settings_state(saved=True)

    def _settings_touched(self, *_args) -> None:
        """A control in the panel moved: write it once it has stopped moving.

        Restarting the timer on every tick is what turns a stepper held down
        from forty writes into one, and what makes typing a temperature save
        the number rather than each prefix of it.
        """
        if self.character is None:
            return
        self.settings_note.setText(SETTINGS_UNSAVED)
        self.save_settings_button.setEnabled(True)
        self._settings_timer.start()

    def save_settings(self) -> None:
        """The panel's Save button: write now, and say where it went.

        The values are live either way — a generation reads them off these
        controls — so this is about the file and about being told, which is the
        half that was missing when the only way to save was to close the panel.
        """
        if self.character is None:
            self.set_status("There is no character open to save these to.")
            return
        name = self.character.name
        if self.persist_settings():
            self.set_status(f"How {name} replies is saved.")
        else:
            self.set_status(f"Could not save how {name} replies — {self._settings_problem}")

    def persist_settings(self) -> bool:
        """Write the panel back to the character it belongs to.

        Called by the timer a moment after the last change, by Save, and at
        every point where the panel is about to stop being the open character's
        — sending, switching character, closing the pane, shutting down — so
        that no path out of the panel can drop what was typed into it.
        """
        self._settings_timer.stop()
        if self.character is None:
            return False
        updated = Character(name=self.character.name, context=self.character.context,
                            greeting=self.character.greeting,
                            temperature=self.temperature.value(), top_p=self.top_p.value(),
                            max_reply_tokens=self.max_tokens.value(), seed=self.seed.value(),
                            system=self.system.toPlainText().strip())
        if updated != self.character:
            try:
                self.characters.save(updated)
            except (OSError, ValueError) as exc:
                # Reported rather than swallowed: a write that silently failed
                # is what "the temperature will not stick" looks like from the
                # outside. It still must not take the chat down with it.
                self._settings_state(saved=False, problem=str(exc))
                return False
            self.character = updated
        self._settings_state(saved=True)
        return True

    def _settings_state(self, saved: bool, problem: str = "") -> None:
        """What the panel says about itself, and whether Save is worth pressing."""
        self._settings_problem = problem
        self.settings_note.setText(SETTINGS_SAVED if saved else f"Not saved — {problem}")
        self.save_settings_button.setEnabled(not saved)

    def open_characters(self) -> None:
        self.persist_settings()
        dialog = CharacterDialog(self.characters, self.metrics, self,
                                 selected=self.character.name if self.character else None)
        dialog.exec()
        self.reload_characters(select=dialog.chosen)

    def open_persona(self) -> None:
        dialog = PersonaDialog(self.persona, self)
        if dialog.exec():
            self.persona = dialog.persona()
            save_persona(self.paths, self.persona)
            self.render()               # the name over your own messages changed

    def speaker_name(self, role: str) -> str:
        if role == USER:
            return self.persona.display
        return self.character.name if self.character else "Assistant"

    def avatar_pixmap(self) -> QPixmap | None:
        """The character's picture, round, at one fingertip across.

        Round because a square photograph beside a rounded bubble is the one
        thing on the row with a corner, and drawn once per character rather
        than once per message — every reply in a chat shows the same face.
        """
        if self.character is None:
            return None
        if self._avatar is None:
            path = self.characters.avatar_for(self.character.name)
            if path is None:
                return None
            side = self.metrics["target"]
            source = QPixmap(str(path)).scaled(
                QSize(side, side), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            round_face = QPixmap(side, side)
            round_face.fill(Qt.GlobalColor.transparent)
            painter = QPainter(round_face)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            clip = QPainterPath()
            clip.addEllipse(0, 0, side, side)
            painter.setClipPath(clip)
            painter.drawPixmap((side - source.width()) // 2, (side - source.height()) // 2, source)
            painter.end()
            self._avatar = round_face
        return self._avatar

    # ── chats ────────────────────────────────────────────────────────────────

    def reload_chats(self, select: str | None = None) -> None:
        if self.character is None:
            return
        rows = self.chats.listing(self.character.name)
        self.chat_box.blockSignals(True)
        self.chat_box.clear()
        for row in rows:
            self.chat_box.addItem(row.title, row.identifier)
        self.chat_box.blockSignals(False)
        wanted = select or self._remembered("chat")
        index = self.chat_box.findData(wanted) if wanted else -1
        if index < 0 and rows:
            index = 0
        if index < 0:
            self.new_chat()
            return
        self.chat_box.blockSignals(True)
        self.chat_box.setCurrentIndex(index)
        self.chat_box.blockSignals(False)
        self.open_chat(self.chat_box.currentData())

    def _chat_chosen(self, _index: int) -> None:
        identifier = self.chat_box.currentData()
        if identifier:
            self.open_chat(identifier)

    def open_chat(self, identifier: str) -> None:
        if self.character is None:
            return
        try:
            self.conversation = self.chats.load(self.character.name, identifier)
        except (OSError, ValueError, FileNotFoundError):
            self.new_chat()
            return
        self.remember(chat=identifier)
        self.pinned = True                # a chat opens on its newest message
        self.render()
        self.set_status("")
        self._enable(True)

    def new_chat(self) -> None:
        if self.character is None:
            return
        self.conversation = self.chats.new(self.character.name)
        greeting = prompt.greeting_text(self.character, self.persona)
        if greeting:
            self.conversation.append(ASSISTANT, greeting)
        self.chats.save(self.conversation)
        self.remember(chat=self.conversation.identifier)
        self._refresh_chat_box()
        self.pinned = True
        self.render()
        self.set_status("")
        self._enable(True)

    def rename_chat(self) -> None:
        if self.conversation is None:
            return
        title, accepted = QInputDialog.getText(self, "Rename chat", "Name this chat",
                                               text=self.conversation.title)
        if not accepted or not title.strip():
            return
        self.conversation.title = title.strip()
        self.chats.save(self.conversation)
        self._refresh_chat_box()

    def delete_chat(self) -> None:
        if self.conversation is None or self.character is None:
            return
        if QMessageBox.question(self, "Delete chat",
                                f"Delete “{self.conversation.title}”? This cannot be undone.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.chats.delete(self.character.name, self.conversation.identifier)
        self.conversation = None
        self.reload_chats()

    def _refresh_chat_box(self) -> None:
        """Rebuild the past-chats list around whatever is open."""
        if self.character is None or self.conversation is None:
            return
        rows = self.chats.listing(self.character.name)
        self.chat_box.blockSignals(True)
        self.chat_box.clear()
        for row in rows:
            self.chat_box.addItem(row.title, row.identifier)
        index = self.chat_box.findData(self.conversation.identifier)
        if index >= 0:
            self.chat_box.setCurrentIndex(index)
        self.chat_box.blockSignals(False)

    # ── the transcript ───────────────────────────────────────────────────────

    def render(self) -> None:
        """Rebuild every bubble from the conversation as it stands."""
        while self.transcript_column.count():
            item = self.transcript_column.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Hidden, then unparented, and the order is the whole of it.
                #
                # Unparenting alone was the first half of a bug worth stating.
                # Taking a widget out of a layout leaves it parented and visible
                # where it was, so a rebuild that only schedules deletion paints
                # the old transcript underneath the new one until the event loop
                # gets around to it — which is why the unparenting is here.
                #
                # But a widget with no parent *is* a top-level window, and Qt
                # marks a reparented widget hidden only if it was never created.
                # These were on screen, so they are created, so they kept their
                # "not hidden" state and Qt duly put each one on the screen as a
                # window of its own on the next pass through the event loop: a
                # blank rectangle per message, flashing up and vanishing again
                # every time the transcript was rebuilt — which is every send.
                #
                # hide() first makes that state explicit, and the widget stays
                # hidden through the reparenting and on into deletion.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.bubbles = []
        self.revealed = -1          # the message under it may not exist any more
        messages = self.conversation.messages if self.conversation else []
        for index, message in enumerate(messages):
            opens_run = index == 0 or messages[index - 1].role != message.role
            bubble = MessageBubble(self, message, index, opens_run)
            self.bubbles.append(bubble)
            self.transcript_column.addWidget(bubble)
        if not messages:
            empty = QLabel("Nothing said yet." if self.character
                           else "Make a character to start talking.")
            empty.setObjectName("fieldLabel")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.transcript_column.addWidget(empty)
        self.transcript_column.addStretch(1)
        self.scroll_to_end()

    def reveal(self, index: int) -> None:
        """Show one message's menu, and no other's."""
        self.revealed = index
        for bubble in self.bubbles:
            bubble.set_revealed(bubble.index == index)

    def bubble_width(self) -> int:
        return int(self.transcript_scroll.viewport().width() * MessageBubble.WIDEST)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        width = self.bubble_width()
        for bubble in self.bubbles:
            bubble.set_max_width(width)

    # ── following the newest message ─────────────────────────────────────────

    def at_end(self) -> bool:
        bar = self.transcript_scroll.verticalScrollBar()
        return bar.value() >= bar.maximum() - STICKY_MARGIN

    def _note_scroll(self, _value: int) -> None:
        """Scrolling away stops the transcript following; scrolling back starts it."""
        self.pinned = self.at_end()

    def _follow_if_pinned(self, *_range) -> None:
        bar = self.transcript_scroll.verticalScrollBar()
        if self.pinned:
            bar.setValue(bar.maximum())

    def scroll_to_end(self, force: bool = False) -> None:
        """Go to the newest message — always when ``force``, otherwise only if
        that is where you already were."""
        if force:
            self.pinned = True
        if not self.pinned:
            return
        bar = self.transcript_scroll.verticalScrollBar()
        # After the layout has actually placed the new bubbles, not before.
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    # ── message actions ──────────────────────────────────────────────────────

    def edit_message(self, index: int, text: str) -> None:
        if self.conversation is None:
            return
        self.conversation.messages[index].text = text.strip()
        self.save()
        self.render()

    def delete_message(self, index: int) -> None:
        if self.conversation is None:
            return
        self.conversation.delete(index)
        self.save()
        self.render()

    def delete_from(self, index: int) -> None:
        if self.conversation is None:
            return
        self.conversation.delete_from(index)
        self.save()
        self.render()

    def branch(self, index: int) -> None:
        if self.conversation is None:
            return
        self.conversation = self.chats.branch(self.conversation, index)
        self.remember(chat=self.conversation.identifier)
        self._refresh_chat_box()
        self.render()
        self.set_status("Branched — this is a new chat, and the one it came from is untouched.")

    def show_version(self, index: int, version: int) -> None:
        if self.conversation is None:
            return
        self.conversation.messages[index].show(version)
        self.save()
        self.render()

    def drop_version(self, index: int) -> None:
        if self.conversation is None:
            return
        self.conversation.messages[index].drop_version()
        self.save()
        self.render()

    def copy_text(self, text: str) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(text)
        self.set_status("Copied.")

    def remove_last(self) -> None:
        """Take back the last exchange, putting your message back in the box."""
        if self.conversation is None or not self.conversation.messages:
            return
        messages = self.conversation.messages
        if messages[-1].role == ASSISTANT:
            messages.pop()
        if messages and messages[-1].role == USER:
            taken = messages.pop()
            self.input.setPlainText(taken.text)
        self.save()
        self.render()

    # ── sending ──────────────────────────────────────────────────────────────

    def attach_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Attach an image", "",
                                                  "Images (*.png *.jpg *.jpeg *.webp)")
        if not filename:
            return
        self.attachment = Path(filename)
        self.attach_button.setText(f"Attached: {self.attachment.name}")
        # Said now rather than when the reply fails: whether a picture can be
        # sent depends on the model running, and a model chosen by hand may have
        # no vision projector at all.
        if not self._vision_ready():
            self.set_status(f"{self.attachment.name} is attached, but the model running has no "
                            "vision projector and cannot be shown it. Choose one under "
                            "Settings → Which model runs.")
            return
        self.set_status(f"{self.attachment.name} goes with your next message. "
                        "Press Attach again to change it.")

    def _vision_ready(self) -> bool:
        """Whether a picture can reach the model — false only when it is known to be.

        A warning about the projector has to be confident: an install that is
        not set up yet, or a state file that cannot be read, is setup's problem
        to describe and not a reason to tell somebody their model cannot see.
        """
        if not self.paths.configured:
            return True
        try:
            return bool(self.service_provider().vision_ready())
        except Exception:
            return True

    def clear_attachment(self) -> None:
        self.attachment = None
        self.attach_button.setText("Attach…")

    def send(self) -> None:
        if self.busy() or self.conversation is None or self.character is None:
            return
        text = self.input.toPlainText().strip()
        if not text and self.attachment is None:
            return
        image, name = "", ""
        if self.attachment is not None:
            try:
                image, name = image_data_url(self.attachment), self.attachment.name
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, "Image error", str(exc))
                return
        self.conversation.append(USER, text, image=image, image_name=name)
        self.pinned = True                # you just wrote it; go and look at it
        self.input.clear()
        self.clear_attachment()
        self.save()
        self._refresh_chat_box()          # an untitled chat has just been named
        self.render()
        self.reply()

    def reply(self) -> None:
        """A fresh assistant message, streamed into."""
        if self.conversation is None:
            return
        message = self.conversation.append(ASSISTANT, "")
        self.render()
        self.stream(len(self.conversation.messages) - 1, message.text)

    def regenerate(self, index: int) -> None:
        """Write this reply again, keeping the one it had as a version."""
        if self.conversation is None or index < 0 or self.busy():
            return
        message = self.conversation.messages[index]
        if message.role != ASSISTANT:
            return
        self.conversation.truncate_after(index)
        message.add_version("")
        self.render()
        self.stream(index, "")

    def resend(self, index: int) -> None:
        """Answer this message of yours again, dropping everything after it."""
        if self.conversation is None or self.busy():
            return
        self.conversation.truncate_after(index)
        self.save()
        self.render()
        self.reply()

    def continue_reply(self, index: int) -> None:
        """Carry the reply on from where it stopped."""
        if self.conversation is None or index < 0 or self.busy() or self.character is None:
            return
        message = self.conversation.messages[index]
        if message.role != ASSISTANT or not message.text.strip():
            return
        # ``upto`` includes the reply itself: the model cannot carry on from
        # text it was not shown.
        self.stream(index, message.text, instruction=prompt.continue_instruction(self.character),
                    upto=index + 1)

    def impersonate(self) -> None:
        """Have the model write your next message into the input box."""
        if self.conversation is None or self.busy() or self.character is None:
            return
        self.stream(-1, "", instruction=prompt.impersonate_instruction(self.persona),
                    into_input=True)

    def stream(self, index: int, prefix: str, instruction: str | None = None,
               upto: int | None = None, into_input: bool = False) -> None:
        """Run one generation, writing into message ``index`` or the input box.

        ``prefix`` is what is already there — the text a continuation extends —
        and ``upto`` bounds the history sent, so continuing a reply does not
        include the empty message being written.
        """
        if self.character is None or self.conversation is None or self.busy():
            return
        self.persist_settings()
        history = self.conversation.messages[:upto if upto is not None else index]
        if into_input:
            history = list(self.conversation.messages)
        reply_tokens = self.max_tokens.value()
        messages = prompt.build(self.character, self.persona, history,
                                context_size=self.context_size(),
                                reply_tokens=reply_tokens, instruction=instruction)
        seed = self.seed.value()
        worker = ChatWorker(self.service_provider(), messages, prompt.has_image(history),
                            self.temperature.value(), self.top_p.value(), reply_tokens,
                            draw_seed() if seed == RANDOM_SEED else seed)
        self._streaming_index = -1 if into_input else index
        self._streaming_prefix = prefix
        self._into_input = into_input
        # A continuation is joined to what is already there, and the model is
        # not reliable about starting with the space that needs.
        self._join_space = bool(prefix) and not prefix[-1].isspace()
        self.thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.chunk.connect(self._chunk)
        worker.completed.connect(self._completed)
        worker.failed.connect(self._failed)
        worker.finished.connect(self.thread.quit)
        worker.finished.connect(worker.deleteLater)
        self.thread.finished.connect(self._generation_done)
        self._enable(False)
        self.stop_button.setEnabled(True)
        self.set_status("Thinking…" if not into_input else "Writing as you…")
        self.thread.start()

    def _chunk(self, text: str) -> None:
        if self._join_space and text and not text[0].isspace():
            self._streaming_prefix += " "
        self._join_space = False
        self._streaming_prefix += text
        if self._into_input:
            self.input.setPlainText(self._streaming_prefix)
            return
        if 0 <= self._streaming_index < len(self.bubbles):
            self.bubbles[self._streaming_index].set_text(self._streaming_prefix)
            self.scroll_to_end()

    def _completed(self, _raw: str) -> None:
        text = self._streaming_prefix
        if self._into_input:
            self.input.setPlainText(prompt.clean_reply(text, self.character, self.persona)
                                    if self.character else text)
            self.set_status("Written as you — edit it, then send.")
            return
        if self.conversation is None or not (0 <= self._streaming_index < len(self.conversation.messages)):
            return
        message = self.conversation.messages[self._streaming_index]
        cleaned = prompt.clean_reply(text, self.character, self.persona) if self.character else text
        message.text = cleaned or "…"
        self.save()
        self.render()
        self.set_status("")

    def _failed(self, message: str) -> None:
        self.set_status(f"Generation failed — {message}")
        # The empty shell of a reply that never arrived is not left behind.
        if (self.conversation is not None and not self._into_input
                and 0 <= self._streaming_index < len(self.conversation.messages)):
            failed = self.conversation.messages[self._streaming_index]
            if not failed.text.strip():
                if len(failed.versions) > 1:
                    failed.drop_version()
                elif self._streaming_index == len(self.conversation.messages) - 1:
                    self.conversation.delete(self._streaming_index)
            self.save()
            self.render()

    def _generation_done(self) -> None:
        self._enable(True)
        self.stop_button.setEnabled(False)
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread, self._worker = None, None

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.set_status("Stopping…")

    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    # ── plumbing ─────────────────────────────────────────────────────────────

    def context_size(self) -> int:
        """The window llama-server was started with, which the history has to fit.

        Read per generation rather than cached: re-running setup can change it,
        and a state file that cannot be read is a reason to use the default
        rather than to refuse to answer.
        """
        try:
            recorded = read_json(self.paths.state_file).get("context_size")
            return int(recorded) if recorded else DEFAULT_CONTEXT
        except (OSError, ValueError, TypeError):
            return DEFAULT_CONTEXT

    def save(self) -> None:
        if self.conversation is not None:
            try:
                self.chats.save(self.conversation)
            except OSError as exc:
                self.set_status(f"Could not save this chat — {exc}")

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def _enable(self, enabled: bool) -> None:
        ready = enabled and self.character is not None and self.conversation is not None
        for button in (self.send_button, self.regenerate_button, self.continue_button,
                       self.impersonate_button, self.undo_button, self.attach_button):
            button.setEnabled(ready)

    def _last(self, role: str) -> int:
        return self.conversation.last_index(role) if self.conversation else -1

    def _toggle_settings(self, shown: bool) -> None:
        self.settings_pane.setVisible(shown)
        if not shown:
            self.persist_settings()

    def apply_metrics(self, metrics: dict) -> None:
        """Follow the window's display size."""
        self.metrics = metrics
        # Bubbles are held off the edges of the transcript rather than against
        # them, so the scroll bar has somewhere to be that is not on top of a
        # message.
        self.transcript_column.setContentsMargins(metrics["pad"], 0,
                                                  metrics["pad"], metrics["gap"])
        self.input.setMinimumHeight(metrics["target"] * 2)
        self.input.setMaximumHeight(metrics["target"] * 4)
        self.system.setMinimumHeight(metrics["target"] * 3)
        self.send_button.setMinimumWidth(metrics["target"] * 3)
        self._avatar = None
        self.render()

    def rebind(self, paths) -> None:
        """Follow a setup run that moved the installation root."""
        self.persist_settings()
        self.paths = paths
        self.characters = CharacterStore.from_paths(paths)
        self.chats = ChatStore.from_paths(paths)
        self.persona = load_persona(paths)
        self.character, self.conversation = None, None
        self.reload_characters()

    def remember(self, character: str | None = None, chat: str | None = None) -> None:
        state = self._state()
        if character is not None:
            state["character"] = character
            state.pop("chat", None)          # the chat belonged to the old one
        if chat is not None:
            state["chat"] = chat
        try:
            atomic_write_json(self.paths.data / STATE_FILE, state)
        except OSError:
            pass                              # remembering is a convenience

    def bar_visible(self, key: str) -> bool:
        """Whether a hideable row is showing. Everything starts out shown."""
        bars = self._state().get("bars")
        if isinstance(bars, dict) and isinstance(bars.get(key), bool):
            return bars[key]
        return True

    def show_bar(self, key: str, shown: bool) -> None:
        bar = self.bars.get(key)
        if bar is None:
            return
        bar.setVisible(shown)
        state = self._state()
        bars = dict(state.get("bars") or {}) if isinstance(state.get("bars"), dict) else {}
        bars[key] = bool(shown)
        state["bars"] = bars
        try:
            atomic_write_json(self.paths.data / STATE_FILE, state)
        except OSError:
            pass                              # remembering is a convenience

    def _remembered(self, key: str) -> str | None:
        value = self._state().get(key)
        return value if isinstance(value, str) else None

    def _state(self) -> dict:
        try:
            return read_json(self.paths.data / STATE_FILE)
        except (OSError, ValueError):
            return {}

    def shutdown(self) -> None:
        self.persist_settings()
        if self._worker is not None:
            self._worker.cancel()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)


def _bar(row: QHBoxLayout) -> QWidget:
    """A row of controls as one widget, so the View menu can hide all of it."""
    holder = QWidget()
    row.setContentsMargins(0, 0, 0, 0)
    holder.setLayout(row)
    return holder


def _pixmap_from_data_url(data_url: str) -> QPixmap | None:
    _, _, encoded = data_url.partition(",")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    pixmap = QPixmap()
    return pixmap if pixmap.loadFromData(raw) else None

