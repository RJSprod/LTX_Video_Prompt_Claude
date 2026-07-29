"""The models on this machine, and which one each mode talks to.

Setup installs one model, pinned and hash-verified, and everything downstream of
it can assume that file. This is the other case: a `.gguf` of your own, put in
the models folder and used as it is. Three things follow from that, and they are
the whole of this module's design.

*A supplied model keeps its own name.* The pinned install names files from the
manifest; a model you chose is recognised by what it is called, so it is
installed under exactly that name and shown by it. Two different files that want
the same name are the only case where one is numbered.

*Nothing here is verified, and it does not pretend to be.* There is no hash to
check a model of your own against — it is not a file this build pins. The
guarantee that survives is narrower and still worth having: the pinned install
is untouched, still recorded in ``setup-state.json``, and is what every mode
falls back to.

*A mode with no model of its own uses the installed one.* Assignments live in
their own small file, and hold only what has been chosen deliberately, so an
install that never opens this menu behaves exactly as it did before the menu
existed — and deleting the file is a complete reset.

A projector is told from a model by its name. That is a convention rather than a
format detail, but it is the convention every publisher of these files follows,
and it decides only what each drop-down offers first: any `.gguf` can still be
chosen as either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import atomic_write_json, read_json

# The two things this application does, as the assignment file names them.
PROMPT = "prompt"
CONVERSATION = "conversation"
MODES = (PROMPT, CONVERSATION)

# Beside the display size and the chat's own state: a preference, not an
# install. Deleting it returns every mode to the model setup installed.
ASSIGNMENTS_FILE = "models.json"

MODEL_SUFFIX = ".gguf"

# What a vision projector is called, by convention, wherever these files are
# published. Used to sort one drop-down; never to refuse a file.
PROJECTOR_MARKS = ("mmproj", "projector")


@dataclass(frozen=True)
class ModelEntry:
    """One model, and the projector it sees with — paths under the install root."""

    model: str
    mmproj: str = ""

    @property
    def name(self) -> str:
        return PurePosixPath(self.model).name

    @property
    def projector_name(self) -> str:
        return PurePosixPath(self.mmproj).name if self.mmproj else ""

    @property
    def has_vision(self) -> bool:
        """Whether images can be sent to this model at all."""
        return bool(self.mmproj)

    def to_dict(self) -> dict:
        return {"model": self.model, "mmproj": self.mmproj}

    @classmethod
    def from_dict(cls, data: object) -> "ModelEntry | None":
        if not isinstance(data, dict) or not data.get("model"):
            return None
        return cls(str(data["model"]), str(data.get("mmproj") or ""))


def is_projector(relative: str) -> bool:
    name = PurePosixPath(relative).name.casefold()
    return any(mark in name for mark in PROJECTOR_MARKS)


class ModelLibrary:
    """Every ``.gguf`` in the models folder, and the per-mode choice of one."""

    def __init__(self, paths):
        self.paths = paths

    # ── what is on disk ──────────────────────────────────────────────────────

    def files(self) -> list[str]:
        """Every model file, as a path relative to the install root."""
        directory = self.paths.root / "models"
        found = [path.relative_to(self.paths.root).as_posix()
                 for path in sorted(directory.glob(f"*{MODEL_SUFFIX}"))]
        return found

    def models(self) -> list[str]:
        """The files offered as models — everything that is not a projector."""
        return [relative for relative in self.files() if not is_projector(relative)]

    def projectors(self) -> list[str]:
        """The files offered as projectors first; the rest are still choosable."""
        return [relative for relative in self.files() if is_projector(relative)]

    def exists(self, relative: str) -> bool:
        if not relative:
            return False
        try:
            return self.paths.contained(relative).is_file()
        except ValueError:
            return False

    # ── what each mode uses ──────────────────────────────────────────────────

    def installed_model(self) -> str:
        """The model setup installed — every mode's fallback."""
        return str(self._read(self.paths.state_file).get("model") or "")

    def projector_for(self, model: str) -> str:
        """The projector this model sees with, if it has one.

        Kept per model rather than per mode, so a model carries its projector
        wherever it is used and switching a mode to it in one tap does not have
        to ask again. The installed model's own projector is its default, which
        is what makes an install that never opens this menu unchanged.
        """
        if not model:
            return ""
        remembered = self._section("projectors").get(model)
        if isinstance(remembered, str) and self.exists(remembered):
            return remembered
        if remembered is not None:
            return ""                       # deliberately none, or since deleted
        state = self._read(self.paths.state_file)
        if model == state.get("model") and self.exists(str(state.get("mmproj") or "")):
            return str(state["mmproj"])
        return ""

    def entry_for(self, mode: str) -> ModelEntry | None:
        """The model this mode will talk to, chosen or inherited.

        An assignment whose file has since been deleted is ignored rather than
        obeyed: the installed model is a better answer than a missing one, and
        the menu shows which is actually in use.
        """
        chosen = self._section("assignments").get(mode)
        model = chosen if isinstance(chosen, str) and self.exists(chosen) else self.installed_model()
        if not model:
            return None
        return ModelEntry(model, self.projector_for(model))

    def assigned(self, mode: str) -> str:
        """What this mode was told to use, whether or not the file is still there."""
        chosen = self._section("assignments").get(mode)
        return chosen if isinstance(chosen, str) else ""

    def assign(self, mode: str, model: str | None) -> None:
        """Point ``mode`` at ``model``, or back at the installed one."""
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}")
        assignments = dict(self._section("assignments"))
        if model:
            assignments[mode] = model
        else:
            assignments.pop(mode, None)
        self._write(assignments=assignments)

    def remember_projector(self, model: str, mmproj: str) -> None:
        """Record what this model sees with — including nothing, deliberately."""
        projectors = dict(self._section("projectors"))
        projectors[model] = mmproj or ""
        self._write(projectors=projectors)

    def _section(self, name: str) -> dict:
        stored = self._read(self.paths.data / ASSIGNMENTS_FILE).get(name)
        return stored if isinstance(stored, dict) else {}

    def _write(self, **sections) -> None:
        stored = self._read(self.paths.data / ASSIGNMENTS_FILE)
        current = {name: self._section(name) for name in ("assignments", "projectors")}
        current.update(sections)
        atomic_write_json(self.paths.data / ASSIGNMENTS_FILE, {**stored, **current})

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            return read_json(path)
        except (OSError, ValueError):
            return {}
