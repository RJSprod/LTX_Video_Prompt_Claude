#!/usr/bin/env python
"""Prompt Master Standalone — launcher.

This is the file to run after installing:

    python app.py

It is deliberately dependency-free at the top level. When it is started with an
interpreter that is not the one the installer built — the usual case, since
``python`` on PATH is whatever Windows resolves it to — it re-executes itself
under ``installer_files\\env`` instead of failing on a missing PySide6. That is
what lets a plain ``python app.py`` work from any command prompt without the
user having to activate anything first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE / "installer_files" / "env"

# Set across the re-exec so a broken environment fails with its own import
# error instead of bouncing between interpreters forever.
REEXEC_GUARD = "PROMPT_MASTER_REEXEC"


def env_python() -> Path | None:
    """The interpreter the one-click installer built, if it exists."""
    for relative in (Path("Scripts") / "python.exe", Path("bin") / "python"):
        candidate = ENV_DIR / relative
        if candidate.is_file():
            return candidate
    return None


def running_in_env(interpreter: Path) -> bool:
    try:
        return Path(sys.executable).resolve() == interpreter.resolve()
    except OSError:
        return False


def reexec_if_needed() -> None:
    interpreter = env_python()
    if interpreter is None or running_in_env(interpreter) or os.environ.get(REEXEC_GUARD):
        return
    environment = dict(os.environ, **{REEXEC_GUARD: "1"})
    arguments = [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]]
    if sys.platform == "win32":
        # os.execv on Windows detaches the child from the console's process
        # list, so an interactive setup run would lose its stdin. Waiting on a
        # subprocess keeps the console attached and the exit code intact.
        import subprocess

        raise SystemExit(subprocess.call(arguments, env=environment))
    os.execve(str(interpreter), arguments, environment)


def main() -> int:
    reexec_if_needed()
    sys.path.insert(0, str(HERE / "src"))
    try:
        from prompt_master.app import main as run
    except ImportError as exc:
        installer = "start_windows.bat" if sys.platform == "win32" else "python one_click.py"
        print(f"Prompt Master is not installed yet: {exc}", file=sys.stderr)
        print(f"Run {installer} first — it builds the environment and downloads the model.", file=sys.stderr)
        return 3
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
