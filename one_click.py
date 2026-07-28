#!/usr/bin/env python
"""One-click installer for Prompt Master Standalone.

Run it through ``start_windows.bat`` (which finds an interpreter for it first)
or directly with any Python 3.8+:

    python one_click.py

It does four things, each skippable once already done, so re-running it is
always safe:

1. finds a Python 3.12 or 3.13 interpreter, downloading a hash-pinned Miniconda
   only if the machine has none;
2. builds an isolated environment under ``installer_files/env`` and installs the
   five runtime dependencies into it;
3. runs the setup questions — install directory, GPU, model quantization — and
   downloads the pinned llama.cpp runtime and Gemma weights;
4. launches the application.

Everything it creates lives under ``installer_files/`` and the install directory
chosen in step 3. Nothing is written to the system Python, PATH or registry.

This file is deliberately written against Python 3.8 syntax: it is the one file
that may be executed by whatever interpreter happens to be on the machine, and
it has to be able to report a too-old Python rather than fail to parse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTALLER_FILES = HERE / "installer_files"
ENV_DIR = INSTALLER_FILES / "env"
CONDA_DIR = INSTALLER_FILES / "conda"
DOWNLOADS = INSTALLER_FILES / "downloads"
REQUIREMENTS_MARKER = INSTALLER_FILES / "requirements.installed"

# The application targets 3.12; the vendored prompt engine and the app both use
# syntax and typing behaviour from it. The upper bound comes from the pinned
# dependencies rather than from the code: PySide6 6.8.1 declares
# Requires-Python <3.14, and neither Pillow 11.0.0 nor numpy 2.2.1 publishes a
# 3.14 wheel, so an environment built on a newer interpreter cannot install
# requirements.txt at all. A newer Python is therefore a reason to skip an
# interpreter, not to prefer it; move this bound only together with the pins.
MIN_PYTHON = (3, 12)
MAX_PYTHON = (3, 14)          # exclusive

# Bootstrap interpreter, used only when the machine has no supported Python. It is
# pinned by version and verified by SHA-256 before it is executed, on the same
# principle as the model manifest: nothing unverified is ever run. Only its
# bundled python.exe is used — no conda command is invoked, and conda is not put
# on PATH or into the registry.
MINICONDA = {
    "url": "https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-0-Windows-x86_64.exe",
    "filename": "Miniconda3-py312_25.5.1-0-Windows-x86_64.exe",
    "size": 91218584,
    "sha256": "82cf1382eaa2c92d4f0cbd826a8888fac25080fe8ff3370bae8d0732b80006fb",
}

# Characters that cmd.exe, pip or NSIS mangle in a path. Spaces are merely
# warned about; these break the install outright.
FORBIDDEN_PATH_CHARS = '!%^&="\''


def banner(text):
    line = "*" * min(shutil.get_terminal_size((80, 24)).columns, 78)
    print("\n{0}\n* {1}\n{0}\n".format(line, text))


def run(command, **kwargs):
    """Run a command, echoing it, and raise on failure."""
    printable = " ".join(str(part) for part in command)
    print("> {0}".format(printable))
    result = subprocess.run([str(part) for part in command], **kwargs)
    if result.returncode != 0:
        raise SystemExit("\nCommand failed ({0}): {1}".format(result.returncode, printable))
    return result


def supported(version):
    """True for a (major, minor) the pinned dependencies publish wheels for."""
    return version is not None and MIN_PYTHON <= tuple(version) < MAX_PYTHON


def supported_versions():
    """The supported range as prose, e.g. "3.12 or 3.13"."""
    names = ["{0}.{1}".format(MIN_PYTHON[0], minor)
             for minor in range(MIN_PYTHON[1], MAX_PYTHON[1])]
    if len(names) == 1:
        return names[0]
    return "{0} or {1}".format(", ".join(names[:-1]), names[-1])


def entry_point():
    return "start_windows.bat" if sys.platform == "win32" else "python one_click.py"


def running_inside(directory):
    try:
        return directory.resolve() in Path(sys.executable).resolve().parents
    except OSError:
        return False


def remove_tree(directory, purpose):
    """Delete a directory the installer owns, refusing to delete the interpreter
    it is itself running under: Windows cannot unlink a running executable, so
    that rebuild has to be started from somewhere else."""
    if running_inside(directory):
        raise SystemExit(
            "This installer is running from the {0} it has to rebuild:\n"
            "  {1}\n"
            "Windows cannot delete an interpreter while it is executing. Run {2} "
            "instead — it starts from a different Python and rebuilds "
            "automatically.".format(purpose, sys.executable, entry_point()))
    shutil.rmtree(str(directory))


# ── step 0: sanity ───────────────────────────────────────────────────────────

def check_path():
    text = str(HERE)
    bad = [character for character in FORBIDDEN_PATH_CHARS if character in text]
    if bad:
        raise SystemExit(
            "This directory cannot be used because its path contains {0}\n"
            "  {1}\n"
            "Move the folder somewhere without those characters and run the installer again."
            .format(" ".join(repr(character) for character in bad), text)
        )
    if not text.isascii():
        raise SystemExit(
            "This directory cannot be used because its path contains non-ASCII characters:\n"
            "  {0}\n"
            "Move the folder to an ASCII-only path and run the installer again.".format(text)
        )
    if " " in text:
        print("Note: this path contains spaces. That normally works, but if pip or\n"
              "llama.cpp misbehaves later, a path without spaces is the first thing to try.\n"
              "  {0}\n".format(text))


# ── step 1: a supported Python interpreter ───────────────────────────────────

def interpreter_version(executable):
    """(major, minor) for an interpreter, or None if it will not report one."""
    try:
        result = subprocess.run(
            [str(executable), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.match(r"\s*(\d+)\.(\d+)", result.stdout)
    return (int(match.group(1)), int(match.group(2))) if match else None


def find_system_python():
    """The newest suitable interpreter already on this machine, if any.

    "Suitable" is a range, not a floor: an interpreter newer than the pinned
    dependencies support is skipped exactly like one that is too old, because
    the environment it builds cannot install requirements.txt.
    """
    candidates = []
    if supported(sys.version_info[:2]):
        candidates.append(Path(sys.executable))
    if sys.platform == "win32":
        # The py launcher knows about installs that are not on PATH.
        launcher = shutil.which("py")
        if launcher:
            for minor in range(MAX_PYTHON[1] - 1, MIN_PYTHON[1] - 1, -1):
                candidates.append("{0} -3.{1}".format(launcher, minor))
    names = ["python{0}.{1}".format(MIN_PYTHON[0], minor)
             for minor in range(MAX_PYTHON[1] - 1, MIN_PYTHON[1] - 1, -1)]
    for name in names + ["python3", "python"]:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    for candidate in candidates:
        if isinstance(candidate, str) and " -3." in candidate:
            launcher, flag = candidate.split(" ", 1)
            probe = [launcher, flag, "-c", "import sys; print(sys.executable)"]
            try:
                result = subprocess.run(probe, capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode != 0 or not result.stdout.strip():
                continue
            candidate = Path(result.stdout.strip())
        if not Path(candidate).is_file():
            continue
        if supported(interpreter_version(candidate)):
            return Path(candidate)
    return None


def verified_download(url, destination, size, sha256):
    """Download to ``destination``, verifying size and SHA-256 before use."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _digest_matches(destination, size, sha256):
        print("Already downloaded and verified: {0}".format(destination.name))
        return destination

    partial = destination.with_name(destination.name + ".part")
    print("Downloading {0} ({1:.1f} MiB)…".format(destination.name, size / 2 ** 20))
    with urllib.request.urlopen(url) as response, partial.open("wb") as stream:
        done = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            stream.write(chunk)
            done += len(chunk)
            sys.stdout.write("\r  {0:5.1f}%".format(100.0 * done / size))
            sys.stdout.flush()
    sys.stdout.write("\n")

    if not _digest_matches(partial, size, sha256):
        partial.unlink()
        raise SystemExit(
            "Verification failed for {0}.\n"
            "The download did not match its pinned SHA-256 and has been deleted. "
            "Check your network and run the installer again.".format(destination.name)
        )
    os.replace(str(partial), str(destination))
    return destination


def _digest_matches(path, size, sha256):
    if path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower() == sha256.lower()


def bootstrap_miniconda():
    """Install the pinned Miniconda privately and return its python.exe."""
    existing = CONDA_DIR / "python.exe"
    if existing.is_file() and supported(interpreter_version(existing)):
        return existing
    if sys.platform != "win32":
        raise SystemExit(
            "No Python {0} was found, and the bundled bootstrap is Windows-only.\n"
            "Install Python {0} and run the installer again.\n"
            "A newer Python will not do: the pinned dependencies publish no wheels for it."
            .format(supported_versions())
        )
    if existing.is_file():
        # A private Python from an earlier run that the pins no longer cover, or
        # one replaced by hand. The pinned build is what goes back in its place.
        print("The private Python in {0} is not one the pinned dependencies support; "
              "replacing it.".format(CONDA_DIR))
        remove_tree(CONDA_DIR, "private Python")

    banner("No Python {0} found — installing a private one".format(supported_versions()))
    print("Nothing is added to PATH or the registry, and your system Python is untouched.")
    print("Everything goes into {0}\n".format(CONDA_DIR))
    installer = verified_download(MINICONDA["url"], DOWNLOADS / MINICONDA["filename"],
                                  MINICONDA["size"], MINICONDA["sha256"])
    CONDA_DIR.parent.mkdir(parents=True, exist_ok=True)
    # NSIS requires /D last and unquoted, and returns before finishing unless it
    # is waited on, which is what `start /wait` is for.
    command = 'start /wait "" "{0}" /InstallationType=JustMe /NoRegistry=1 /NoShortcuts=1 /S /D={1}'.format(
        installer, CONDA_DIR)
    print("> {0}".format(command))
    if subprocess.call(command, shell=True) != 0 or not existing.is_file():
        raise SystemExit("The bootstrap Python installer did not complete. Try running the installer again.")
    return existing


def bootstrap_python():
    found = find_system_python()
    if found is not None:
        version = interpreter_version(found)
        print("Using Python {0}.{1} at {2}".format(version[0], version[1], found))
        return found
    return bootstrap_miniconda()


# ── step 2: the environment ──────────────────────────────────────────────────

def env_python():
    for relative in (Path("Scripts") / "python.exe", Path("bin") / "python"):
        candidate = ENV_DIR / relative
        if candidate.is_file():
            return candidate
    return None


def env_problem(existing):
    """Why the environment on disk cannot be used, in one clause."""
    if existing is None:
        return "it has no interpreter"
    version = interpreter_version(existing)
    if version is None:
        return "its interpreter does not run"
    return ("it was built with Python {0}.{1}, which the pinned dependencies "
            "publish no wheels for".format(version[0], version[1]))


def remove_env():
    """Delete the environment and the marker that says what is installed in it.

    The marker lives outside ENV_DIR, so leaving it behind would make the next
    dependency install think a freshly created, empty environment was already
    populated.
    """
    remove_tree(ENV_DIR, "environment")
    if REQUIREMENTS_MARKER.is_file():
        REQUIREMENTS_MARKER.unlink()


def create_env(python, reinstall=False):
    if reinstall and ENV_DIR.exists():
        print("Removing {0}".format(ENV_DIR))
        remove_env()
    existing = env_python()
    if existing is not None and supported(interpreter_version(existing)):
        print("Environment already present: {0}".format(ENV_DIR))
        return existing
    if ENV_DIR.exists():
        # Present but not usable — a half-created venv, one whose base
        # interpreter was uninstalled, or one built on a Python the pins do not
        # cover. Rebuilding is the only repair.
        print("Environment at {0} is unusable: {1}. Rebuilding it."
              .format(ENV_DIR, env_problem(existing)))
        remove_env()
    banner("Creating the environment")
    ENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    run([python, "-m", "venv", ENV_DIR])
    created = env_python()
    if created is None:
        raise SystemExit("venv reported success but produced no interpreter at {0}".format(ENV_DIR))
    return created


def install_requirements(python, update=False):
    version = interpreter_version(python)
    if not supported(version):
        # pip's own diagnosis of this is forty lines of ignored versions ending
        # in "No matching distribution found", so say it plainly first.
        raise SystemExit(
            "The environment at {0} runs Python {1}, and requirements.txt pins versions "
            "that only publish wheels for Python {2}.\n"
            "Delete that directory and run {3} again to rebuild it on a supported "
            "interpreter.".format(
                ENV_DIR,
                "{0}.{1}".format(*version) if version else "an interpreter that does not report a version",
                supported_versions(),
                entry_point()))
    requirements = HERE / "requirements.txt"
    current = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if (not update and REQUIREMENTS_MARKER.is_file()
            and REQUIREMENTS_MARKER.read_text(encoding="utf-8").strip() == current):
        print("Dependencies already installed.")
        return
    banner("Installing dependencies")
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", requirements])
    REQUIREMENTS_MARKER.write_text(current, encoding="utf-8")


# ── step 3 and 4: setup and launch ───────────────────────────────────────────

def already_configured():
    """True when a previous run left a validated model install behind."""
    marker = HERE / "install.json"
    if not marker.is_file():
        return False
    try:
        root = Path(json.loads(marker.read_text(encoding="utf-8"))["install_root"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return (root / "data" / "setup-state.json").is_file()


def parse_args():
    parser = argparse.ArgumentParser(description="Install and launch Prompt Master Standalone.")
    parser.add_argument("--update", action="store_true", help="Reinstall dependencies from requirements.txt")
    parser.add_argument("--reinstall", action="store_true", help="Delete and rebuild the environment")
    parser.add_argument("--setup", action="store_true", help="Re-run the model and hardware questions")
    parser.add_argument("--no-launch", action="store_true", help="Install without starting the application")
    return parser.parse_known_args()


def main():
    options, passthrough = parse_args()
    banner("Prompt Master Standalone — installer")
    print("Directory : {0}".format(HERE))
    print("Platform  : {0} {1}".format(platform.system(), platform.machine()))

    check_path()
    INSTALLER_FILES.mkdir(parents=True, exist_ok=True)

    python = create_env(bootstrap_python(), reinstall=options.reinstall)
    install_requirements(python, update=options.update or options.reinstall)

    launcher = HERE / "app.py"
    if options.setup or not already_configured():
        banner("Model and hardware setup")
        result = subprocess.call([str(python), str(launcher), "--setup"] + passthrough)
        if result != 0:
            print("\nSetup did not complete. Re-run this installer, or run:\n"
                  "  python app.py --setup")
            return result
    else:
        print("\nA configured model install was found; skipping setup.")
        print("Re-run the questions any time with:  python app.py --setup")

    if options.no_launch:
        banner("Install complete")
        print("Launch it with:  python app.py")
        return 0

    banner("Starting Prompt Master")
    print("From now on you can start it directly with:\n  python app.py\n")
    return subprocess.call([str(python), str(launcher)])


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        sys.exit("This installer needs Python 3.8 or newer to run (it will install "
                 "Python {0} for the application itself).".format(supported_versions()))
    sys.exit(main())
