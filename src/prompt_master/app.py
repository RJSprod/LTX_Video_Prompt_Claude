"""Application entry point.

``--setup`` runs the console wizard rather than the Qt one. That is deliberate:
setup is also what the one-click installer runs, before there is any window to
put a wizard in, and having one setup path means a reinstall from the command
line and a reinstall from the installer ask the same questions in the same
order. The Qt wizard is still reachable from Settings → Models and Hardware,
for reconfiguring an app that is already open.
"""

from __future__ import annotations

import sys

from prompt_master.core.paths import AppPaths

VERSION = "1.0.0"

USAGE = """Prompt Master Standalone — LTX-Video 2.3 prompt generator

  python app.py                 Open the application
  python app.py --setup         Choose GPU, model and install directory
  python app.py --setup --help  Setup options for unattended reinstall
  python app.py --version       Print the version
"""


def main() -> int:
    argv = sys.argv[1:]

    if "--version" in argv:
        print(f"Prompt Master Standalone {VERSION}")
        return 0
    if argv and argv[0] in ("-h", "--help", "help") and "--setup" not in argv:
        print(USAGE)
        return 0

    if "--setup" in argv:
        from prompt_master import setup_cli

        rest = [argument for argument in argv if argument != "--setup"]
        code = setup_cli.run(rest)
        if code != 0:
            return code
        # Setup finished. Opening the window is the point of having run it, and
        # the fresh AppPaths below picks up the directory setup just recorded.

    paths = AppPaths.discover()
    if not paths.configured:
        print("No model is configured yet. Run `python app.py --setup` first.", file=sys.stderr)
        print(f"(Looked for {paths.state_file})", file=sys.stderr)
        return 2
    paths.create_managed_dirs()

    from PySide6.QtWidgets import QApplication

    from prompt_master.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Prompt Master Standalone")
    window = MainWindow(paths)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
