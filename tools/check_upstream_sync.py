"""File-integrity check for the vendored Prompt-Master-LD engine.

Compares each module under ``src/prompt_master/prompt_engine/upstream`` against
an untouched Prompt-Master-LD checkout and fails when anything other than an
approved packaging change has been made.

The porting contract allows only these edits to a vendored module:

* rewriting imports to package-relative form,
* deleting ComfyUI / Torch imports,
* adding ``from __future__`` or typing-only imports.

Anything else -- a changed prompt string, a shortened dictionary, a dropped
conditional branch, a reordered rule -- must be declared as an explicit
exception in ``UPSTREAM_DIFF_NOTES.md`` or this check fails.

Usage::

    python tools/check_upstream_sync.py --upstream ../Prompt-Master-LD
    python tools/check_upstream_sync.py --upstream ../Prompt-Master-LD --json
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Engine modules that are ported and must stay behaviorally identical.
ENGINE_MODULES = (
    "brain.py",
    "accents.py",
    "music.py",
    "styles.py",
    "cinematics.py",
    "hands.py",
    "identity.py",
    "wardrobe.py",
    "negative.py",
    "shotscript.py",
    "imaging.py",
)

# Retained purely as implementation references. They are still hashed so that
# drift is visible, but they are not imported by the standalone application.
REFERENCE_MODULES = (
    "node.py",
    "routes.py",
    "backend.py",
    "selftest.py",
    "README.md",
)

DEFAULT_PORT_DIR = REPO_ROOT / "src" / "prompt_master" / "prompt_engine" / "upstream"
DEFAULT_NOTES = REPO_ROOT / "UPSTREAM_DIFF_NOTES.md"
DEFAULT_COMMIT_FILE = (
    REPO_ROOT / "src" / "prompt_master" / "prompt_engine" / "UPSTREAM_COMMIT.txt"
)

# An import-ish line: import machinery only, never behavior.
IMPORT_LINE = re.compile(
    r"""^\s*(?:
          from\s+[.\w]+\s+import\s.*
        | import\s+[.\w]+.*
        | \)\s*$            # closing paren of a parenthesised import list
        | [\w\s,]+,?\s*$    # a name inside a parenthesised import list
    )$""",
    re.VERBOSE,
)

# Lines that are always safe to ignore inside an otherwise import-only hunk.
IGNORABLE_LINE = re.compile(r"^\s*(?:#.*)?$")

# Markers that make a hunk behaviorally sensitive. Even with an approved
# exception these are reported loudly, because they are exactly the edits the
# porting contract forbids.
SENSITIVE_PATTERNS = (
    (re.compile(r"""["']"""), "string literal"),
    (re.compile(r"[\[{]"), "list/dict literal"),
    (re.compile(r"\b(?:if|elif|else|for|while|return|yield|and|or|not)\b"), "control flow"),
    (re.compile(r"^\s*(?:def|class)\s"), "definition"),
    (re.compile(r"^[A-Z_][A-Z0-9_]*\s*="), "module constant"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[str]:
    """Read a file, normalizing only line endings and trailing newline."""
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


@dataclass
class Exception_:
    """One approved difference declared in UPSTREAM_DIFF_NOTES.md."""

    module: str
    original_lines: tuple[int, int]
    new_lines: tuple[int, int]
    reason: str
    behavior_confirmed: bool

    def covers(self, old_range: tuple[int, int], new_range: tuple[int, int]) -> bool:
        return (
            self.original_lines[0] <= old_range[0]
            and old_range[1] <= self.original_lines[1]
            and self.new_lines[0] <= new_range[0]
            and new_range[1] <= self.new_lines[1]
        )


@dataclass
class Hunk:
    old_range: tuple[int, int]
    new_range: tuple[int, int]
    removed: list[str]
    added: list[str]

    @property
    def import_only(self) -> bool:
        lines = self.removed + self.added
        if not lines:
            return False
        return all(
            IGNORABLE_LINE.match(line) or IMPORT_LINE.match(line) for line in lines
        )

    @property
    def sensitive(self) -> list[str]:
        hits: list[str] = []
        for line in self.removed + self.added:
            for pattern, label in SENSITIVE_PATTERNS:
                if pattern.search(line) and label not in hits:
                    hits.append(label)
        return hits

    def describe(self) -> str:
        return (
            f"upstream lines {self.old_range[0]}-{self.old_range[1]} -> "
            f"ported lines {self.new_range[0]}-{self.new_range[1]}"
        )


@dataclass
class ModuleResult:
    module: str
    required: bool
    status: str  # ok | missing | approved | unapproved | absent-upstream
    upstream_sha: str | None = None
    ported_sha: str | None = None
    diff: str = ""
    problems: list[str] = field(default_factory=list)
    approved: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status in {"missing", "unapproved", "absent-upstream"}


def parse_notes(path: Path) -> tuple[list[Exception_], list[str]]:
    """Parse approved exceptions out of UPSTREAM_DIFF_NOTES.md.

    Each exception is a markdown section shaped like::

        ### accents.py
        - Original lines: 1-4
        - New lines: 1-6
        - Reason: rewrote the ComfyUI import as a package-relative import
        - Behavior unchanged: yes
    """
    if not path.exists():
        return [], [f"{path.name} not found; no approved differences are declared"]

    exceptions: list[Exception_] = []
    warnings: list[str] = []
    current: str | None = None
    pending: dict[str, str] = {}

    def flush() -> None:
        if current is None or not pending:
            return
        original = pending.get("original lines")
        new = pending.get("new lines")
        reason = pending.get("reason", "").strip()
        confirmed = pending.get("behavior unchanged", "").strip().lower()
        if not original or not new:
            warnings.append(f"{current}: exception missing a line range; ignored")
            pending.clear()
            return
        try:
            old_range = parse_range(original)
            new_range = parse_range(new)
        except ValueError:
            warnings.append(f"{current}: unparsable line range; ignored")
            pending.clear()
            return
        if not reason:
            warnings.append(f"{current}: exception missing a Reason; ignored")
            pending.clear()
            return
        exceptions.append(
            Exception_(
                module=current,
                original_lines=old_range,
                new_lines=new_range,
                reason=reason,
                behavior_confirmed=confirmed in {"yes", "true", "confirmed"},
            )
        )
        pending.clear()

    for raw in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^#{2,4}\s+(\S+\.(?:py|md))\s*$", raw.strip())
        if heading:
            flush()
            current = heading.group(1)
            continue
        field_match = re.match(r"^\s*[-*]\s*([A-Za-z ]+?)\s*:\s*(.+?)\s*$", raw)
        if field_match and current:
            key = field_match.group(1).strip().lower()
            if key in pending:
                flush()
            pending[key] = field_match.group(2)
    flush()
    return exceptions, warnings


def read_provenance(path: Path) -> tuple[str, dict[str, str], list[str]]:
    """Read UPSTREAM_COMMIT.txt.

    Accepts either a bare 40-character commit SHA (upstream arrived as a clone)
    or the provenance form: a ``git-commit:`` line plus a ``sha256  filename``
    manifest (upstream arrived as an archive, so there is no revision to pin and
    the content is pinned instead).
    """
    if not path.exists():
        return "", {}, [f"{path.name} is missing; the upstream revision is unrecorded"]

    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if re.fullmatch(r"[0-9a-f]{40}", stripped):
        return stripped, {}, []

    commit = ""
    match = re.search(r"(?mi)^git-commit:\s*(\S+)\s*$", text)
    if match:
        commit = match.group(1)

    manifest = {
        name: digest
        for digest, name in re.findall(r"(?m)^([0-9a-f]{64})\s+(\S+)\s*$", text)
    }

    warnings: list[str] = []
    if not commit and not manifest:
        warnings.append(
            f"{path.name} records neither a commit SHA nor a digest manifest; "
            f"the upstream source is unpinned"
        )
    elif commit.upper() == "UNKNOWN" and not manifest:
        warnings.append(
            f"{path.name} records no commit and no digests to pin instead"
        )
    elif commit.upper() == "UNKNOWN":
        warnings.append(
            f"{path.name} has no upstream commit SHA; pinned by content digest "
            f"instead ({len(manifest)} files)"
        )
    return commit, manifest, warnings


def parse_range(value: str) -> tuple[int, int]:
    value = value.strip()
    if value.lower() in {"none", "n/a", "-"}:
        return (0, 0)
    match = re.match(r"^(\d+)\s*(?:[-–]\s*(\d+))?$", value)
    if not match:
        raise ValueError(value)
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    return (start, end)


def build_hunks(upstream: list[str], ported: list[str]) -> list[Hunk]:
    matcher = difflib.SequenceMatcher(a=upstream, b=ported, autojunk=False)
    hunks: list[Hunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunks.append(
            Hunk(
                old_range=(i1 + 1, max(i1 + 1, i2)),
                new_range=(j1 + 1, max(j1 + 1, j2)),
                removed=upstream[i1:i2],
                added=ported[j1:j2],
            )
        )
    return hunks


def check_module(
    module: str,
    required: bool,
    upstream_dir: Path,
    port_dir: Path,
    exceptions: list[Exception_],
) -> ModuleResult:
    upstream_path = upstream_dir / module
    ported_path = port_dir / module

    if not upstream_path.exists():
        return ModuleResult(
            module,
            required,
            "absent-upstream",
            problems=[f"not present in the upstream checkout at {upstream_path}"],
        )
    if not ported_path.exists():
        return ModuleResult(
            module,
            required,
            "missing",
            upstream_sha=sha256(upstream_path),
            problems=[f"not vendored at {ported_path}"],
        )

    result = ModuleResult(
        module,
        required,
        "ok",
        upstream_sha=sha256(upstream_path),
        ported_sha=sha256(ported_path),
    )
    if result.upstream_sha == result.ported_sha:
        return result

    upstream_lines = read_lines(upstream_path)
    ported_lines = read_lines(ported_path)
    result.diff = "\n".join(
        difflib.unified_diff(
            upstream_lines,
            ported_lines,
            fromfile=f"upstream/{module}",
            tofile=f"ported/{module}",
            lineterm="",
        )
    )

    if upstream_lines == ported_lines:
        # Byte difference was line endings or a trailing newline only.
        result.approved.append("line-ending normalization only")
        result.status = "approved"
        return result

    module_exceptions = [exc for exc in exceptions if exc.module == module]
    for hunk in build_hunks(upstream_lines, ported_lines):
        if hunk.import_only:
            result.approved.append(f"import rewrite ({hunk.describe()})")
            continue
        match = next(
            (
                exc
                for exc in module_exceptions
                if exc.covers(hunk.old_range, hunk.new_range)
            ),
            None,
        )
        sensitive = hunk.sensitive
        if match is None:
            detail = f" [touches {', '.join(sensitive)}]" if sensitive else ""
            result.problems.append(
                f"undeclared change at {hunk.describe()}{detail}"
            )
            continue
        if not match.behavior_confirmed:
            result.problems.append(
                f"exception at {hunk.describe()} lacks a "
                f"'Behavior unchanged: yes' confirmation"
            )
            continue
        note = f"approved: {match.reason} ({hunk.describe()})"
        if sensitive:
            note += f" [touches {', '.join(sensitive)} -- verify against parity harness]"
        result.approved.append(note)

    result.status = "unapproved" if result.problems else "approved"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--upstream",
        required=True,
        type=Path,
        help="path to an untouched Prompt-Master-LD checkout",
    )
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT_DIR)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--commit-file", type=Path, default=DEFAULT_COMMIT_FILE)
    parser.add_argument("--show-diff", action="store_true", help="print unified diffs")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    if not args.upstream.is_dir():
        print(f"error: upstream checkout not found: {args.upstream}", file=sys.stderr)
        return 2
    if not args.port.is_dir():
        print(f"error: vendored engine not found: {args.port}", file=sys.stderr)
        return 2

    exceptions, warnings = parse_notes(args.notes)

    results = [
        check_module(m, True, args.upstream, args.port, exceptions)
        for m in ENGINE_MODULES
    ] + [
        check_module(m, False, args.upstream, args.port, exceptions)
        for m in REFERENCE_MODULES
    ]

    recorded_commit, digest_manifest, commit_warnings = read_provenance(args.commit_file)
    warnings.extend(commit_warnings)

    # When the provenance file carries per-file digests (the case when upstream
    # arrived as an archive rather than a clone), the digests are the pin: check
    # the reference checkout still matches what was recorded.
    for r in results:
        recorded = digest_manifest.get(r.module)
        if recorded and r.upstream_sha and recorded != r.upstream_sha:
            r.problems.append(
                f"upstream checkout does not match the digest recorded in "
                f"{args.commit_file.name} (recorded {recorded[:16]}…, "
                f"found {r.upstream_sha[:16]}…)"
            )
            if r.status == "ok":
                r.status = "unapproved"

    failures = [r for r in results if r.failed and r.required]
    reference_failures = [r for r in results if r.failed and not r.required]

    if args.as_json:
        print(
            json.dumps(
                {
                    "upstream_commit": recorded_commit,
                    "warnings": warnings,
                    "modules": [
                        {
                            "module": r.module,
                            "required": r.required,
                            "status": r.status,
                            "upstream_sha256": r.upstream_sha,
                            "ported_sha256": r.ported_sha,
                            "problems": r.problems,
                            "approved": r.approved,
                        }
                        for r in results
                    ],
                    "ok": not failures,
                },
                indent=2,
            )
        )
        return 1 if failures else 0

    print(f"upstream checkout : {args.upstream}")
    print(f"vendored engine   : {args.port}")
    print(f"pinned revision   : {recorded_commit or '(unrecorded)'}")
    print()
    print(f"{'module':<16} {'status':<10} {'upstream sha256':<18} {'ported sha256'}")
    print("-" * 70)
    for r in results:
        tag = r.module if r.required else f"{r.module} (ref)"
        print(
            f"{tag:<16} {r.status:<10} "
            f"{(r.upstream_sha or '')[:16]:<18} {(r.ported_sha or '')[:16]}"
        )

    for r in results:
        if r.approved or r.problems:
            print(f"\n{r.module}:")
            for note in r.approved:
                print(f"  ok   {note}")
            for problem in r.problems:
                print(f"  FAIL {problem}")
            if args.show_diff and r.diff:
                print("\n".join(f"    {line}" for line in r.diff.splitlines()))

    for warning in warnings:
        print(f"\nwarning: {warning}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} required module(s) differ without approval")
        return 1
    if reference_failures:
        print(
            f"passed with {len(reference_failures)} reference module(s) missing "
            f"or changed"
        )
        return 0
    print("passed: all vendored modules match upstream or are explicitly approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
