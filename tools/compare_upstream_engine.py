"""Parity harness: the untouched upstream engine vs the standalone port.

Loads BOTH engines into one process — an untouched Prompt-Master-LD checkout and
``prompt_engine.upstream`` — drives them with the same inputs, and compares
every output that reaches a render:

  * system prompt          (the whole brief, character for character)
  * user prompt
  * base negative prompt
  * frame count
  * word budget, token budget, beat budget, spoken-line count
  * selected conditional blocks (which laws fired)
  * output format contract
  * smart-negative filtering

Normalization is limited to what packaging genuinely forces: line endings. No
wording, punctuation, rule order or missing block is normalized away, and any
difference at all fails the run.

Usage::

    python tools/compare_upstream_engine.py --upstream ../Prompt-Master-LD
    python tools/compare_upstream_engine.py --upstream ../Prompt-Master-LD --verbose
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import itertools
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Engine modules the harness drives. imaging.py is excluded on purpose: it is
# the one module with an approved difference, it needs Torch upstream, and it
# contributes nothing to prompt text.
NEEDED = ("accents", "negative", "hands", "identity", "styles", "shotscript",
          "cinematics", "music", "wardrobe", "brain")


def load_reference(checkout: Path) -> types.ModuleType:
    """Import an untouched checkout as a package so its relative imports work."""
    pkg = types.ModuleType("_pmld_ref")
    pkg.__path__ = [str(checkout)]
    sys.modules["_pmld_ref"] = pkg
    loaded = {}
    for name in NEEDED:
        path = checkout / f"{name}.py"
        if not path.is_file():
            raise SystemExit(f"error: {path} not found — is --upstream a Prompt-Master-LD checkout?")
        spec = importlib.util.spec_from_file_location(f"_pmld_ref.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_pmld_ref.{name}"] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    pkg.__dict__.update(loaded)
    return pkg


def normalize(value):
    """Line endings only. Nothing else is touched."""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return value


# ── the control matrix ───────────────────────────────────────────────────────

def cases(reference) -> list[dict]:
    """Configs covering every control, its boundaries, and the interactions the
    upstream comments call out as load-bearing."""
    accents = reference.accents.ACCENT_KEYS
    styles = reference.styles.STYLE_KEYS
    cameras = reference.cinematics.CAMERA_KEYS
    transitions = reference.cinematics.TRANSITION_KEYS
    musics = ["auto"] + list(reference.music.MUSIC_KEYS)
    formats = reference.shotscript.FORMATS
    strengths = list(reference.accents.STRENGTHS)

    base = dict(mode="t2v", pov="off", accent="off", accent_strength="natural",
                fps=24, seconds=12.0, dialogue=20, wardrobe="auto", undress=False,
                seed=7, intent="a woman speaks at a window", camera="off",
                transition="off", music="off", music_bg=False, lexicon="",
                fmt="flowing", style="off", style_hint="", has_image=False)

    out: list[dict] = []

    def add(**kw):
        out.append({**base, **kw})

    # every value of every control, one at a time
    for accent in accents:
        add(accent=accent, dialogue=40)
    for strength in strengths:
        for accent in ("jamaican", "french", "new_york", "off"):
            add(accent=accent, accent_strength=strength, dialogue=40)
    for style in styles:
        add(style=style)
    for camera in cameras:
        add(camera=camera)
    for transition in transitions:
        add(transition=transition)
    for music in musics:
        add(music=music, dialogue=30)
        add(music=music, music_bg=True, dialogue=30)
    for fmt in formats:
        for seconds in (2.0, 12.0, 50.0):
            add(fmt=fmt, seconds=seconds)

    # POV combinations, both modes, with and without another person on screen
    for pov, mode, intent in itertools.product(
            ("off", "male", "female"), ("t2v", "i2v"),
            ("i reach for the door", "she leans close and looks at me",
             "pov driving through the city alone")):
        add(pov=pov, mode=mode, intent=intent, dialogue=40,
            has_image=(mode == "i2v"))

    # dialogue levels including both boundaries and the performance threshold
    for dialogue in (0, 1, 19, 20, 45, 69, 70, 71, 100):
        add(dialogue=dialogue)
        add(dialogue=dialogue, pov="female", intent="i look at her")

    # duration and fps boundaries
    for seconds in (1.0, 2.0, 3.0, 12.0, 16.0, 30.0, 50.0, 60.0):
        for fps in (8, 16, 24, 40, 60):
            add(seconds=seconds, fps=fps, dialogue=45)

    # wardrobe and undress
    for wardrobe in ("auto", "off", "her", "him"):
        for undress in (False, True):
            for mode in ("t2v", "i2v"):
                add(wardrobe=wardrobe, undress=undress, mode=mode,
                    has_image=(mode == "i2v"), intent="she undresses slowly")
    add(wardrobe="her", intent="a woman in a red dress dances")
    add(wardrobe="him", intent="the man in a black suit walks in")

    # image authority: i2v with and without the still actually on the wire
    for has_image in (True, False):
        for hint in ("", "cartoon"):
            add(mode="i2v", has_image=has_image, style_hint=hint,
                intent="bring the still to life")
            add(mode="i2v", has_image=has_image, style_hint=hint,
                style="anime_cel", accent="australian", dialogue=40,
                intent="she talks to the camera")

    # cue gates fire off the intent, not off a substring
    for intent in ("she dances hard", "she sits quietly", "a bracelet on her wrist",
                   "she looks in the mirror", "a glass of wine", "she turns away",
                   "he strolls past", "a brass handle", "barefoot on the sand",
                   'a man in a car says "that is not it"'):
        add(intent=intent, dialogue=40)

    # conflicting controls
    add(pov="female", style="spongebob", accent="jamaican", music="auto",
        dialogue=80, camera="locked_off", transition="hard_cut", undress=True,
        intent="i dance with her in a club", seconds=30, fmt="shotscript")
    add(mode="i2v", has_image=True, style="film_noir", wardrobe="her",
        accent="rp_british", dialogue=0, intent="she waits")
    add(dialogue=0, accent="polish", music="techno", intent="a woman walks")
    add(pov="male", dialogue=100, accent="aave", music="auto",
        intent="i rap into the mirror", seconds=20)

    # lexicon: filtered against the intent
    lex = "Ada = a tall welder in a leather apron\nBoris = a bald bouncer"
    add(lexicon=lex, intent="Ada walks into the bar")
    add(lexicon=lex, intent="a stranger walks into the bar")
    add(lexicon="", intent="Ada walks into the bar")

    # "off" everywhere — the default fallbacks
    add()
    add(mode="i2v", has_image=True)

    return out


# ── comparison ───────────────────────────────────────────────────────────────

BLOCKS = ("FIRST PERSON", "POV HANDS", "VOICE", "NO SPEECH", "SPEECH", "MUSIC",
          "MOVEMENT", "MIRROR", "WARDROBE", "UNDRESS", "CAST LOOK", "VIEWER LOOK",
          "VISUAL STYLE", "STYLE FIRST", "CAMERA —", "TRANSITION —", "ANATOMY",
          "ORIENTATION", "LIGHT & OPTICS", "OUTPUT CONTRACT", "LEXICON",
          "PERFORMED VOCAL", "PACING", "OPENING", "MEDIUM")


def outputs(engine, case: dict) -> dict:
    """Everything one config produces, from one side of the comparison."""
    brain = engine.brain
    system = brain.build_system(**case)
    wsec = brain.write_seconds(case["seconds"])
    pct = brain.talk_pct(case["dialogue"])
    nb_lo, nb_hi = brain.beat_budget(wsec)
    return {
        "system": system,
        "user": brain.build_user(intent=case["intent"], mode=case["mode"],
                                 has_image=case["has_image"],
                                 style_hint=case["style_hint"]),
        "negative": brain.build_negative(
            pov=case["pov"], dialogue=case["dialogue"], undress=case["undress"],
            fmt=case["fmt"], transition=case["transition"], intent=case["intent"],
            extra="", camera=case["camera"], style=case["style"], mode=case["mode"]),
        "frames": brain.frame_count(case["fps"], case["seconds"]),
        "max_tokens": brain.max_tokens(case["seconds"], case["dialogue"], fmt=case["fmt"]),
        "word_budget": list(brain.word_budget(wsec, case["dialogue"])),
        "beat_budget": [nb_lo, nb_hi],
        "dialogue_lines": brain.dialogue_lines(wsec, pct, beats=nb_lo),
        "write_seconds": wsec,
        "talk_pct": pct,
        "format_contract": engine.shotscript.contract(
            case["fmt"], nb_lo=nb_lo, nb_hi=nb_hi, lo=brain.word_budget(wsec, case["dialogue"])[0],
            hi=brain.word_budget(wsec, case["dialogue"])[1], seconds=case["seconds"]),
        "blocks": [b for b in BLOCKS if b in system],
    }


# Every option table, compared whole. The config matrix is generated FROM the
# reference engine, so an option missing on one side would simply never be
# driven — comparing the tables themselves is what makes a dropped accent, a
# shortened genre list or a renamed key fail instead of silently passing.
CONSTANTS = [
    ("accents", "ACCENTS"), ("accents", "ACCENT_KEYS"), ("accents", "STRENGTHS"),
    ("accents", "_VARIETIES"), ("accents", "_LABEL_FIX"), ("accents", "_DENSITY"),
    ("accents", "_EXTRA"),
    ("music", "MUSIC"), ("music", "MUSIC_KEYS"), ("music", "MUSIC_LABELS"),
    ("music", "_FOR_ACCENT"), ("music", "_VOCAL"), ("music", "_PERF"),
    ("styles", "STYLES"), ("styles", "STYLE_KEYS"), ("styles", "STYLE_LABELS"),
    ("styles", "STYLE_GROUPS"), ("styles", "_NON_PHOTO"), ("styles", "_UNIVERSES"),
    ("styles", "_ANIMATION"), ("styles", "_LIVE"), ("styles", "_STYLE_ENFORCE"),
    ("styles", "_STYLE_ABSORB"),
    ("cinematics", "CAMERA"), ("cinematics", "CAMERA_KEYS"),
    ("cinematics", "CAMERA_LABELS"), ("cinematics", "TRANSITION"),
    ("cinematics", "TRANSITION_KEYS"), ("cinematics", "TRANSITION_LABELS"),
    ("cinematics", "_NEG_CAM"), ("cinematics", "_CAM_ENFORCE"),
    ("shotscript", "FORMATS"), ("shotscript", "FORMAT_LABELS"),
    ("negative", "_CORE"), ("negative", "_CORE_TEMPORAL"), ("negative", "_POV"),
    ("negative", "_SILENT"), ("negative", "_UNDRESS"), ("negative", "_MOTION"),
    ("negative", "_TRANS_SMOOTH"), ("negative", "_TRANS_CUT"),
    ("negative", "_FORMAT_TEXT"), ("negative", "CUT_TRANSITIONS"),
    ("negative", "AUTO_SYSTEM"), ("negative", "_BANNED_TOKENS"),
    ("hands", "_LAW"), ("hands", "_NEG"),
    ("identity", "_REGIONS"), ("identity", "_ACCENT_REGION"),
    ("identity", "_REGION_ALIAS"), ("identity", "_BUILD"), ("identity", "_BUILD_HIM"),
    ("identity", "_BUST"), ("identity", "_HAIR_LEN"), ("identity", "_HAIR_TEX"),
    ("brain", "_CORE"), ("brain", "_SHOT"), ("brain", "_SPEECH"),
    ("brain", "_PERFORMANCE"), ("brain", "_POV_RULES"), ("brain", "_POV_VOICE"),
    ("brain", "_POV_VOICE_SOLO"), ("brain", "_MIRROR"), ("brain", "_MOVEMENT"),
    ("brain", "_I2V_OPEN"), ("brain", "_I2V_OPEN_BLIND"), ("brain", "_T2V_OPEN"),
    ("brain", "_ANATOMY_CUES"), ("brain", "_ORIENT_CUES"), ("brain", "_MIRROR_CUES"),
    ("brain", "_MOTION_CUES"), ("brain", "_VOICE_ONE_LANE"), ("brain", "_VOICE_REGISTER"),
]


def constant(engine, module: str, name: str):
    mod = getattr(engine, module, None)
    if mod is None:
        return "<module missing>"
    if not hasattr(mod, name):
        return "<constant missing>"
    value = getattr(mod, name)
    # Sets are unordered; everything else keeps its declared order.
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    return value


SMART_CASES = [
    ("daylight, overcast, locked tripod, wet asphalt, x, "
     "a phrase far too long to be a visual term, daylight, NEGATIVE prompt",
     "The camera holds on wet asphalt while rain falls through neon."),
    ("<think>hmm</think>```\nphotoreal skin, live action, cel shading\n```",
     "A cel-shaded anime street, flat colour, hard shadow shapes."),
    ("", "an empty pass"),
    ("no daylight, not dry skin, avoid wide master shot",
     "A tight close-up at night, wet skin under rain."),
    ("one, two, three, four, five, six, seven, eight, nine, ten, eleven, "
     "twelve, thirteen, fourteen, fifteen, sixteen", "a limit test"),
]


@dataclass
class Mismatch:
    case: int
    field: str
    reference: object
    ported: object
    config: dict

    def render(self, verbose: bool) -> str:
        where = "constant" if self.case < 0 else f"case {self.case}"
        head = f"{where}: {self.field} differs"
        if isinstance(self.reference, str) and isinstance(self.ported, str):
            diff = "\n".join(itertools.islice(difflib.unified_diff(
                self.reference.split("\n"), self.ported.split("\n"),
                fromfile="upstream", tofile="ported", lineterm=""), 0, 40))
            body = diff
        else:
            body = f"  upstream: {self.reference!r}\n  ported:   {self.ported!r}"
        cfg = ""
        if verbose:
            cfg = "\n  config: " + json.dumps(
                {k: v for k, v in self.config.items() if v not in ("", False, "off")},
                sort_keys=True)
        return f"{head}{cfg}\n{body}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--upstream", required=True, type=Path,
                        help="path to an untouched Prompt-Master-LD checkout")
    parser.add_argument("--verbose", action="store_true",
                        help="print the config behind each mismatch")
    parser.add_argument("--max-report", type=int, default=10)
    args = parser.parse_args(argv)

    if not args.upstream.is_dir():
        print(f"error: upstream checkout not found: {args.upstream}", file=sys.stderr)
        return 2

    reference = load_reference(args.upstream.resolve())

    ported = types.SimpleNamespace()
    for name in NEEDED:
        setattr(ported, name, importlib.import_module(
            f"prompt_master.prompt_engine.upstream.{name}"))

    matrix = cases(reference)
    mismatches: list[Mismatch] = []
    compared = 0

    # Option tables first: if these differ, the matrix below is already unfair.
    for module, name in CONSTANTS:
        compared += 1
        a = constant(reference, module, name)
        b = constant(ported, module, name)
        if a != b:
            mismatches.append(Mismatch(-1, f"{module}.{name}", a, b, {}))

    for index, case in enumerate(matrix):
        left = outputs(reference, case)
        right = outputs(ported, case)
        for field in left:
            compared += 1
            a, b = normalize(left[field]), normalize(right[field])
            if a != b:
                mismatches.append(Mismatch(index, field, a, b, case))

    # smart-negative filtering, over the same raw model output
    for index, (raw, script) in enumerate(SMART_CASES):
        compared += 2
        a = normalize(reference.negative.clean_auto(raw, script=script))
        b = normalize(ported.negative.clean_auto(raw, script=script))
        if a != b:
            mismatches.append(Mismatch(index, "clean_auto", a, b, {"raw": raw}))
        a = reference.negative.auto_messages(script)
        b = ported.negative.auto_messages(script)
        if a != b:
            mismatches.append(Mismatch(index, "auto_messages", a, b, {"script": script}))

    print(f"upstream checkout : {args.upstream}")
    print(f"configs compared  : {len(matrix)}")
    print(f"smart-neg cases   : {len(SMART_CASES)}")
    print(f"field comparisons : {compared}")
    print()

    if mismatches:
        print(f"FAILED: {len(mismatches)} mismatch(es)\n")
        for m in mismatches[:args.max_report]:
            print(m.render(args.verbose))
            print()
        if len(mismatches) > args.max_report:
            print(f"... and {len(mismatches) - args.max_report} more")
        return 1

    print("passed: the ported engine is output-identical to the upstream checkout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
