"""MiniMax-H3 prompts — a second engine, written to MiniMax's own guide.

This module has nothing to do with the LTX engine and deliberately touches none
of it. ``prompt_engine.upstream`` is a byte-for-byte port whose every string is
pinned by ``tools/check_upstream_sync.py``; H3 wants a different prompt, in a
different shape, obeying different laws, and the only honest way to have both is
two engines that never share a line. Nothing here imports ``upstream``, and
nothing in ``upstream`` knows this file exists.

**Where the rules come from.** MiniMax publishes ``Video Prompt Writing Guide
(T2VA / I2VA / FL2VA / L2VA)`` with the H3 weights, and that guide is what this
module implements. It is specific in a way most prompt advice is not, and the
specifics are the reason this is a mode of its own rather than a style preset on
the existing one:

* An H3 prompt is **three named fields**, always in the same order:
  ``integrated_multimodal_description``, ``overall_soundscape``,
  ``non_diegetic_music``. They are not headings for a human — they are the
  format the model was trained to read.
* The description is a **timeline**, not a paragraph. ``[Shot 1]`` opens it and
  carries no timestamp; every later shot opens with ``[Shot N] At MM:SS.mmm,``
  and a cut phrase, at a strictly increasing time inside the duration.
* Sound is **directed, not mentioned**. Dialogue, singing and any audio with a
  source in the scene — a radio, a television, a phone — belong on the timeline
  with the picture; ambience belongs in the soundscape; score belongs in the
  music field, described by instrumentation rather than by mood. Saying the same
  sound twice is a fault, not emphasis. Either sound field may be exactly
  ``N/A``, and nothing else stands in for it.
* Speech has a **syntax**: a stable ``(S1)`` per voice, the speaker and delivery
  written outside the tag, and only a language tag and the words themselves
  inside ``<d>…</d>``, preserved to the punctuation mark. A line that crosses a
  cut is marked ``<scenetrans>`` on both sides; one the video cuts off is marked
  ``<cutoff>``.
* Three of the four modes open with an **instruction line**, then one blank
  line, then the fields. Each mode has its own wording, and the guide's wording
  is reproduced here verbatim — including where it brackets ``<Picture 1>`` and
  where it does not.

**What is written here and what is written by the model.** Everything mechanical
is assembled in this file: the instruction line, the field labels, their order,
the blank line after the instruction. Those are the parts a language model gets
subtly wrong on a bad day, and they are also the parts with one correct answer,
so there is no reason to ask. One of them cannot be settled before the writing,
though — the instruction names ``[Shot N]``, *the actual final shot* — so the
line is built after the brief comes back, from the shot numbers the writer
actually used.

**Checks, not enforcement.** ``checks()`` re-reads the finished brief and says
what looks wrong — a first shot that grew a timestamp, cut times that go
backwards or run past the end, a soundscape that ran to eight sentences, an
``N/A`` where a sound was asked for, a brief over the 7,000-character limit the
API takes. It never rewrites: a brief that breaks a rule on purpose is still the
writer's to keep, and a silent correction is how you end up with a prompt nobody
chose.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# ── the format ───────────────────────────────────────────────────────────────

# The three fields, in the order the guide prints them. These are keys the model
# reads, not headings, so they are spelled exactly as MiniMax spells them.
DESCRIPTION = "integrated_multimodal_description"
SOUNDSCAPE = "overall_soundscape"
MUSIC = "non_diegetic_music"
FIELDS = (DESCRIPTION, SOUNDSCAPE, MUSIC)

# What a sound field says when there is no sound to describe. The guide gives
# this exact token for both of them, so it is a value rather than a sentence:
# "there is no music" is prose about silence, and N/A is silence.
NOT_APPLICABLE = "N/A"

# What the product surface accepts: 5-15 seconds at 24 FPS, and a prompt of at
# most 7,000 characters. The duration bounds are the spin box's range; the
# character limit is a check rather than a truncation, because a brief three
# characters over is a sentence to cut by hand, not by machine.
MIN_SECONDS, MAX_SECONDS = 5.0, 15.0
FPS = 24
CHARACTER_LIMIT = 7000

# ── modes ────────────────────────────────────────────────────────────────────
# The guide's four tasks. Each is the T2VA body plus, for the three anchored
# ones, an instruction line and a rule about where the picture sits in time.

T2VA, I2VA, FL2VA, L2VA = "t2va", "i2va", "fl2va", "l2va"

MODES: list[tuple[str, str]] = [
    (T2VA, "Text to video — no frames (T2VA)"),
    (I2VA, "First frame (I2VA)"),
    (FL2VA, "First and last frame (FL2VA)"),
    (L2VA, "Last frame (L2VA)"),
]

# ── look ─────────────────────────────────────────────────────────────────────
# The guide's own list of common styles, spelled the way it spells them — its
# example opens "[Shot 1] Live-action, cinematic, …", and two of these are
# tokens that combine, so the combination is offered as well.

STYLES: list[tuple[str, str]] = [
    ("auto", "From the intent, or from the picture"),
    ("live_action_cinematic", "Live-action, cinematic"),
    ("cinematic", "Cinematic"),
    ("live_action", "Live-action"),
    ("animation_2d", "2D-animated"),
    ("cg_3d", "3D CG"),
    ("claymation", "Claymation"),
    ("watercolor", "Watercolor"),
    ("vintage_film", "Vintage film"),
]

STYLE_PHRASES: dict[str, str] = {
    "live_action_cinematic": "Live-action, cinematic",
    "cinematic": "Cinematic",
    "live_action": "Live-action",
    "animation_2d": "2D-animated",
    "cg_3d": "3D CG",
    "claymation": "claymation",
    "watercolor": "watercolor",
    "vintage_film": "vintage film",
}

# H3's camera vocabulary and what each move means, transcribed from the guide's
# own table. The descriptions travel into the prompt with the names: "Pedestal
# Up" and "Tilt Up" are different instructions to a camera and the same guess to
# a writer that was only given the label.
CAMERA_MOVES: tuple[tuple[str, str], ...] = (
    ("Zoom In / Zoom Out", "the focal length changes while the camera body stays still"),
    ("Push In / Pull Out", "the camera moves forward / backward"),
    ("Pan Left / Pan Right", "the camera stays in place while the lens pivots horizontally"),
    ("Truck Left / Truck Right", "the camera translates horizontally"),
    ("Tilt Up / Tilt Down", "the camera stays in place while the lens pivots vertically"),
    ("Pedestal Up / Pedestal Down", "the entire camera moves upward / downward"),
    ("Arc Shot", "the camera moves in an arc around the subject"),
    ("Tracking Shot", "the camera follows a moving subject"),
    ("Static Shot", "camera position and lens remain still"),
    ("Shake Slightly / Shake Strongly", "slight / strong camera shake"),
    ("POV", "the subject's point of view"),
    ("Roll Clockwise / Roll Counterclockwise", "the camera rolls around the lens axis"),
)

# One entry per move a shot can actually be given, which is each side of the
# paired ones taken on its own.
CAMERA_NAMES: tuple[str, ...] = tuple(
    part.strip() for name, _ in CAMERA_MOVES for part in name.split("/")
)

CAMERAS: list[tuple[str, str]] = (
    [("auto", "From the intent")]
    + [(name.lower().replace(" ", "_"), name) for name in CAMERA_NAMES]
)

# The guide's exact expressions, and its rule for leaving them out: medium
# amplitude and normal speed are the unmarked case, and writing them spends
# words saying nothing.
AMPLITUDES: list[tuple[str, str]] = [
    ("medium", "Medium — left unwritten"), ("small", "Small"), ("large", "Large"),
]
AMPLITUDE_PHRASES: dict[str, str] = {"small": "with small amplitude",
                                     "large": "with large amplitude"}
SPEEDS: list[tuple[str, str]] = [
    ("normal", "Normal — left unwritten"), ("slow", "Slow"), ("fast", "Fast"),
]
SPEED_PHRASES: dict[str, str] = {"slow": "at slow speed", "fast": "at fast speed"}

# Ordinary cuts are the default and the guide gives five interchangeable
# phrasings for them. The other three are named as things to use only when the
# user explicitly requested them — choosing one here is that request.
CUTS: list[tuple[str, str]] = [
    ("cut", "Ordinary cut"),
    ("cross_dissolve", "Cross-dissolve"),
    ("fade", "Fade"),
    ("wipe", "Wipe"),
]

CUT_PHRASES: dict[str, str] = {
    "cut": "the camera cuts to",
    "cross_dissolve": "the shot cross-dissolves to",
    "fade": "the shot fades to",
    "wipe": "the shot wipes to",
}

CUT_ALTERNATES = (
    "the camera cuts to", "the shot cuts to", "the shot transitions to",
    "the shot changes to", "the shot switches to",
)

# A 5-15 second clip does not hold many shots, and the guide is explicit that a
# cut has to earn itself with new information — a closer look at the same thing
# is a camera move, not a cut. "Auto" leaves the count to the beat, except in
# the two modes that interpolate towards a frame, where the guide asks for one.
SHOTS: list[tuple[str, str]] = (
    [("auto", "As many as the beat needs")]
    + [(str(n), "1 shot" if n == 1 else f"{n} shots") for n in range(1, 7)]
)

# ── voice ────────────────────────────────────────────────────────────────────
# Spoken content goes inside <d>[Language] … </d>, and the tag has to name a
# real language. This is the list the drop-down offers; the value is the tag.

LANGUAGES: list[tuple[str, str]] = [
    (name, name) for name in (
        "English", "Mandarin", "Cantonese", "Japanese", "Korean", "Spanish",
        "French", "German", "Italian", "Portuguese", "Russian", "Arabic",
        "Hindi", "Thai", "Vietnamese", "Indonesian",
    )
]

# How much of the clip is spoken over. The guide's own cases carry one or two
# lines in six to ten seconds, so this is a budget rather than a licence, and
# ``spoken_lines`` turns it into a number the brief is written against.
TALK: list[tuple[str, str]] = [
    ("none", "None — nobody speaks"),
    ("sparse", "Sparse — a line or two"),
    ("steady", "Steady — a real exchange"),
    ("dense", "Dense — talking throughout"),
]

TALK_RATE: dict[str, float] = {"none": 0.0, "sparse": 0.14, "steady": 0.28, "dense": 0.45}

MAX_SPEAKERS = 6

# ── sound ────────────────────────────────────────────────────────────────────

SOUNDSCAPES: list[tuple[str, str]] = [
    ("scene", "The sound the scene makes"),
    # The guide allows N/A here only when complete silence was actually asked
    # for, so asking for it is what this option is.
    ("silence", "Complete silence — N/A"),
]

MUSIC_MODES: list[tuple[str, str]] = [
    ("off", "None — N/A"),
    ("score", "Scored — music only the audience hears"),
]

# The writer pass. Cooler than the LTX engine's 0.85, and for a reason particular
# to this format: an H3 brief is a specification — timestamps, IDs, tags, a fixed
# field order — and the failure mode of a hot sampler here is not a dull shot but
# a malformed one.
TEMPERATURE, TOP_P = 0.72, 0.92


@dataclass(slots=True)
class H3Request:
    """One H3 brief to write, in the vocabulary this module speaks.

    Every key is this module's own. Nothing here travels to the LTX engine and
    nothing from a ``PromptRequest`` arrives here — the two prompt modes share a
    window, a model and a llama-server, and not a single prompt string.
    """

    intent: str
    mode: str = T2VA
    seconds: float = 10.0
    shots: str = "auto"
    style: str = "auto"
    camera: str = "auto"
    amplitude: str = "medium"
    speed: str = "normal"
    cut: str = "cut"
    language: str = "English"
    speakers: int = 0
    talk: str = "none"
    # Words that must appear on screen, reproduced exactly and in quotes.
    on_screen_text: str = ""
    # Free notes folded into the soundscape, and the switch that replaces it
    # with N/A outright.
    soundscape: str = ""
    ambience: str = "scene"
    music: str = "off"
    music_brief: str = ""
    # "Name = description" lines, the way LTX mode's lexicon reads, so a
    # recurring character keeps one description across shots.
    cast: str = ""
    notes: str = ""
    # Anchor frames, already reduced to a JPEG data URL by imaging.preprocess.
    # L2VA has one picture and it is the last frame, so the two fields are named
    # for where they sit in the video rather than for their picture number.
    first_frame: str | None = None
    last_frame: str | None = None
    seed: int = 7


class VisionUnavailable(RuntimeError):
    """Raised when an anchor frame cannot reach the model that must describe it.

    Same policy as LTX prompt mode: a frame-anchored brief whose frame was never
    seen is not a slightly worse brief, it is a brief about a different video,
    and the instruction line at the top of it would be a false statement. So the
    request stops here rather than quietly becoming a T2VA one.
    """


# ── what each mode needs ─────────────────────────────────────────────────────

def needs_first_frame(mode: str) -> bool:
    """Whether a picture anchors 0.00 seconds. Not L2VA: its one picture is the
    last frame, and the opening is inferred rather than given."""
    return (mode or T2VA).strip().lower() in (I2VA, FL2VA)


def needs_last_frame(mode: str) -> bool:
    return (mode or T2VA).strip().lower() in (FL2VA, L2VA)


def anchored(mode: str) -> bool:
    """Whether this mode puts a picture on the wire at all — which is the
    question the vision projector has to be asked about, and it is not the same
    question as "does it have a first frame": L2VA has only a last one."""
    return needs_first_frame(mode) or needs_last_frame(mode)


def frames(request: H3Request) -> list[str]:
    """The anchor frames in picture order, which is the order H3 numbers them.

    For FL2VA that is first then last; for L2VA the single picture is the last
    frame and is still ``<Picture 1>``.
    """
    found: list[str] = []
    if needs_first_frame(request.mode) and request.first_frame:
        found.append(request.first_frame)
    if needs_last_frame(request.mode) and request.last_frame:
        found.append(request.last_frame)
    return found


def missing_frame(request: H3Request) -> str:
    """Which frame this mode needs and has not been given, or ``""``."""
    if needs_first_frame(request.mode) and not request.first_frame:
        return "first"
    if needs_last_frame(request.mode) and not request.last_frame:
        return "last"
    return ""


# ── budgets ──────────────────────────────────────────────────────────────────

def shot_range(request: H3Request) -> tuple[int, int]:
    """How many shots to write. A pinned count is exact; auto follows the mode.

    FL2VA and L2VA default to one. That is the guide's instruction rather than a
    preference: a single shot is what lets the model interpolate continuously
    towards the frame it has to land on, and it asks for more only when more
    were explicitly specified — which pinning a number here is.

    For the two modes with no frame to land on, auto stays narrow. Five seconds
    is one or two shots however it is cut, and fifteen is not eight: the test for
    a cut is that it brings new information, and a clip this short runs out of
    new information long before it runs out of seconds.
    """
    pinned = (request.shots or "auto").strip().lower()
    if pinned.isdigit():
        count = max(1, min(6, int(pinned)))
        return count, count
    if (request.mode or T2VA).strip().lower() in (FL2VA, L2VA):
        return 1, 1
    seconds = clamp_seconds(request.seconds)
    if seconds < 7:
        return 1, 2
    if seconds < 11:
        return 1, 3
    return 2, 4


def word_budget(seconds: float) -> tuple[int, int]:
    """Words for the description, from the duration it has to cover.

    Calibrated against the guide's own four cases, which run 75 to 95 words for
    six to ten seconds of video — roughly 7 to 16 words a second. An H3 brief is
    a specification, not an essay: every clause has to name something visible or
    audible, and the way to fail that test is to keep writing after the events
    have all been described. The band is wide because the cases are: the same 90
    words describe a busy eight seconds and a still ten.
    """
    seconds = clamp_seconds(seconds)
    return max(60, round(seconds * 7)), max(110, round(seconds * 16))


def spoken_lines(request: H3Request) -> int:
    """How many spoken lines the brief is written against.

    Zero when nobody speaks, and zero when there is nobody to speak: a speaker
    count of none and a talk budget above none disagree, and the count is the
    one that wins, because a line needs a voice to be assigned to.
    """
    if request.speakers <= 0:
        return 0
    rate = TALK_RATE.get((request.talk or "none").strip().lower(), 0.0)
    if rate <= 0:
        return 0
    return max(1, round(clamp_seconds(request.seconds) * rate))


def max_tokens(request: H3Request) -> int:
    """Room for the longest brief the budgets allow, and a margin over it.

    Generous on purpose: the failure this guards against is a brief truncated
    mid-timestamp, which is worse than a brief that stopped early on its own.
    """
    _, high = word_budget(request.seconds)
    return int(high * 3 + 500)


def sampling() -> tuple[float, float]:
    return TEMPERATURE, TOP_P


def clamp_seconds(seconds: float) -> float:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return 10.0
    return max(MIN_SECONDS, min(MAX_SECONDS, value))


# ── the mechanical parts ─────────────────────────────────────────────────────

def timecode(seconds: float) -> str:
    """``MM:SS.mmm``, the form every shot after the first opens with."""
    total = max(0.0, float(seconds))
    minutes, remainder = divmod(total, 60)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def instruction(request: H3Request, final_shot: int = 1) -> str:
    """The line a frame-anchored brief opens with, or ``""`` for T2VA.

    Three wordings, reproduced from the guide exactly as it prints them —
    including the inconsistency between them, where FL2VA writes ``Picture 1
    (from Shot 1)`` bare and the other two bracket both. These are templates the
    model was trained on, not prose to improve.

    ``final_shot`` is the guide's ``N``: the index of the *actual* final shot,
    which is why this is called after the brief comes back rather than before it
    is asked for. The times are the guide's ``S.SS`` — the effective duration to
    exactly two decimal places.
    """
    mode = (request.mode or T2VA).strip().lower()
    seconds = clamp_seconds(request.seconds)
    if mode == I2VA:
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    if mode == FL2VA:
        return ("How the reference pictures align with the target video — Picture 1 "
                "(from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot {final_shot}) aligns with the {seconds:.2f}-second "
                "mark of the target video.")
    if mode == L2VA:
        return ("How the reference pictures align with the target video — <Picture 1> "
                f"(from [Shot {final_shot}]) aligns with the {seconds:.2f}-second mark of "
                "the target video.")
    return ""


def final_shot(description: str) -> int:
    """The number of the last ``[Shot N]`` the writer actually used."""
    marks = shot_marks(description)
    return marks[-1][0] if marks else 1


def assemble(fields: dict[str, str], request: H3Request) -> str:
    """The finished brief: instruction line, blank line, the three fields in order.

    Missing fields are answered rather than skipped. The format is three fields,
    and a brief with two of them is not a shorter brief — it is one H3 will read
    the wrong way round. Where there is nothing to say, the guide's answer is the
    token ``N/A``, so that is what goes in.
    """
    described = (fields.get(DESCRIPTION) or "").strip()
    ambience = (fields.get(SOUNDSCAPE) or "").strip()
    score = (fields.get(MUSIC) or "").strip()
    if not ambience and (request.ambience or "scene").strip().lower() == "silence":
        ambience = NOT_APPLICABLE
    if not score and (request.music or "off").strip().lower() == "off":
        score = NOT_APPLICABLE

    body = "\n\n".join(
        f"{label}: {text}"
        for label, text in ((DESCRIPTION, described), (SOUNDSCAPE, ambience), (MUSIC, score))
        if text
    )
    opener = instruction(request, final_shot(described))
    return f"{opener}\n\n{body}" if opener else body


# ── reading the model back ───────────────────────────────────────────────────

# Reasoning that leaked, a fence that was opened, a sentence of introduction.
# The writer is told to emit the fields and nothing else; this is what happens
# when it does it anyway.
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# A field label, however the writer decorated it: bold, a heading hash, spaces
# where the underscores were, a colon or a dash after it.
_LABEL = re.compile(
    r"^[ \t>*#-]*\**\s*(integrated[ _]multimodal[ _]description|overall[ _]soundscape|"
    # The decoration can sit on either side of the colon — a writer that emitted
    # **label:** put the closing asterisks after it, not before.
    r"non[ _]diegetic[ _]music)\s*\**\s*[:\-—]?\s*\**\s*",
    re.IGNORECASE | re.MULTILINE,
)

# An instruction line the writer produced anyway, in any of its three wordings.
# Stripped rather than kept: the one this module assembles carries the real
# final-shot number, and two of them is worse than none.
_INSTRUCTION = re.compile(
    r"^\s*(For the target video,.*?referenced\.|How the reference pictures align.*?video\.)\s*",
    re.IGNORECASE | re.DOTALL,
)


def parse(raw: str) -> dict[str, str]:
    """Pull the three fields out of whatever the writer actually returned.

    Three shapes are accepted, because all three turn up: the labelled text the
    prompt asks for, a JSON object with the same keys, and — when the writer
    ignored the format entirely — one block of prose, which is taken as the
    description alone so nothing is thrown away.
    """
    text = _THINK.sub("", raw or "").strip()
    text = _FENCE.sub("", text).strip()
    text = _INSTRUCTION.sub("", text).strip()
    if not text:
        return {}

    # JSON first: a model that emitted an object emitted valid keys with it, and
    # guessing at its braces with a regex would only find them again badly.
    if text.startswith("{"):
        try:
            loaded = json.loads(text)
        except ValueError:
            pass
        else:
            if isinstance(loaded, dict):
                found = {key: str(loaded[key]).strip() for key in FIELDS
                         if isinstance(loaded.get(key), str) and loaded[key].strip()}
                if found:
                    return found

    marks = list(_LABEL.finditer(text))
    if not marks:
        return {DESCRIPTION: text}

    fields: dict[str, str] = {}
    for index, mark in enumerate(marks):
        key = mark.group(1).lower().replace(" ", "_")
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        body = text[mark.end():end].strip()
        # Later wins: a writer that restated a field meant the second one, and a
        # writer that emitted it once is unaffected either way.
        if body:
            fields[key] = body
    return fields


# ── checks ───────────────────────────────────────────────────────────────────

_SHOT = re.compile(r"\[\s*Shot\s+(\d+)\s*\]", re.IGNORECASE)
_TIME = re.compile(r"\b(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\b")
_SENTENCE = re.compile(r"[.!?](?:\s|$)")


def shot_marks(description: str) -> list[tuple[int, float | None]]:
    """Every ``[Shot N]`` in order, with the cut time that opens it if any.

    The time has to belong to the shot rather than to the sentence, so only the
    text between this marker and the next is searched, and only the first time
    in it counts — a shot that mentions 00:04.000 in passing three lines later
    did not cut there.
    """
    found: list[tuple[int, float | None]] = []
    marks = list(_SHOT.finditer(description or ""))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(description)
        window = description[mark.end():end]
        stamp = _TIME.search(window)
        at = None
        if stamp:
            at = int(stamp.group(1)) * 60 + float(stamp.group(2))
        found.append((int(mark.group(1)), at))
    return found


def sentence_count(text: str) -> int:
    body = (text or "").strip()
    return len(_SENTENCE.findall(body)) if body else 0


def _is_na(text: str) -> bool:
    return (text or "").strip().rstrip(".").upper() == "N/A"


def checks(brief: str, fields: dict[str, str], request: H3Request) -> list[str]:
    """What looks wrong with the finished brief, in the guide's own terms.

    Reported, never repaired. Every line here names the rule it is about, so the
    fix is a sentence to edit rather than a puzzle to solve, and a writer who
    broke a rule deliberately can read the note and keep the brief.
    """
    notes: list[str] = []
    seconds = clamp_seconds(request.seconds)
    described = fields.get(DESCRIPTION, "")

    if len(brief) > CHARACTER_LIMIT:
        notes.append(f"{len(brief)} characters — over the {CHARACTER_LIMIT} the API accepts.")

    for label in FIELDS:
        if not (fields.get(label) or "").strip():
            notes.append(f"{label} is missing.")

    marks = shot_marks(described)
    if not marks:
        notes.append("No [Shot 1] — the description has to open with one.")
    else:
        numbers = [number for number, _ in marks]
        if numbers[0] != 1:
            notes.append(f"The first shot is [Shot {numbers[0]}] — shots are numbered from 1.")
        if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
            notes.append("Shot numbers are not sequential.")
        if marks[0][1] is not None:
            notes.append("[Shot 1] carries a timestamp — the first shot never does.")
        previous = 0.0
        for number, at in marks[1:]:
            if at is None:
                notes.append(f"[Shot {number}] has no cut time.")
                continue
            if at <= previous:
                notes.append(f"[Shot {number}] cuts at {timecode(at)}, which is not after the shot before it.")
            if at >= seconds:
                notes.append(f"[Shot {number}] cuts at {timecode(at)}, at or past the {seconds:g} s end.")
            previous = max(previous, at)

    notes += _sound_checks(fields, request)
    notes += _speech_checks(described, request)
    return notes


def _sound_checks(fields: dict[str, str], request: H3Request) -> list[str]:
    """The two sound fields: their sentence counts, and where N/A is allowed.

    N/A is a value the guide gives rather than a way out of writing one, so it
    is checked in both directions — present where nothing was asked for, and
    absent where silence was.
    """
    notes: list[str] = []
    ambience = fields.get(SOUNDSCAPE, "")
    silent = (request.ambience or "scene").strip().lower() == "silence"
    if _is_na(ambience):
        if not silent:
            notes.append("overall_soundscape is N/A, which the guide allows only when "
                         "complete silence was asked for.")
    elif ambience:
        if silent:
            notes.append("Complete silence was asked for, so overall_soundscape should be N/A.")
        count = sentence_count(ambience)
        if not 1 <= count <= 4:
            notes.append(f"overall_soundscape runs to {count} sentences — the guide asks for 1 to 4.")

    score = fields.get(MUSIC, "")
    scored = (request.music or "off").strip().lower() != "off"
    if _is_na(score):
        if scored:
            notes.append("A score was asked for, so non_diegetic_music should not be N/A.")
    elif score:
        if not scored:
            notes.append("No score was asked for, so non_diegetic_music should be N/A.")
        count = sentence_count(score)
        if not 1 <= count <= 3:
            notes.append(f"non_diegetic_music runs to {count} sentences — the guide asks for 1 to 3.")
    return notes


def _speech_checks(described: str, request: H3Request) -> list[str]:
    notes: list[str] = []
    if spoken_lines(request):
        if "<d>" not in described:
            notes.append("Speech was asked for but no <d> tag was written.")
        elif described.count("<d>") != described.count("</d>"):
            notes.append("A <d> tag was left unclosed.")
        elif f"[{request.language}]" not in described:
            notes.append(f"No [{request.language}] tag inside the spoken content.")
        if not re.search(r"\(S\d", described):
            notes.append("No speaker ID — every voice needs a stable (S1), (S2), …")
    # A line carried across a cut is marked at both connecting points, so an odd
    # number of markers means one side of the join is unmarked.
    if described.count("<scenetrans>") % 2:
        notes.append("<scenetrans> appears an odd number of times — a line crossing a cut is "
                     "marked at both connecting points.")
    return notes


# ── the prompt ───────────────────────────────────────────────────────────────

def _law_format(request: H3Request) -> str:
    """The output contract. First and non-negotiable, because everything else
    in the brief is worthless if it arrives under the wrong labels."""
    mode = (request.mode or T2VA).strip().lower()
    lines = [
        "OUTPUT FORMAT — EXACTLY THIS, NOTHING ELSE",
        "Write three fields, in this order, each label on its own line followed by its text:",
        "",
        f"{DESCRIPTION}: [Shot 1] …",
        "",
        f"{SOUNDSCAPE}: …",
        "",
        f"{MUSIC}: …",
        "",
        "No preamble, no explanation, no markdown, no code fence, no headings, no bullet "
        "lists, and no fourth field. Write plain prose after each label. Either sound field "
        f"may be exactly {NOT_APPLICABLE} where the rules below allow it, and nothing else "
        "stands in for that.",
    ]
    if mode != T2VA:
        lines.append(
            "Do not write a reference or alignment line of your own — one is added above your "
            "first field after you are done, and it names the real final shot number. Start "
            "at the first label."
        )
    return "\n".join(lines)


def _law_description(request: H3Request) -> str:
    """How the timeline is written: shots, times, cuts, camera, style."""
    seconds = clamp_seconds(request.seconds)
    low, high = shot_range(request)
    words_low, words_high = word_budget(seconds)
    count = (f"exactly {low} shot" + ("" if low == 1 else "s")) if low == high \
        else f"{low} to {high} shots"
    cut = (request.cut or "cut").strip().lower()
    mode = (request.mode or T2VA).strip().lower()

    lines = [
        f"{DESCRIPTION} — THE TIMELINE",
        f"This is a {seconds:g}-second video at {FPS} FPS. It is the main body of the prompt. "
        f"Write it along its timeline, in {words_low}-{words_high} words, as {count}.",
        "",
        "Every detail must correspond to something visible or audible: visual style, initial "
        "composition, subject appearance and position, the scene and its key props, actions "
        "and reactions, shot changes, spoken language, and synchronised diegetic sound. "
        "Nothing about what the video means, is about, or makes anyone feel.",
        "",
        "At the beginning of [Shot 1], state the overall style and the initial composition. "
        "[Shot 1] carries NO timestamp:",
        "    [Shot 1] Live-action, cinematic, a medium-wide shot frames …",
    ]
    if low != 1 or high != 1:
        alternates = ", ".join(f'"{phrase}"' for phrase in CUT_ALTERNATES)
        lines += [
            "",
            "Every later shot uses the next number and opens with a strictly increasing cut "
            "time inside the video duration:",
            "    [Shot 2] At 00:03.500, the camera cuts to …",
            "The time is MM:SS.mmm. Never repeat one and never go backwards.",
        ]
        if cut == "cut":
            lines.append(f"For ordinary cuts use any of: {alternates}. Vary them.")
        else:
            lines.append(
                f'Use "{CUT_PHRASES[cut]}" for the transitions. The guide reserves this for '
                "when the user asks for it, and the user has asked for it."
            )
        lines += [
            "",
            "A cut must introduce new information about the subject, the space, the state, "
            "the viewpoint or the time. If only the distance or a slight angle needs to "
            "change, prefer camera motion.",
        ]
    elif mode in (FL2VA, L2VA):
        lines += [
            "",
            "One shot, so that the motion between the anchor frames is continuous. There are "
            "no cuts and no timestamps anywhere in this brief.",
        ]

    table = "\n".join(f"    {name} — {meaning}" for name, meaning in CAMERA_MOVES)
    lines += [
        "",
        "CAMERA MOTION — MOTION TYPE + AMPLITUDE + SPEED",
        table,
        "Amplitude: \"with small amplitude\" or \"with large amplitude\". Speed: \"at slow "
        "speed\" or \"at fast speed\". Add either only when it means something — medium "
        "amplitude and normal speed are usually omitted.",
        "Write the motion as a natural English action inside the shot, never stacked as "
        "labels at the end of a sentence:",
        "    The camera pushes in with small amplitude at slow speed toward the folded "
        "letter in her hands.",
        "    The camera pans right with large amplitude at fast speed, revealing the open "
        "doorway.",
        "    The camera holds a static shot as the runner exits the frame.",
        "",
        "SOUND ON THE TIMELINE. Dialogue, singing and every sound whose source is in the "
        "scene belong here, at the moment they happen, tied to the action that makes them — "
        "including a radio, a television, a phone or an instrument playing in the shot. "
        "Ambience and score do not belong here; they have fields of their own below.",
    ]
    if request.on_screen_text.strip():
        lines += [
            "",
            "ON-SCREEN TEXT. Any banner, sign, label, subtitle or neon text actually visible "
            "on screen goes in English double quotation marks, preserved verbatim and "
            "untranslated, with where and when it appears:",
            "    A red neon sign reading \"营业中\" glows above the doorway.",
            "The text: " + request.on_screen_text.strip(),
        ]
    return "\n".join(lines)


def _law_speech(request: H3Request) -> str:
    """The speech syntax, only when there is speech."""
    lines = spoken_lines(request)
    if not lines:
        return ("SPEECH\nNobody speaks. Write no <d> tags and no speaker IDs. Voices may still "
                "be heard as non-verbal sound — a laugh, a gasp, a sigh — and those belong in "
                f"{SOUNDSCAPE}.")
    language = (request.language or "English").strip() or "English"
    people = request.speakers
    _, most_shots = shot_range(request)
    # The rule for a line that crosses a cut is only worth a paragraph in a
    # brief that can have one. A single-shot brief has no cut to cross.
    crossing = ([
        "",
        "When one line of dialogue or lyric crosses a cut, write <scenetrans> at the "
        "connecting point in BOTH parts and say the audio continues across it — \"continues "
        "seamlessly across the cut\", \"continues uninterrupted into the next shot\", "
        "\"carries over from the previous shot\", \"remains audible across the transition\".",
    ] if most_shots > 1 else [])
    return "\n".join([
        "SPEECH — A SYNTAX, NOT A STYLE",
        f"{people} speaking character{'' if people == 1 else 's'}, about {lines} spoken "
        f"line{'' if lines == 1 else 's'} across the clip. Keep the lines short enough to be "
        "said in the seconds they are given.",
        "",
        "Subjects who speak, sing, or produce an off-screen human voice use stable IDs: (S1), "
        "(S2), … kept the same across shots. Characters who never vocalise receive no ID. "
        "When several already-numbered speakers speak or sing together, use a compound ID: "
        "(S1,S2).",
        "",
        "When a speaker first appears, establish a stable identity from the visual and audio "
        "context — character type, age, gender, whether they are on screen, pitch, timbre, "
        "speaking rate, accent. The identifying phrase, the ID, the action and the delivery "
        "go OUTSIDE <d>. Inside <d>, only the language tag and the actual spoken content:",
        f"    The young woman with a quiet, breathy voice (S1) says: <d>[{language}] I get "
        "off at the next station.</d>",
        f"    The two children (S1,S2) shout together, <d>[{language}] Wait for us!</d>",
        "Preserve every original word and punctuation mark verbatim; never translate or "
        "rewrite them.",
        "",
        "For a voiceover use the exact phrase \"says in an off-screen voiceover\", and "
        "immediately after the <d> block state that the on-screen character's lips remain "
        "closed:",
        f"    The man (S1) says in an off-screen voiceover: <d>[{language}] I still remember "
        "that road.</d> while his lips remain completely closed.",
    ] + crossing + [
        "",
        "Use <cutoff> when speech is truncated by the end of the video.",
    ])


def _law_sound(request: H3Request) -> str:
    """The two sound fields, and the rule that keeps them apart."""
    music = (request.music or "off").strip().lower()
    silent = (request.ambience or "scene").strip().lower() == "silence"
    lines = [f"{SOUNDSCAPE} — 1 TO 4 SENTENCES, ONE PARAGRAPH"]
    if silent:
        lines.append("Complete silence was asked for throughout the video, which is the one "
                     f"case that takes the token. Write exactly: {NOT_APPLICABLE}")
    else:
        lines += [
            "One continuous paragraph summarising the ambient sound, the sound of physical "
            "action, and non-verbal human sound across the full video: wind, rain, traffic, "
            "footsteps, fabric movement, impacts, breathing, laughter, panting. Tie a sound "
            "to the thing that makes it, so it lands on the frame.",
            "Do not repeat dialogue, singing or diegetic music here — they already belong to "
            "the multimodal description, and a sound described twice is asked for twice.",
            "    Steady rain taps against the café windows while low room ambience continues "
            "underneath. The entrance bell rings once, followed by wet footsteps and the soft "
            "scrape of a chair.",
        ]
        if request.soundscape.strip():
            lines.append("The sound to build it from: " + request.soundscape.strip())

    lines += ["", f"{MUSIC} — 1 TO 3 SENTENCES"]
    if music == "off":
        lines.append(f"There is no non-diegetic music. Write exactly: {NOT_APPLICABLE}")
    else:
        lines += [
            "Background music the characters cannot hear and only the audience can. Focus on "
            "instrumentation, speed, rhythm and dynamic changes. Do not use abstract mood "
            "words, do not explain the emotional function of the score, and never name a "
            "song, an artist or a band. Singing, instruments, radio, television or phone "
            "music audible to the characters are diegetic events and belong in the "
            "description instead.",
            "    Sparse piano notes at a slow tempo, joined by sustained low strings that "
            "gradually increase in volume before fading out.",
        ]
        if request.music_brief.strip():
            lines.append("The score is: " + request.music_brief.strip())
    return "\n".join(lines)


def _law_frames(request: H3Request) -> str:
    """What the anchor frames mean, for the three modes that have them."""
    mode = (request.mode or T2VA).strip().lower()
    seconds = clamp_seconds(request.seconds)
    if mode == I2VA:
        return "\n".join([
            "THE FIRST FRAME — BEGIN FROM THE IMAGE AND DEVELOP FORWARD",
            "The attached picture IS the video at 0.00 seconds and belongs to [Shot 1]. First "
            "establish the style, the subjects, the composition and the scene anchors that "
            "are actually in it, then describe the next action. Character identity, clothing, "
            "colours, key objects and spatial relationships stay consistent with it.",
            "Structure: first-frame anchor → action onset → continuous development → result "
            "or reaction.",
        ])
    if mode == FL2VA:
        return "\n".join([
            "THE FIRST AND LAST FRAMES — DESCRIBE THE PATH BETWEEN THEM",
            "The first picture is the opening and the second is the ending. Do not describe "
            "the two stills one after the other; supply the motion path that connects them — "
            "how the subject moves, how poses change, how objects are handled, how the "
            "composition evolves, how the scene or the lighting transitions. The last frame "
            f"must be reached by the final shot at the {seconds:.2f}-second end of the video.",
            "Structure: first-frame state → observable intermediate changes → progressively "
            "narrowing differences → last-frame state.",
        ])
    if mode == L2VA:
        return "\n".join([
            "THE LAST FRAME — INFER THE OPENING AND LAND ON THE IMAGE",
            "The attached picture IS the final frame of the video and belongs to the last "
            "shot. It does not belong to [Shot 1]. Infer a plausible earlier state from the "
            "brief and from that frame, then describe how the characters, the objects, the "
            "camera and the scene gradually approach it, landing on its exact composition, "
            f"positions and lighting at the {seconds:.2f}-second end.",
            "Structure: plausible preceding state → explicit action and transition path → "
            "gradual convergence in the final shot → last-frame landing.",
        ])
    return ""


def _law_cast(request: H3Request) -> str:
    if not request.cast.strip():
        return ""
    return "\n".join([
        "THE CAST",
        "Written as Name = description. Where a name below appears in the brief, describe "
        "that character with this description, the same way, every time they are on screen — "
        "H3 has no memory between shots except the words that describe someone.",
        request.cast.strip(),
    ])


def build_system(request: H3Request) -> str:
    """The whole instruction, assembled from the blocks that apply.

    Ordered the way the guide is read rather than the way the fields print: what
    to emit, then the timeline, then the frames that anchor it, then the voices,
    then the sound. A block is absent when its setting is off — a brief with
    nobody speaking should not carry a page about speaker IDs.
    """
    mode = (request.mode or T2VA).strip().lower()
    style = STYLE_PHRASES.get((request.style or "auto").strip().lower(), "")
    camera = dict(CAMERAS).get((request.camera or "auto").strip().lower(), "")
    amplitude = (request.amplitude or "medium").strip().lower()
    speed = (request.speed or "normal").strip().lower()

    blocks = [
        "You write prompts for MiniMax-H3, a video model that generates picture and sound "
        "together. You are writing the finished prompt itself, in H3's own format — not "
        "advice about one, and not a description of what you would write.",
        _law_format(request),
        _law_description(request),
        _law_frames(request),
        _law_speech(request),
        _law_sound(request),
        _law_cast(request),
    ]

    asked: list[str] = []
    if style:
        asked.append(f"Style: {style}. State it at the start of [Shot 1].")
    elif mode != T2VA:
        # The guide splits this: keyframe tasks take the style from the picture,
        # T2VA takes it from the user's text.
        asked.append("Style: take it from the attached picture and name it at the start of "
                     "[Shot 1].")
    if camera and camera != "From the intent":
        move = camera
        if amplitude in AMPLITUDE_PHRASES:
            move += f" {AMPLITUDE_PHRASES[amplitude]}"
        if speed in SPEED_PHRASES:
            move += f" {SPEED_PHRASES[speed]}"
        asked.append(f"Camera: {move}. Write it as an action, inside the sentence.")
    if request.notes.strip():
        asked.append("Also: " + request.notes.strip())
    if asked:
        blocks.append("WHAT WAS ASKED FOR\n" + "\n".join(asked))

    blocks.append("Write the fields now.")
    return "\n\n".join(block for block in blocks if block)


def build_user(request: H3Request) -> str:
    """The turn carrying the intent, and the frames when there are frames."""
    intent = (request.intent or "").strip()
    mode = (request.mode or T2VA).strip().lower()
    lines = []
    if mode == I2VA:
        lines.append("The attached picture is the first frame of the video.")
    elif mode == FL2VA:
        lines.append("The two attached pictures are the first and the last frame of the "
                     "video, in that order.")
    elif mode == L2VA:
        lines.append("The attached picture is the last frame of the video.")
    lines.append(f"Video to write: {intent}")
    return "\n\n".join(lines)


def messages(request: H3Request, *, vision_available: bool = True) -> list[dict]:
    """The chat turns, with the anchor frames attached in picture order.

    Raises ``VisionUnavailable`` rather than dropping a frame the model cannot
    be shown, for the reason on that class.
    """
    wanted = missing_frame(request)
    if wanted:
        raise VisionUnavailable(f"This mode needs a {wanted} frame, and none is attached.")
    attached = frames(request)
    if attached and not vision_available:
        raise VisionUnavailable(
            "The model running has no vision projector, so the anchor frames cannot be sent "
            "to it. Choose a model with one under Settings, or switch to text to video."
        )

    user = build_user(request)
    if attached:
        content: list[dict] = [{"type": "image_url", "image_url": {"url": url}}
                               for url in attached]
        content.append({"type": "text", "text": user})
        turn: dict = {"role": "user", "content": content}
    else:
        turn = {"role": "user", "content": user}
    return [{"role": "system", "content": build_system(request)}, turn]
