"""MiniMax-H3 prompts — a second engine, written to MiniMax's own guide.

This module has nothing to do with the LTX engine and deliberately touches none
of it. ``prompt_engine.upstream`` is a byte-for-byte port whose every string is
pinned by ``tools/check_upstream_sync.py``; H3 wants a different prompt, in a
different shape, obeying different laws, and the only honest way to have both is
two engines that never share a line. Nothing here imports ``upstream``, and
nothing in ``upstream`` knows this file exists.

**Where the rules come from.** MiniMax publishes
``docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`` with the H3 weights, and *base
mode* — no reference material, just a text brief and up to two anchor frames —
is what this module writes. The guide is specific in a way most prompt advice is
not, and the specifics are the reason this is a mode of its own rather than a style
preset on the existing one:

* An H3 prompt is **three named fields**, always in the same order:
  ``integrated_multimodal_description``, ``overall_soundscape``,
  ``non_diegetic_music``. They are not headings for a human — they are the
  format the model was trained to read.
* The description is a **timeline**, not a paragraph. ``[Shot 1]`` opens it and
  carries no timestamp; every later shot opens with ``[Shot N] At MM:SS.mmm,``
  and a cut phrase, at a strictly increasing time inside the duration.
* Sound is **directed, not mentioned**. Dialogue, singing and diegetic audio
  belong on the timeline with the picture; ambience belongs in the soundscape;
  score belongs in the music field, described by instrumentation rather than by
  mood. Saying the same sound twice is a fault, not emphasis.
* Speech has a **syntax**: a stable ``(S1)`` per voice, the speaker and delivery
  written outside the tag, and only a language tag and the words themselves
  inside ``<d>…</d>``, preserved to the punctuation mark.
* With an anchor frame, one **alignment line comes first**, then a blank line,
  then the fields — and the times in it are written to exactly two decimals.

**What is written here and what is written by the model.** Everything mechanical
is assembled in this file: the alignment line, the field labels, their order,
the blank line after the alignment. Those are the parts a language model gets
subtly wrong on a bad day, and they are also the parts with one correct answer,
so there is no reason to ask. The model is asked only for prose, and is asked
for it one field at a time under the laws above.

**Checks, not enforcement.** ``checks()`` re-reads the finished brief and says
what looks wrong — a first shot that grew a timestamp, cut times that go
backwards or run past the end, a soundscape that ran to eight sentences, a brief
over the 7,000-character limit the API takes. It never rewrites: a brief that
breaks a rule on purpose is still the writer's to keep, and a silent correction
is how you end up with a prompt nobody chose.
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

# What the product surface accepts: 5-15 seconds at 24 FPS, and a prompt of at
# most 7,000 characters. The duration bounds are the spin box's range; the
# character limit is a check rather than a truncation, because a brief three
# characters over is a sentence to cut by hand, not by machine.
MIN_SECONDS, MAX_SECONDS = 5.0, 15.0
FPS = 24
CHARACTER_LIMIT = 7000

# ── modes ────────────────────────────────────────────────────────────────────
# The guide's three base-mode shapes. Each is the T2VA body plus, for the two
# frame-anchored ones, an instruction line and a rule about where the picture
# goes in time.

T2VA, I2VA, FL2VA = "t2va", "i2va", "fl2va"

MODES: list[tuple[str, str]] = [
    (T2VA, "Text to video — no frames (T2VA)"),
    (I2VA, "First frame (I2VA)"),
    (FL2VA, "First and last frame (FL2VA)"),
]

# ── look ─────────────────────────────────────────────────────────────────────
# The guide names these as the common styles and asks for one of them at the
# start of [Shot 1], alongside the opening composition.

STYLES: list[tuple[str, str]] = [
    ("auto", "From the intent"),
    ("cinematic", "Cinematic live-action"),
    ("animation_2d", "2D animation"),
    ("cg_3d", "3D CG"),
    ("claymation", "Claymation"),
    ("watercolor", "Watercolour"),
    ("vintage_film", "Vintage film"),
]

STYLE_PHRASES: dict[str, str] = {
    "cinematic": "cinematic live-action",
    "animation_2d": "2D-animated",
    "cg_3d": "3D CG",
    "claymation": "claymation",
    "watercolor": "watercolour",
    "vintage_film": "vintage film",
}

# H3's camera vocabulary, verbatim. The guide asks for these as English actions
# inside the shot — "the camera pushes in slowly toward her hands" — never
# stacked as labels after the sentence, which is why the UI offers the move and
# the prompt asks for it to be written rather than pasted.
CAMERA_MOVES: tuple[str, ...] = (
    "Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
    "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
    "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
    "Roll Counterclockwise",
)

CAMERAS: list[tuple[str, str]] = (
    [("auto", "From the intent")]
    + [(move.lower().replace(" ", "_"), move) for move in CAMERA_MOVES]
)

# Amplitude and speed are written only when they are not the default, because
# the guide says so: medium amplitude and normal speed are the unmarked case,
# and writing them spends words saying nothing.
AMPLITUDES: list[tuple[str, str]] = [
    ("medium", "Medium — left unwritten"), ("small", "Small"), ("large", "Large"),
]
SPEEDS: list[tuple[str, str]] = [
    ("normal", "Normal — left unwritten"), ("slow", "Slow"), ("fast", "Fast"),
]

# Straight cuts are the default and the guide gives five interchangeable
# phrasings for them. The other three are named as things to use only when they
# were actually asked for — choosing one here is that request.
CUTS: list[tuple[str, str]] = [
    ("cut", "Straight cut"),
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

# The guide's own alternates for a straight cut, offered to the writer so a
# four-shot brief does not open every shot with the same five words.
CUT_ALTERNATES = (
    "the camera cuts to", "the shot cuts to", "the shot transitions to",
    "the shot changes to", "the shot switches to",
)

# A 5-15 second clip does not hold many shots, and the guide is explicit that a
# cut has to earn itself with new information — a closer look at the same thing
# is a camera move, not a cut. "Auto" leaves the count to the beat.
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

# How much of the clip is spoken over. The guide asks for lines proportional to
# the duration rather than as many as will fit, so this is a budget, and
# ``spoken_lines`` turns it into a number the brief is written against.
TALK: list[tuple[str, str]] = [
    ("none", "None — nobody speaks"),
    ("sparse", "Sparse — a line or two"),
    ("steady", "Steady — a real exchange"),
    ("dense", "Dense — talking throughout"),
]

# Lines per second, by density. A line is a sentence or two of speech, which at
# an ordinary delivery is about two seconds of screen time — so "dense" is one
# line every two and a bit seconds, and there is no setting that fills a ten
# second clip with fifteen of them.
TALK_RATE: dict[str, float] = {"none": 0.0, "sparse": 0.14, "steady": 0.28, "dense": 0.45}

MAX_SPEAKERS = 6

# ── music ────────────────────────────────────────────────────────────────────

MUSIC_MODES: list[tuple[str, str]] = [
    ("off", "None — the scene's own sound only"),
    ("score", "Scored — music only the audience hears"),
]

# What ``non_diegetic_music`` says when there is no score. The field is part of
# the format, so it is answered rather than dropped, and answered in the terms
# the guide uses for that field.
NO_MUSIC = "There is no non-diegetic music; the audience hears only the scene's own sound."

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
    # Free notes folded into the two sound fields.
    soundscape: str = ""
    music: str = "off"
    music_brief: str = ""
    # "Name = description" lines, the way LTX mode's lexicon reads, so a
    # recurring character keeps one description across shots.
    cast: str = ""
    notes: str = ""
    # Anchor frames, already reduced to a JPEG data URL by imaging.preprocess.
    first_frame: str | None = None
    last_frame: str | None = None
    seed: int = 7


class VisionUnavailable(RuntimeError):
    """Raised when an anchor frame cannot reach the model that must describe it.

    Same policy as LTX prompt mode: a frame-anchored brief whose frame was never
    seen is not a slightly worse brief, it is a brief about a different video,
    and the alignment line at the top of it would be a false statement. So the
    request stops here rather than quietly becoming a T2VA one.
    """


# ── budgets ──────────────────────────────────────────────────────────────────

def shot_range(request: H3Request) -> tuple[int, int]:
    """How many shots to write. A pinned count is exact; auto scales with time.

    Auto is deliberately narrow. Five seconds is one or two shots however it is
    cut, and fifteen seconds is not eight — the guide's test for a cut is that
    it brings new information, and a clip this short runs out of new information
    long before it runs out of seconds.
    """
    pinned = (request.shots or "auto").strip().lower()
    if pinned.isdigit():
        count = max(1, min(6, int(pinned)))
        return count, count
    seconds = clamp_seconds(request.seconds)
    if seconds < 7:
        return 1, 2
    if seconds < 11:
        return 1, 3
    return 2, 4


def word_budget(seconds: float) -> tuple[int, int]:
    """Words for the description, from the duration it has to cover.

    A shot brief is dense — composition, appearance, action, camera, sound, and
    the speech itself — but H3 reads a timeline, and a timeline padded past the
    events it describes stops being one. Roughly 24 to 38 words per second lands
    a ten-second brief at 240-380 words, which is a full page and no more.
    """
    seconds = clamp_seconds(seconds)
    return max(90, round(seconds * 24)), max(150, round(seconds * 38))


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
    # Two tokens a word for the description, plus the two sound fields, plus the
    # labels and the punctuation that a specification is full of.
    return int(high * 2 + 500)


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


def needs_first_frame(mode: str) -> bool:
    return (mode or T2VA).strip().lower() in (I2VA, FL2VA)


def needs_last_frame(mode: str) -> bool:
    return (mode or T2VA).strip().lower() == FL2VA


def alignment(request: H3Request) -> str:
    """The instruction line a frame-anchored brief opens with, or ``""``.

    The guide is exact about three things here and all three are mechanical, so
    none of them is left to the writer: it is the first line, exactly one blank
    line separates it from the fields, and every time in it carries two decimal
    places — ``0.00``, not ``0`` and not ``0.000``.
    """
    mode = (request.mode or T2VA).strip().lower()
    if mode == I2VA:
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    if mode == FL2VA:
        low, high = shot_range(request)
        # The last picture belongs to the last shot, which only has a number
        # when the count was pinned. When it was not, it is named rather than
        # numbered — a wrong number is worse than no number.
        last = f"[Shot {high}]" if low == high else "the final shot"
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced, and at "
                f"{clamp_seconds(request.seconds):.2f} seconds into the target video, "
                f"<Picture 2> (from {last}) is fully referenced.")
    return ""


def assemble(fields: dict[str, str], request: H3Request) -> str:
    """The finished brief: alignment line, blank line, the three fields in order.

    Missing fields are answered rather than skipped. The format is three fields,
    and a brief with two of them is not a shorter brief — it is one H3 will read
    the wrong way round.
    """
    described = (fields.get(DESCRIPTION) or "").strip()
    ambience = (fields.get(SOUNDSCAPE) or "").strip()
    score = (fields.get(MUSIC) or "").strip()
    if not score:
        score = NO_MUSIC if (request.music or "off").strip().lower() == "off" else ""

    body = "\n\n".join(
        f"{label}: {text}"
        for label, text in ((DESCRIPTION, described), (SOUNDSCAPE, ambience), (MUSIC, score))
        if text
    )
    opener = alignment(request)
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


def parse(raw: str) -> dict[str, str]:
    """Pull the three fields out of whatever the writer actually returned.

    Three shapes are accepted, because all three turn up: the labelled text the
    prompt asks for, a JSON object with the same keys, and — when the writer
    ignored the format entirely — one block of prose, which is taken as the
    description alone so nothing is thrown away.
    """
    text = _THINK.sub("", raw or "").strip()
    text = _FENCE.sub("", text).strip()
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

    ambience = sentence_count(fields.get(SOUNDSCAPE, ""))
    if ambience and not 1 <= ambience <= 4:
        notes.append(f"overall_soundscape runs to {ambience} sentences — the guide asks for 1 to 4.")

    score = sentence_count(fields.get(MUSIC, ""))
    if score and not 1 <= score <= 3:
        notes.append(f"non_diegetic_music runs to {score} sentences — the guide asks for 1 to 3.")

    if spoken_lines(request):
        if "<d>" not in described:
            notes.append("Speech was asked for but no <d> tag was written.")
        elif described.count("<d>") != described.count("</d>"):
            notes.append("A <d> tag was left unclosed.")
        elif f"[{request.language}]" not in described:
            notes.append(f"No [{request.language}] tag inside the spoken content.")
        if not re.search(r"\(S\d", described):
            notes.append("No speaker ID — every voice needs a stable (S1), (S2), …")
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
        f"{DESCRIPTION}: <text>",
        "",
        f"{SOUNDSCAPE}: <text>",
        "",
        f"{MUSIC}: <text>",
        "",
        "No preamble, no explanation, no markdown, no code fence, no headings, no bullet "
        "lists, and no fourth field. Write plain prose after each label.",
    ]
    if mode != T2VA:
        lines.append(
            "Do not write an alignment or reference line of your own — one is added above "
            "your first field after you are done. Start at the first label."
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

    lines = [
        f"{DESCRIPTION} — THE TIMELINE",
        f"This is a {seconds:g}-second video at {FPS} FPS. Describe it along its timeline, "
        f"in {words_low}-{words_high} words, as {count}.",
        "",
        "[Shot 1] opens the field and carries NO timestamp. At its start, state the overall "
        "style and the opening composition, then the subject's appearance and position, the "
        "environment and its light, what moves, and how the camera moves.",
    ]
    if low != 1 or high != 1:
        alternates = ", ".join(f'"{phrase}"' for phrase in CUT_ALTERNATES)
        lines += [
            "",
            "Every later shot opens exactly like this:",
            "    [Shot 2] At 00:03.000, the camera cuts to …",
            "The time is MM:SS.mmm, is strictly later than the shot before it, and falls "
            f"inside the {seconds:g} seconds. Never repeat a time and never go backwards.",
        ]
        if cut == "cut":
            lines.append(f"For the cut phrase use any of: {alternates}. Vary them.")
        else:
            lines.append(
                f'Use "{CUT_PHRASES[cut]}" for the transitions, as asked for — this is one '
                "of the deliberate transitions, so it is written rather than a straight cut."
            )
        lines += [
            "",
            "A cut must bring new information — a new subject, a new space, a new state, a "
            "new viewpoint, a new moment in time. If all that changes is the distance or a "
            "small angle on the same thing, do not cut: move the camera instead.",
        ]
    lines += [
        "",
        "CAMERA. Name the move in H3's own vocabulary — " + ", ".join(CAMERA_MOVES) + " — but "
        "write it as an English action inside the sentence, never as a label stuck on the end. "
        'Write "the camera pushes in slowly toward the folded letter in her hands", not '
        '"…her hands. (Push In, slow)". Give amplitude and speed only when they are not the '
        "ordinary case: medium amplitude and normal speed are left unwritten.",
        "",
        "SOUND ON THE TIMELINE. Dialogue, singing and any sound whose source is in the scene "
        "belong here, placed at the moment they happen, tied to the action that makes them. "
        "Ambience and score do not belong here — they have fields of their own below.",
    ]
    if request.on_screen_text.strip():
        lines += [
            "",
            "ON-SCREEN TEXT. Reproduce these words exactly, inside English double quotation "
            "marks, and say where and when they appear: " + request.on_screen_text.strip(),
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
    return "\n".join([
        "SPEECH — A SYNTAX, NOT A STYLE",
        f"{people} speaking character{'' if people == 1 else 's'}, about {lines} spoken "
        f"line{'' if lines == 1 else 's'} across the clip. Keep the lines short enough to be "
        "said in the seconds they are given.",
        "",
        "Give every voice a stable ID: (S1), (S2), … kept the same in every shot. A character "
        "who never makes a sound gets no ID. Two voices at once share one: (S1,S2).",
        "",
        "Who is speaking, what they look like and how they say it go OUTSIDE the tag. Inside "
        "the tag goes the language tag and the words, and nothing else:",
        f'    The young woman with a quiet, breathy voice (S1) says: <d>[{language}] I get off '
        "at the next station.</d>",
        "",
        "The first time a voice speaks, establish it outside the tag — apparent age, gender, "
        "pitch, timbre, speaking rate, accent. Off screen, write that they say it in an "
        "off-screen voiceover, and say what their mouth is doing. Reproduce any words the "
        "brief already quotes exactly, to the punctuation mark: never translate, never "
        f"paraphrase, never tidy them. The language tag is [{language}].",
    ])


def _law_sound(request: H3Request) -> str:
    """The two sound fields, and the rule that keeps them apart."""
    music = (request.music or "off").strip().lower()
    lines = [
        f"{SOUNDSCAPE} — 1 TO 4 SENTENCES, ONE PARAGRAPH",
        "Summarise the ambient sound, the sound of physical action, and non-verbal human "
        "sound across the whole video: wind, rain, traffic, footsteps, fabric, impacts, "
        "breathing, laughter, panting. Tie a sound to the thing that makes it, so it lands "
        'on the frame — "a pop as the cork clears the bottle", not "a pop".',
        "Do not repeat dialogue, singing or in-scene music here. They are already on the "
        "timeline, and a sound described twice is asked for twice.",
        "",
        f"{MUSIC} — 1 TO 3 SENTENCES",
    ]
    if music == "off":
        lines.append(f"There is no score. Write exactly this and nothing more: {NO_MUSIC}")
    else:
        lines += [
            "Music only the audience hears — the characters cannot. Describe instrumentation, "
            "tempo, rhythm and how the dynamics change across the clip. No mood words, no "
            "explaining what the music makes the viewer feel, and never name a song, an "
            "artist or a band.",
        ]
        if request.music_brief.strip():
            lines.append("The score is: " + request.music_brief.strip())
    if request.soundscape.strip():
        lines += ["", "The sound to build the ambience from: " + request.soundscape.strip()]
    return "\n".join(lines)


def _law_frames(request: H3Request) -> str:
    """What the anchor frames mean, for the two modes that have them."""
    mode = (request.mode or T2VA).strip().lower()
    seconds = clamp_seconds(request.seconds)
    if mode == I2VA:
        return "\n".join([
            "THE FIRST FRAME",
            "The attached picture IS the video at 0.00 seconds, and it belongs to [Shot 1]. "
            "Open by describing what is actually in it — the subject, their appearance and "
            "position, the space, the light — and then develop forward from it. Everything "
            "after 0.00 seconds is a path away from that frame: nothing in [Shot 1]'s opening "
            "composition may contradict it.",
        ])
    if mode == FL2VA:
        return "\n".join([
            "THE FIRST AND LAST FRAMES",
            "The first picture IS the video at 0.00 seconds and belongs to [Shot 1]. The "
            f"second IS the video at {seconds:.2f} seconds and belongs to the final shot. "
            "Describe both from what is actually in them, and write one continuous path "
            "between them: the first-frame state, then the changes that can be seen "
            "happening, then those differences narrowing, then the last-frame state. The "
            "video must be able to arrive at the second picture without a jump.",
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
    style = STYLE_PHRASES.get((request.style or "auto").strip().lower(), "")
    camera = dict(CAMERAS).get((request.camera or "auto").strip().lower(), "")
    amplitude = (request.amplitude or "medium").strip().lower()
    speed = (request.speed or "normal").strip().lower()

    blocks = [
        "You write prompts for MiniMax-H3, a video model that generates picture and sound "
        "together. You are writing the finished prompt itself, in H3's base-mode format — not "
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
        asked.append(f"Style: {style}. Say so at the start of [Shot 1].")
    if camera and camera != "From the intent":
        move = camera
        if amplitude != "medium":
            move += f", {amplitude} amplitude"
        if speed != "normal":
            move += f", {speed}"
        asked.append(f"Camera: {move}. Write it as an action, in the sentence.")
    if request.notes.strip():
        asked.append("Also: " + request.notes.strip())
    if asked:
        blocks.append("WHAT WAS ASKED FOR\n" + "\n".join(asked))

    blocks.append(
        "Write the fields now. Every sentence describes something that can be seen or heard; "
        "nothing describes what the video is about, how it feels, or what it means."
    )
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
    lines.append(f"Video to write: {intent}")
    return "\n\n".join(lines)


def messages(request: H3Request, *, vision_available: bool = True) -> list[dict]:
    """The chat turns, with the anchor frames attached in first-then-last order.

    Raises ``VisionUnavailable`` rather than dropping a frame the model cannot
    be shown, for the reason on that class.
    """
    frames = []
    if needs_first_frame(request.mode):
        if not request.first_frame:
            raise VisionUnavailable("This mode needs a first frame, and none is attached.")
        frames.append(request.first_frame)
    if needs_last_frame(request.mode):
        if not request.last_frame:
            raise VisionUnavailable("This mode needs a last frame, and none is attached.")
        frames.append(request.last_frame)
    if frames and not vision_available:
        raise VisionUnavailable(
            "The model running has no vision projector, so the anchor frames cannot be sent "
            "to it. Choose a model with one under Settings, or switch to text to video."
        )

    user = build_user(request)
    if frames:
        content: list[dict] = [{"type": "image_url", "image_url": {"url": url}} for url in frames]
        content.append({"type": "text", "text": user})
        turn: dict = {"role": "user", "content": content}
    else:
        turn = {"role": "user", "content": user}
    return [{"role": "system", "content": build_system(request)}, turn]
