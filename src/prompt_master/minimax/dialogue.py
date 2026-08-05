"""Dialogue intensity — how much of an H3 video is somebody talking.

This is the one file in ``minimax/`` whose English is this application's own.
``prompt_enhancer.py`` is vendored verbatim and pinned by digest, and
``enhancer.py`` is WanGP's calling convention around it and writes no prompt
text of its own; both stay that way. Everything a dialogue slider needs to say
is said here instead, and nothing here is claimed to be WanGP's.

It is a slider rather than a switch because the request is a quantity: at 1 a
scene has a line or two in it, at 10 a voice is talking essentially without
stopping while the scene described goes on underneath. Off is the left-hand
position and the default, and at Off this module builds byte-for-byte the
request ``enhancer.messages`` builds — no pass runs, no instruction is added,
no token budget moves. A control nobody touched changes nothing.

Above Off it works in two halves, and it needs both:

*A pass writes the lines.* Before the enhancer is asked for anything, a
separate call reads the prompt — and the image caption, when there is one, so
the people in the picture can speak — casts whoever is in it, and writes their
speech with the emotional delivery of each line beside it. A prompt that
already quotes dialogue has its own voice, and the new lines continue it. A
prompt with nobody in it who could speak gets an unseen narrator, whose
register is taken from what the scene actually is: hushed over a landscape,
clipped over a chase. The lines then travel into the request as part of what
the user asked for, which is where H3 expects content to come from.

*A directive says how the lines land.* The vendored instructions end with "Do
not add dialogue, narration, music, cuts, or story events that conflict with
the user's request" — which is right, and which is exactly what a lower-priority
pile of extra speech runs into. So the intensity goes in as well, through the
same seam WanGP's own ``@`` suffix uses: the density to hit, the requirement
that every supplied line reaches the timeline whole, and how H3 marks the
things this feature depends on — stable speaker IDs, ``<d>[Language] ...</d>``
around the exact words and nothing else, ``<scenetrans>`` when a line runs
across a cut. The delivery bracket is direction rather than dialogue and is
written into the prose around the line, never inside the tag.

The pass never raises. Somebody pressed a button to get an H3 prompt; a
dialogue pass that failed costs them the extra speech, not the prompt. The
directive is still applied when the pass comes back empty, because the
intensity is a thing they asked for and the enhancer can act on it alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import enhancer

# The left-hand position, and the default: the H3 request exactly as WanGP
# would have built it. 10 is the right-hand end — somebody talking throughout.
OFF = 0
MOST = 10

# However far the slider goes, a video is a few seconds long and a prompt has a
# context window. The model is asked for what the intensity says, up to this.
LINE_CEILING = 24

# A line of two words flickers past; a line of thirty is a monologue that will
# be cut. Both ends are enforced on what comes back, not asked for and hoped.
MIN_WORDS = 3
MAX_WORDS = 26

# What the enhancer is given per supplied line, on top of the vendored budget:
# the line itself plus the delivery prose around it. Without this the timeline
# at intensity 10 is written up to the token limit and stops mid-sentence.
TOKENS_PER_LINE = 48
TOKEN_CEILING = 4096

# Speech already in the prompt, straight quotes or curly.
QUOTED = re.compile(r'"([^"\n]{2,240})"|[“„]([^”\n]{2,240})[”“]')

# ``(S1) = the woman in the red coat`` and
# ``(S1) [quiet, uncertain] "You said you would be back."`` — the two shapes the
# pass is asked for. Anything else in its reply is not used.
ROSTER = re.compile(r'^\(?\s*(S\d+)\s*\)?\s*[=:]\s*(.+?)\s*$')
LINE = re.compile(r'^\(?\s*(S\d+)\s*\)?\s*\[\s*([^\]]{0,80}?)\s*\]\s*(.+?)\s*$')


@dataclass(frozen=True, slots=True)
class Line:
    """One thing somebody says, and how they say it."""

    speaker: str
    delivery: str
    text: str

    def __str__(self) -> str:
        bracket = f"[{self.delivery}] " if self.delivery else ""
        return f'({self.speaker}) {bracket}"{self.text}"'


# ── what a position on the slider means ──────────────────────────────────────

def clamp(intensity) -> int:
    try:
        value = int(intensity)
    except (TypeError, ValueError):
        return OFF
    return max(OFF, min(MOST, value))


def share(intensity: int) -> int:
    """Roughly how much of the running time carries speech, as a percentage.

    1 is a scene with some talking in it; 10 is a scene that is talking, which
    is 95 rather than 100 because a video whose every frame is mid-word has no
    room to breathe and reads as a fault rather than as a style.
    """
    intensity = clamp(intensity)
    if intensity <= OFF:
        return 0
    return round(20 + (intensity - 1) * (95 - 20) / (MOST - 1))


def target_lines(intensity: int) -> int:
    """How many spoken lines the whole video should end up with."""
    intensity = clamp(intensity)
    if intensity <= OFF:
        return 0
    return round(2 + (intensity - 1) * (LINE_CEILING - 2) / (MOST - 1))


def wanted(intensity: int, quoted: int) -> int:
    """How many new lines the pass is asked for.

    Speech the prompt already quotes counts toward the target: somebody who
    typed six lines and set the slider low did not ask for six more.
    """
    return max(0, target_lines(intensity) - max(0, int(quoted)))


def max_tokens(variant: str, lines: int, intensity: int = MOST) -> int:
    """The enhancer's budget, with room for the speech that was added.

    At Off this is ``enhancer.max_tokens`` and nothing else, which is WanGP's
    own number for the variant.
    """
    base = enhancer.max_tokens(variant)
    if clamp(intensity) <= OFF or lines <= 0:
        return base
    return min(TOKEN_CEILING, base + TOKENS_PER_LINE * int(lines))


def describe(intensity: int) -> str:
    """The sentence under the slider. Ten positions of a bare track say nothing."""
    intensity = clamp(intensity)
    if intensity <= OFF:
        return "Speech only where the prompt asks for it"
    return (f"{share(intensity)}% of the video is talking — about {target_lines(intensity)} "
            "lines, written for whoever is in the scene")


# ── reading the prompt ───────────────────────────────────────────────────────

def quoted_lines(text: str) -> list[str]:
    """The speech the prompt already quotes, in the order it appears."""
    found: list[str] = []
    for straight, curly in QUOTED.findall(text or ""):
        line = (straight or curly).strip()
        if line and line not in found:
            found.append(line)
    return found


# ── the pass that writes the lines ───────────────────────────────────────────

SYSTEM = """You cast and write the spoken audio for a video that is about to be generated.

You are given the description of that video. Decide who speaks in it, then write the exact words they say.

Who speaks:
- If the description has people, characters or creatures in it who could plausibly speak, the lines are theirs. Give each one a speaker ID (S1), (S2), and so on, in the order they first matter to the scene.
- If nothing in the description can speak on camera, the voice is one unseen narrator, (S1). Take the narrator's register from what the scene actually is — hushed and observing, urgent and clipped, warm and remembering, dry and amused — and have them speak about the scene as it happens.
- If lines are already quoted in the description, those speakers exist already. Keep their voice, register and subject exactly, and write the new lines as more of the same conversation.

Output this and nothing else. No preamble, no numbering, no explanation, no Markdown, no blank line inside a section:

SPEAKERS
(S1) = who they are, in a few words, exactly as the description has them
(S2) = ...

LINES
(S1) [delivery] "The exact words spoken."
(S2) [delivery] "The exact words spoken."

Every LINES entry is one speaker ID, one bracket and one double-quoted line, on a single line of output.

The bracket is how the line is said — two or three words of tone, volume, pace, and what the speaker feels as they say it. It is direction. It is never part of the words spoken.

The words are ordinary speech said out loud, three to twenty-six words long, in the language the description is written in unless it asks for another. They run in the order they are spoken, each following from the one before as real talk does, and together they carry the scene from its beginning to its end. Invent no character, no place and no event the description does not have."""

# What the pass is told about how densely the lines should run, by band. The
# same quantity the directive states to the enhancer, said to the writer in the
# terms a writer works in — spacing, not percentages.
PACE = ((40, "Leave silence between the lines. This is a scene with some talking in it, "
             "not a conversation."),
        (75, "The talking carries most of the scene. Keep the lines coming, with only short "
             "gaps between them."),
        (101, "The scene is wall-to-wall speech. Each line runs straight into the next with "
              "almost no silence anywhere, and the talking continues while the action does."))


def pace(intensity: int) -> str:
    percent = share(intensity)
    for ceiling, text in PACE:
        if percent < ceiling:
            return text
    return PACE[-1][1]


def messages(prompt: str, intensity: int, count: int, *,
             image_caption: str | None = None, quoted: list[str] | None = None) -> list[dict]:
    """The pass's own request: what the video is, and how much of it is talking."""
    parts = [f"VIDEO: {str(prompt or '').strip()}"]
    if image_caption:
        parts.append(f"WHAT THE OPENING PICTURE SHOWS: {image_caption.strip()}")
    if quoted:
        spoken = "\n".join(f'"{line}"' for line in quoted)
        parts.append(f"LINES ALREADY SPOKEN IN IT:\n{spoken}")
    parts.append(f"PACE: {pace(intensity)}")
    parts.append(f"Write the speakers and {count} lines now.")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


def parse(raw: str, existing: list[str] | None = None,
          count: int = LINE_CEILING) -> tuple[list[tuple[str, str]], list[Line]]:
    """The pass's reply as a cast and a script.

    Lenient about everything except the two things that matter: a line has to
    be speech, and it has to belong to somebody. A reply that lost the speaker
    IDs is given to the first speaker rather than thrown away, because a
    monologue is a usable answer and an empty pane is not.

    A speaker who ends up with nothing to say is dropped along with their lines.
    Naming somebody in the cast and then never having them speak is an invitation
    to write them a line, and the count of lines is the whole of what the slider
    controls.
    """
    roster: list[tuple[str, str]] = []
    lines: list[Line] = []
    seen = {line.casefold() for line in (existing or [])}
    named: set[str] = set()

    for row in (raw or "").splitlines():
        row = row.strip().lstrip("-*").strip()
        if not row or row.rstrip(":").upper() in {"SPEAKERS", "LINES"}:
            continue

        spoken = LINE.match(row)
        if spoken is not None:
            speaker, delivery, text = spoken.groups()
            line = _kept(speaker.upper(), delivery, text, seen)
            if line is not None:
                lines.append(line)
                seen.add(line.text.casefold())
                if len(lines) >= count:
                    break
            continue

        cast = ROSTER.match(row)
        if cast is not None and '"' not in row and "[" not in row:
            speaker, who = cast.group(1).upper(), cast.group(2).strip()
            if who and speaker not in named:
                roster.append((speaker, who))
                named.add(speaker)
            continue

        # A line that kept its quotes but lost its prefix: still speech.
        for candidate in quoted_lines(row):
            line = _kept(roster[0][0] if roster else "S1", "", candidate, seen)
            if line is not None:
                lines.append(line)
                seen.add(line.text.casefold())
        if len(lines) >= count:
            break

    lines = lines[:count]
    speaking = {line.speaker for line in lines}
    return [entry for entry in roster if entry[0] in speaking], lines


def _kept(speaker: str, delivery: str, text: str, seen: set[str]) -> Line | None:
    """One entry, or nothing when it is not a line this can use."""
    text = text.strip().strip('"“”').strip()
    if not text or text.casefold() in seen:
        return None
    if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
        return None
    # A model that starts explaining itself part-way down the list: drop the
    # sentence rather than the rest, since what follows it is usually fine.
    if text.rstrip(".!?").casefold().startswith(("here are", "here is", "note", "these lines")):
        return None
    delivery = re.sub(r"\s+", " ", str(delivery or "")).strip().strip(".,;")
    return Line(speaker=speaker, delivery=delivery, text=text)


def transcript(roster: list[tuple[str, str]], lines: list[Line]) -> str:
    """The cast and the script, as the pane shows them and as the request says them."""
    cast = "\n".join(f"({speaker}) = {who}" for speaker, who in roster)
    script = "\n".join(str(line) for line in lines)
    return "\n\n".join(part for part in (cast, script) if part)


def write(prompt: str, intensity: int, chat_stream, *, image_caption: str | None = None,
          seed=None) -> tuple[list[tuple[str, str]], list[Line], str]:
    """``(roster, lines, what happened)``. Never raises — see the module docstring.

    The prompt is split on ``@`` first, so a generation carrying its own
    instructions has its dialogue written about the video rather than about the
    instructions.
    """
    intensity = clamp(intensity)
    body = enhancer.split_system_suffix(prompt)[0]
    if intensity <= OFF or not body.strip():
        return [], [], ""

    quoted = quoted_lines(body)
    count = wanted(intensity, len(quoted))
    if count <= 0:
        return [], [], "the prompt already speaks this much"

    try:
        raw = "".join(chat_stream(
            messages(body, intensity, count, image_caption=image_caption, quoted=quoted),
            temperature=0.9, top_p=0.95, max_tokens=128 + 44 * count, seed=seed))
    except Exception as exc:                      # noqa: BLE001 - never costs the prompt
        return [], [], f"skipped ({exc})"

    roster, lines = parse(raw, quoted, count)
    if not lines:
        return [], [], "none came back — the intensity still applies"
    voices = len({line.speaker for line in lines})
    who = "one voice" if voices == 1 else f"{voices} voices"
    return roster, lines, f"{len(lines)} lines in {who}"


# ── the directive that makes them land ───────────────────────────────────────

# Which field of the finished prompt the timeline is, by variant. Ref2VA writes
# six sections and FL2VA three, and the directive has to name the right one or
# it is telling the model to put speech in a field the model is not writing.
TIMELINE = {enhancer.FL2VA: "integrated_multimodal_description",
            enhancer.REF2VA: "detailed_description"}

HEADER = ("The user has set the dialogue intensity for this generation. It takes priority "
          "over any default reluctance to add speech, and over nothing else:")

CONTINUOUS = ("Speech is continuous from the first frame to the last. A voice is talking at "
              "essentially every moment, with no silence longer than about half a second "
              "anywhere in the video, and the action described carries on underneath the "
              "talking rather than pausing for it.")

SUPPLIED = ("The cast and the lines listed under \"Spoken audio\" in the request are required "
            "content. Every line appears in the finished timeline, in the order given, word "
            "for word, inside <d>[Language] ...</d>, spoken by the speaker ID it was given. Do "
            "not merge, shorten, paraphrase, reorder or drop any of them. Where the running "
            "time is tight, close the gaps between lines and let them overlap the action "
            "rather than cutting a line.")

INVENT = ("No lines are supplied. Write the speech yourself, from the people the request "
          "describes; if nobody in it can speak on camera, use one unseen narrator whose "
          "register comes from what the scene is.")

SPEAKERS = ("Speaker IDs stay exactly as given and stay stable across every shot. A speaker "
            "the cast describes as unseen, a narrator or a voice-over has no source in frame: "
            "write them as an off-screen voice over the action, and do not invent a person, a "
            "radio, a screen or a telephone to account for them. A speaker who is someone in "
            "the scene speaks on camera, with mouth movement and breath matching the line.")

DELIVERY = ("Each supplied line carries its emotional delivery in square brackets. That "
            "bracket is direction, never dialogue: its words must not appear inside "
            "<d>...</d>. Render it in the prose immediately around the line — tone, volume, "
            "pace, breath, expression and what the body is doing as the words come out — so "
            "the feeling is heard and seen rather than stated.")

ACROSS_CUTS = ("A line that runs across a hard cut is marked <scenetrans> on both sides with "
               "an explicit statement that the audio continues across it. <cutoff> is only for "
               "a line the final frame genuinely interrupts.")

BOUNDS = ("Speech belongs in {timeline} and nowhere else: do not repeat any of it in "
          "overall_soundscape, and do not turn narration into non_diegetic_music. Everything "
          "else the request asks for — the action, the setting, the camera work, the ending — "
          "is unchanged. The speech accompanies that video; it does not replace it.")


def directive(intensity: int, lines: int = 0, *, variant: str = enhancer.FL2VA) -> str:
    """What the enhancer is told on top of its own instructions, or "" at Off."""
    intensity = clamp(intensity)
    if intensity <= OFF:
        return ""
    percent = share(intensity)
    density = (CONTINUOUS if percent >= 90 else
               f"This is a speaking scene: roughly {percent}% of the video's running time "
               "carries audible speech, spread across the whole of it rather than gathered "
               "into one stretch.")
    timeline = TIMELINE[enhancer.variant_of(variant)]
    parts = [HEADER, density, SUPPLIED if lines > 0 else INVENT, SPEAKERS]
    if lines > 0:
        parts.append(DELIVERY)
    parts += [ACROSS_CUTS, BOUNDS.format(timeline=timeline)]
    return "\n".join(parts)


def spoken_audio(roster: list[tuple[str, str]], lines: list[Line]) -> str:
    """The cast and script as they are attached to the user's own request."""
    written = transcript(roster, lines)
    return f"Spoken audio for this video, all of which is spoken:\n{written}" if written else ""


def mixed(body: str, roster: list[tuple[str, str]], lines: list[Line]) -> str:
    """The prompt with the speech added to it, as more of the same request."""
    attached = spoken_audio(roster, lines)
    if not attached:
        return body
    return f"{body.rstrip()}\n\n{attached}"


def request(prompt: str, *, variant: str = enhancer.FL2VA, image_caption: str | None = None,
            intensity: int = OFF, roster: list[tuple[str, str]] | None = None,
            lines: list[Line] | None = None) -> list[dict[str, str]]:
    """The whole H3 request, with the dialogue in it.

    Built from ``enhancer``'s own pieces rather than around them, and at Off it
    is what ``enhancer.messages`` returns, element for element: no directive, no
    attached script, the vendored instructions and the labelled user turn.

    The directive always goes in ahead of whatever ``@`` added, so a prompt
    carrying instructions of its own still has the last word — which is what
    WanGP's own preamble promises it. ``@@`` is the one case worth stating:
    it replaces WanGP's instructions, and the slider is not those. It is a
    control on this page that was deliberately moved, so it survives, before
    the replacement rather than after it.
    """
    intensity = clamp(intensity)
    lines = list(lines or [])
    body, suffix, replace = enhancer.split_system_suffix(prompt)
    added = directive(intensity, len(lines), variant=variant)
    if intensity > OFF:
        body = mixed(body, list(roster or []), lines)
    if replace:
        system = "\n".join(part for part in (added, suffix) if part)
    else:
        instructions = enhancer.instructions(variant, image_caption is not None)
        if added:
            instructions = f"{instructions.rstrip()}\n{added}"
        system = enhancer.merge_system(instructions, suffix, False)
    return [{"role": "system", "content": system},
            {"role": "user", "content": enhancer.user_content(body, image_caption)}]
