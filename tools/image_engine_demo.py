#!/usr/bin/env python3
"""
tools/image_engine_demo.py — exercise the image engine with no server running.

    python tools/image_engine_demo.py

Uses a stub writer, so what you see is the engine's plumbing: sentinel
resolution, brief construction, per-model divergence, sanitising and negative
gating. The prompt text itself is canned. Point ``ImagePromptEngine`` at the
real LlamaClient to see it write for real.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt_master.image_engine import (
    AUTO,
    CHOOSE,
    RANDOM,
    EditControls,
    ImageControls,
    ImagePromptEngine,
    PROFILES,
    ReferenceImage,
    build_edit_system,
    build_system,
    editing_profiles,
)

WRAP = 76


def rule(title: str = "") -> None:
    print("\n" + "=" * WRAP)
    if title:
        print(title)
        print("-" * WRAP)


def show(label: str, text: str) -> None:
    print(f"{label}:")
    print(textwrap.fill(text, WRAP, initial_indent="  ", subsequent_indent="  "))


class StubWriter:
    """Canned replies. Real usage injects the application's LlamaClient."""

    GEN = (
        "A shipwright in her fifties kneels on a half-planked hull, driving a "
        "caulking iron into a seam with short controlled strokes, documentary "
        "photograph, shot on a 35mm lens at f/5.6, lit by soft daylight "
        "through the open boatshed doors, weathered timber and loose oakum "
        "around her knees, visible skin pores and fine lines, a muted earth "
        "palette of clay and dust."
    )
    EDIT = (
        "Replace the weathered wooden door on the left of the frame with a "
        "riveted steel door of the same width and position, its surface "
        "scratched and lightly rusted along the base. Keep the surrounding "
        "brickwork, the direction of the afternoon light, the photograph's "
        "grain and the framing exactly as they are."
    )
    SELECT = '{"lighting": "Soft window light", "mood": "Weary"}'

    def __init__(self) -> None:
        self.calls = 0

    def stream_chat(self, *, system, user, temperature, top_p, seed, image_jpeg_b64=None):
        self.calls += 1
        if "choose settings" in system.lower():
            body = self.SELECT
        elif "editing instruction" in system.lower():
            body = self.EDIT
        elif "failure modes" in system.lower():
            body = "harsh on-camera flash, blown white shirt"
        else:
            body = self.GEN
        for word in body.split(" "):
            yield word + " "


# ---------------------------------------------------------------------------
# 1. The same intent across all three profiles
# ---------------------------------------------------------------------------

rule("1. ONE INTENT, THREE MODELS — note where the negative appears")

intent = "a shipwright caulking a wooden hull in a boatshed"
for key in PROFILES:
    controls = ImageControls(
        profile_key=key,
        shot_type="Medium shot",
        lens="35mm reportage",
        aperture="f/5.6 — balanced",
        lighting="Soft window light",
        visual_style="Documentary photograph",
        colour_palette="Muted earth",
        material_focus="Weathered timber",
        seed=1234,
    )
    result = ImagePromptEngine(StubWriter()).generate(
        intent, controls, use_model_for_choice=False
    )
    print(f"\n{result.status_line()}")
    show("  positive", result.positive)
    if result.negative:
        show("  negative", result.negative)


# ---------------------------------------------------------------------------
# 2. The three sentinels
# ---------------------------------------------------------------------------

rule("2. AUTO vs CHOOSE vs RANDOM on the same intent")

night = "a lone bartender closing up after a long shift"

for label, controls in [
    ("AUTO   ", ImageControls(lighting=AUTO, mood=AUTO, seed=7)),
    ("CHOOSE ", ImageControls(lighting=CHOOSE, mood=CHOOSE, seed=7)),
    ("RANDOM ", ImageControls(lighting=RANDOM, mood=RANDOM, seed=7)),
]:
    brief = ImagePromptEngine(None).plan(night, controls)
    sel = brief.selection
    chosen = sel.resolved() if sel else {}
    src = sel.source if sel else {}
    rendered = (
        ", ".join(f"{k}={v!r} [{src.get(k)}]" for k, v in chosen.items()) or "— nothing stated —"
    )
    print(f"{label} {rendered}")

print("\nRANDOM is reproducible for a fixed seed, fresh when the seed is -1:")
for seed in (7, 7, 42, -1, -1):
    brief = ImagePromptEngine(None).plan(night, ImageControls(lighting=RANDOM, seed=seed))
    got = (brief.selection.resolved() if brief.selection else {}).get("lighting", "—")
    print(f"  seed {str(seed):>4} -> {got}")

print("\nCHOOSE with the model available (vs the local fallback):")
engine = ImagePromptEngine(StubWriter())
brief = engine.plan(night, ImageControls(lighting=CHOOSE, mood=CHOOSE, seed=7))
sel = brief.selection
print(f"  {sel.resolved()}  sources={sel.source}")


# ---------------------------------------------------------------------------
# 3. Edit mode
# ---------------------------------------------------------------------------

rule("3. EDIT MODE — a terse request becomes a full instruction")

print("Models offered in edit mode by default:")
for p in editing_profiles():
    print(f"  {p.display_name} — up to {p.max_reference_images} references, "
          f"{p.edit_steps} steps, CFG {p.edit_cfg:g}")
print("Only with the experimental option enabled:")
for p in editing_profiles(include_experimental=True):
    if p.edit_is_experimental:
        print(f"  {p.display_name}")

request = "swap the wooden door for a steel one"
controls = EditControls(
    operation=CHOOSE,
    target_element="the weathered wooden door",
    localization="Left of frame",
    preservation=("Identity", "Background", "Lighting", "Texture and grain"),
    edit_strength="Balanced",
    seed=2024,
)
engine = ImagePromptEngine(StubWriter())
result = engine.edit(request, controls)
print(f"\nrequest : {request!r}")
print(f"{result.status_line()}")
print(f"operation inferred: {result.operation!r}")
show("instruction", result.instruction)
for w in result.warnings:
    print(f"  warning: {w}")

print("\nWith references, addressed by number AND role:")
multi = EditControls(
    operation="Transfer a subject",
    references=(
        ReferenceImage(1, "Subject"),
        ReferenceImage(2, "Background"),
        ReferenceImage(3, "Lighting"),
    ),
    seed=5,
)
brief = ImagePromptEngine(None).plan_edit("put him in the other scene", multi)
print("  " + " | ".join(brief.reference_phrases()))

print("\nA model that cannot edit refuses up front:")
try:
    ImagePromptEngine(None).plan_edit("make it sunset", EditControls(profile_key="krea_2_turbo"))
except ValueError as exc:
    print(f"  ValueError: {str(exc)[:150]}...")


# ---------------------------------------------------------------------------
# 4. Why generation and edit need different sanitisers
# ---------------------------------------------------------------------------

rule("4. THE SANITISERS ARE NOT INTERCHANGEABLE")

from prompt_master.image_engine import FLUX_KLEIN_9B, sanitize_edit_instruction, sanitize_positive

# Preservation clauses are often phrased negatively, and that is exactly what
# the generation sanitiser is built to delete.
for sample in [
    "Add a bench beside the tree, without altering the background.",
    "Make the sky overcast; nothing else changes.",
]:
    print(f"\n  in : {sample}")
    print(f"  gen: {sanitize_positive(sample, FLUX_KLEIN_9B)}   <- clause gone")
    print(f"  ed : {sanitize_edit_instruction(sample, FLUX_KLEIN_9B)[0]}   <- clause kept")


# ---------------------------------------------------------------------------
# 5. System prompt divergence
# ---------------------------------------------------------------------------

rule("5. SYSTEM PROMPTS DIVERGE BY MODEL AND BY TASK")

gen_flux = build_system(ImagePromptEngine(None).plan("x", ImageControls(profile_key="flux_klein_9b")))
gen_krea = build_system(ImagePromptEngine(None).plan("x", ImageControls(profile_key="krea_2_turbo")))
edit_flux = build_edit_system(ImagePromptEngine(None).plan_edit("x", EditControls()))

for name, text in [
    ("generation / FLUX", gen_flux),
    ("generation / Krea", gen_krea),
    ("edit       / FLUX", edit_flux),
]:
    print(f"  {name}: {len(text):>5} chars")
print(f"\n  generation prompts identical across models? {gen_flux == gen_krea}")
print(f"  edit prompt identical to generation?        {edit_flux == gen_flux}")
print()
