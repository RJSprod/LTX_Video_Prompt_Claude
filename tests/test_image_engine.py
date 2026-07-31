"""
tests/test_image_engine.py — the Image Prompt mode suite.

Pure stdlib + pytest. No Qt, no llama-server, no network: the writer is a
scripted double, so every assertion is deterministic.

The most important tests here are ``test_no_import_from_prompt_engine`` and
``test_imports_are_stdlib_only``. They are the mechanical guarantee that Image
Prompt mode cannot affect LTX prompt parity. If either fails, the parity report
is no longer trustworthy — treat it as a release blocker, not a flaky test.

As delivered, this suite imported the engine as a top-level ``image_engine``.
The package lives at ``prompt_master.image_engine`` here, so the two import
lines below are the only edit made to it — everything else is the bundle's own
file, which is what keeps re-syncing a new engine drop to a copy and two lines.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from prompt_master import image_engine as ie
from prompt_master.image_engine import (
    AUTO,
    CHOOSE,
    FLUX_KLEIN_9B,
    KREA_2_RAW,
    KREA_2_TURBO,
    NEGATIVE_BANKS,
    NO_PREFERENCE,
    PROFILES,
    RANDOM,
    SELECTABLE_BANKS,
    UNIVERSAL_AVOID,
    EditCapability,
    EditControls,
    EditGeneration,
    Generation,
    ImageControls,
    ImagePromptEngine,
    NegativeMode,
    PromptShape,
    ReferenceImage,
    SelectionPass,
    build_brief,
    build_edit_brief,
    build_edit_system,
    build_edit_user_turn,
    build_negative,
    build_system,
    build_user_turn,
    editing_profiles,
    gates_for,
    heuristic_pick,
    option_banks,
    profile_for,
    random_pick,
    resolve_dimensions,
    resolve_edit_sentinels,
    resolve_sentinels,
    sanitize_edit_instruction,
    sanitize_positive,
    seeded_rng_for,
)

PKG = Path(ie.__file__).parent


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class ScriptedClient:
    """Returns canned text and records exactly what it was asked."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies) or [""]
        self.calls: list[dict] = []

    def stream_chat(self, *, system, user, temperature, top_p, seed, image_jpeg_b64=None):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "image": image_jpeg_b64,
            }
        )
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        for word in reply.split(" "):
            yield word + " "


class ExplodingClient:
    """Fails after the first call, to exercise fallback paths."""

    def __init__(self, first: str = "") -> None:
        self._first = first
        self.n = 0

    def stream_chat(self, **kwargs):
        self.n += 1
        if self.n == 1 and self._first:
            yield self._first
        else:
            raise RuntimeError("server went away")


PROSE = (
    "A shipwright in her fifties kneels on a half-planked hull, driving a "
    "caulking iron into a seam, documentary photograph, shot on a 35mm lens "
    "at f/5.6, lit by soft daylight through the boatshed doors, weathered "
    "timber and oakum, visible skin pores and fine lines."
)

EDIT = (
    "Replace the wooden door on the left of the frame with a riveted steel "
    "door of the same size and position, its surface scratched and lightly "
    "rusted at the base. Keep the surrounding brickwork, the light direction "
    "and the photograph's grain exactly as they are."
)


# ---------------------------------------------------------------------------
# The wall
# ---------------------------------------------------------------------------


def _modules() -> list[Path]:
    return sorted(PKG.glob("*.py"))


def test_no_import_from_prompt_engine() -> None:
    """Image mode must never reach the vendored LTX engine.

    An import would place this package inside the surface that
    tools/compare_upstream_engine.py verifies, and "0 mismatches" would stop
    meaning what PARITY_REPORT.md says it means.
    """
    for path in _modules():
        tree = ast.parse(path.read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module)
        bad = [m for m in imported if "prompt_engine" in m or "prompt_master" in m]
        assert not bad, f"{path.name} imports {bad}"


def test_imports_are_stdlib_only() -> None:
    """The client is injected, so the engine loads with no Qt and no network."""
    roots: set[str] = set()
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    third_party = roots - set(sys.stdlib_module_names) - {"image_engine"}
    assert not third_party, f"non-stdlib imports: {third_party}"


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_distilled_profiles_declare_no_negative() -> None:
    for p in (FLUX_KLEIN_9B, KREA_2_TURBO):
        assert p.supports_negative is False
        assert p.negative_mode is NegativeMode.NONE
        assert p.negative_supported is False


def test_raw_profile_supports_negative() -> None:
    assert KREA_2_RAW.negative_supported is True
    assert KREA_2_RAW.negative_mode is NegativeMode.CFG
    assert KREA_2_RAW.default_cfg > 1.0


def test_no_profile_honours_attention_weights() -> None:
    assert not any(p.honours_attention_weights for p in PROFILES.values())


def test_only_flux_edits_natively() -> None:
    assert FLUX_KLEIN_9B.edit_capability is EditCapability.NATIVE
    for p in (KREA_2_TURBO, KREA_2_RAW):
        assert p.edit_capability is EditCapability.EXPERIMENTAL


def test_editing_profiles_excludes_experimental_by_default() -> None:
    assert [p.key for p in editing_profiles()] == ["flux_klein_9b"]
    assert len(editing_profiles(include_experimental=True)) == 3


def test_experimental_profiles_state_their_requirements() -> None:
    for p in PROFILES.values():
        if p.edit_is_experimental:
            assert p.edit_requirements, f"{p.key} is experimental but says nothing"


def test_neither_model_does_masked_inpainting() -> None:
    assert not any(p.supports_masked_inpainting for p in PROFILES.values())


def test_profile_for_falls_back_rather_than_raising() -> None:
    assert profile_for("a_model_from_the_future").key in PROFILES


def test_profiles_have_sane_word_bands() -> None:
    for p in PROFILES.values():
        assert p.min_words < p.target_words <= p.max_words < p.hard_word_ceiling
        assert p.edit_min_words < p.edit_target_words <= p.edit_max_words


# ---------------------------------------------------------------------------
# Sentinels: RANDOM
# ---------------------------------------------------------------------------


def test_random_is_reproducible_for_a_seed() -> None:
    a = random_pick(99, "lighting", list(ie.options.LIGHTING))
    b = random_pick(99, "lighting", list(ie.options.LIGHTING))
    assert a == b


def test_random_differs_between_seeds() -> None:
    picks = {random_pick(s, "visual_style", list(ie.options.VISUAL_STYLES)) for s in range(12)}
    assert len(picks) > 1


def test_random_is_independent_per_control() -> None:
    """Adding a control must not shift another control's draw.

    A shared sequential RNG would break every stored seed the moment a new
    control landed. Per-control derivation is what prevents that.
    """
    before = random_pick(7, "lighting", list(ie.options.LIGHTING))
    _ = random_pick(7, "a_brand_new_control", ["x", "y", "z"])
    after = random_pick(7, "lighting", list(ie.options.LIGHTING))
    assert before == after


def test_seeded_rng_is_stable_across_processes() -> None:
    """crc32, not builtin hash: string hashing is salted per process."""
    assert seeded_rng_for(5, "lighting").random() == seeded_rng_for(5, "lighting").random()


def test_random_control_resolves_to_a_real_option() -> None:
    controls = ImageControls(lighting=RANDOM, visual_style=RANDOM, seed=3)
    settled, outcome = resolve_sentinels("a barn", controls, 3, use_model=False)
    assert settled.lighting in ie.options.LIGHTING
    assert outcome.source["lighting"] == "random"


# ---------------------------------------------------------------------------
# Sentinels: CHOOSE
# ---------------------------------------------------------------------------


def test_heuristic_picks_something_sensible() -> None:
    assert heuristic_pick("a neon-lit street at night in the rain", ie.options.LIGHTING) == "Neon"
    assert heuristic_pick("a macro photograph of a dew droplet", ie.options.LENSES) == "100mm macro"
    assert heuristic_pick("an anime panel of a schoolgirl", ie.options.VISUAL_STYLES) == "Manga"


def test_heuristic_returns_empty_when_nothing_fits() -> None:
    assert heuristic_pick("qqq zzz", ie.options.LIGHTING) == ""


def test_choose_without_a_client_uses_the_heuristic() -> None:
    controls = ImageControls(lighting=CHOOSE, seed=1)
    settled, outcome = resolve_sentinels(
        "a campfire in the woods after dark", controls, 1, use_model=False
    )
    assert settled.lighting == "Firelight"
    assert outcome.source["lighting"] == "heuristic"


def test_choose_uses_the_model_when_available() -> None:
    client = ScriptedClient('{"lighting": "Chiaroscuro"}')
    picker = SelectionPass(client)
    settled, outcome = resolve_sentinels(
        "a lone figure in a dark hall",
        ImageControls(lighting=CHOOSE),
        5,
        selection_pass=picker,
    )
    assert settled.lighting == "Chiaroscuro"
    assert outcome.source["lighting"] == "model"


def test_model_choice_is_validated_against_the_bank() -> None:
    """An invented option must not reach the brief."""
    client = ScriptedClient('{"lighting": "Disco Inferno Lighting"}')
    settled, outcome = resolve_sentinels(
        "a campfire at night",
        ImageControls(lighting=CHOOSE),
        1,
        selection_pass=SelectionPass(client),
    )
    assert settled.lighting in ie.options.LIGHTING or settled.lighting == AUTO
    assert outcome.source["lighting"] != "model"


def test_case_drift_is_tolerated() -> None:
    client = ScriptedClient('{"lighting": "chiaroscuro"}')
    settled, _ = resolve_sentinels(
        "a dark hall",
        ImageControls(lighting=CHOOSE),
        1,
        selection_pass=SelectionPass(client),
    )
    assert settled.lighting == "Chiaroscuro"


def test_escape_value_falls_back_to_auto() -> None:
    client = ScriptedClient(json.dumps({"lighting": NO_PREFERENCE}))
    settled, outcome = resolve_sentinels(
        "something abstract",
        ImageControls(lighting=CHOOSE),
        1,
        selection_pass=SelectionPass(client),
    )
    assert settled.lighting == AUTO
    assert outcome.source["lighting"] == "none"


def test_selection_failure_falls_back_to_heuristic() -> None:
    settled, outcome = resolve_sentinels(
        "a campfire in the woods",
        ImageControls(lighting=CHOOSE),
        1,
        selection_pass=SelectionPass(ExplodingClient()),
    )
    assert settled.lighting == "Firelight"
    assert outcome.source["lighting"] == "heuristic"
    assert any("fell back" in n for n in outcome.notes)


def test_choose_is_batched_not_one_call_each() -> None:
    client = ScriptedClient("{}")
    controls = ImageControls(
        lighting=CHOOSE, visual_style=CHOOSE, mood=CHOOSE, lens=CHOOSE
    )
    resolve_sentinels("a barn", controls, 1, selection_pass=SelectionPass(client))
    assert len(client.calls) == 1


def test_large_choose_sets_are_split_into_several_calls() -> None:
    client = ScriptedClient("{}")
    controls = ImageControls(**{k: CHOOSE for k in list(SELECTABLE_BANKS)[:12]
                                if k in ImageControls().__dict__})
    resolve_sentinels("a barn", controls, 1, selection_pass=SelectionPass(client))
    assert len(client.calls) >= 2


def test_options_are_shuffled_to_blunt_position_bias() -> None:
    """Two controls must not see the bank in the same order."""
    client = ScriptedClient("{}")
    resolve_sentinels(
        "a barn",
        ImageControls(lighting=CHOOSE, time_of_day=CHOOSE),
        1,
        selection_pass=SelectionPass(client),
    )
    prompt = client.calls[0]["user"]
    lighting_block = prompt.split("lighting:")[1].split("time_of_day:")[0]
    listed = re.findall(r"^  - (.+?)(?:  \(|$)", lighting_block, re.M)
    assert listed and listed != list(ie.options.LIGHTING)


def test_selection_prompt_offers_the_escape_value() -> None:
    client = ScriptedClient("{}")
    resolve_sentinels(
        "a barn", ImageControls(mood=CHOOSE), 1, selection_pass=SelectionPass(client)
    )
    assert NO_PREFERENCE in client.calls[0]["system"]


def test_selection_runs_cold() -> None:
    client = ScriptedClient("{}")
    resolve_sentinels(
        "a barn", ImageControls(mood=CHOOSE), 1, selection_pass=SelectionPass(client)
    )
    assert client.calls[0]["temperature"] <= 0.35


def test_resolve_does_not_mutate_the_callers_controls() -> None:
    controls = ImageControls(lighting=CHOOSE)
    resolve_sentinels("a campfire", controls, 1, use_model=False)
    assert controls.lighting == CHOOSE


def test_chosen_values_are_reported_on_the_result() -> None:
    engine = ImagePromptEngine(ScriptedClient(PROSE))
    result = engine.generate(
        "a campfire in the woods at night",
        ImageControls(lighting=CHOOSE, seed=1),
        use_model_for_choice=False,
    )
    assert result.chosen.get("lighting") == "Firelight"
    assert result.chosen_source.get("lighting") == "heuristic"


# ---------------------------------------------------------------------------
# Brief and seeding
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_brief() -> None:
    c = ImageControls(shot_type=RANDOM, lighting=RANDOM, wardrobe=RANDOM, seed=4242)
    s1, _ = resolve_sentinels("a welder", c, 4242, use_model=False)
    s2, _ = resolve_sentinels("a welder", c, 4242, use_model=False)
    assert build_brief("a welder", s1, 4242).people == build_brief("a welder", s2, 4242).people


def test_auto_controls_are_omitted_not_defaulted() -> None:
    brief = build_brief("a harbour", ImageControls(shot_type=AUTO, lighting=AUTO), 1)
    assert brief.framing == [] and brief.light == []


def test_people_banks_only_fire_when_people_are_present() -> None:
    assert build_brief("an empty warehouse", ImageControls(), 3).people == []
    assert build_brief("a welder in a warehouse", ImageControls(), 3).people != []


def test_seed_shared_between_brief_and_sampler() -> None:
    client = ScriptedClient(PROSE)
    ImagePromptEngine(client).generate("a welder", ImageControls(seed=77))
    assert client.calls[0]["seed"] == 77


def test_negative_one_draws_a_real_seed() -> None:
    engine = ImagePromptEngine(ScriptedClient(PROSE))
    assert engine.plan("x", ImageControls(seed=-1)).seed >= 0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", list(PROFILES.values()))
@pytest.mark.parametrize("ratio", list(option_banks()["aspect_ratios"]))
def test_dimensions_are_always_legal(profile, ratio) -> None:
    w, h = resolve_dimensions(ratio, 2048, profile)
    assert w % profile.edge_multiple == 0 and h % profile.edge_multiple == 0
    assert profile.min_edge <= max(w, h) <= profile.max_edge
    assert w * h <= profile.max_pixels


def test_absurd_long_edge_is_clamped_not_rejected() -> None:
    w, h = resolve_dimensions("16:9 widescreen", 99999, KREA_2_TURBO)
    assert max(w, h) <= KREA_2_TURBO.max_edge


# ---------------------------------------------------------------------------
# Generation system prompt
# ---------------------------------------------------------------------------


def test_system_prompts_differ_between_families() -> None:
    flux = build_system(build_brief("x", ImageControls(profile_key="flux_klein_9b"), 1))
    krea = build_system(build_brief("x", ImageControls(profile_key="krea_2_turbo"), 1))
    assert flux != krea


def test_distilled_system_prompt_explains_absent_negative() -> None:
    s = build_system(build_brief("x", ImageControls(profile_key="krea_2_turbo"), 1))
    assert "no negative conditioning" in s


def test_hex_guidance_only_when_model_supports_it() -> None:
    def sys_for(key):
        return build_system(
            build_brief(
                "x",
                ImageControls(
                    profile_key=key, colour_palette="Deep jewel", use_hex_swatches=True
                ),
                1,
            )
        )

    assert "Bind each one" in sys_for("flux_klein_9b")
    assert "Bind each one" not in sys_for("krea_2_turbo")


def test_json_shape_only_offered_to_models_that_parse_it() -> None:
    krea = build_brief("x", ImageControls(profile_key="krea_2_turbo", prompt_shape=PromptShape.JSON), 1)
    flux = build_brief("x", ImageControls(profile_key="flux_klein_9b", prompt_shape=PromptShape.JSON), 1)
    assert krea.prompt_shape is PromptShape.PROSE
    assert flux.prompt_shape is PromptShape.JSON


def test_user_turn_carries_every_resolved_facet() -> None:
    turn = build_user_turn(
        build_brief(
            "a farmer at the gate",
            ImageControls(shot_type="Close-up", lighting="Golden hour", mood="Solemn"),
            1,
        )
    )
    assert "Framing" in turn and "Light" in turn and "Mood" in turn


# ---------------------------------------------------------------------------
# Edit mode
# ---------------------------------------------------------------------------


def _edit_controls(**kw) -> EditControls:
    base = dict(profile_key="flux_klein_9b", operation="Replace an object", seed=1)
    base.update(kw)
    return EditControls(**base)


def test_edit_refuses_a_model_that_cannot_edit() -> None:
    with pytest.raises(ValueError, match="no native editing"):
        build_edit_brief("make it sunset", _edit_controls(profile_key="krea_2_turbo"), 1)


def test_edit_allows_experimental_when_opted_in() -> None:
    brief = build_edit_brief(
        "make it sunset",
        _edit_controls(profile_key="krea_2_turbo", allow_experimental_edit=True),
        1,
    )
    assert brief.experimental is True
    assert brief.requirements


def test_edit_brief_carries_preservation_clauses() -> None:
    brief = build_edit_brief(
        "put her in a red coat",
        _edit_controls(preservation=("Identity", "Background", "Lighting")),
        1,
    )
    assert len(brief.preservation_clauses) == 3
    assert any("face" in c for c in brief.preservation_clauses)


def test_references_are_renumbered_from_one() -> None:
    brief = build_edit_brief(
        "combine these",
        _edit_controls(
            references=(
                ReferenceImage(index=9, role="Subject"),
                ReferenceImage(index=4, role="Background"),
            )
        ),
        1,
    )
    assert [r.index for r in brief.references] == [1, 2]


def test_references_are_capped_at_the_profile_limit() -> None:
    refs = tuple(ReferenceImage(index=i, role="Style") for i in range(1, 9))
    brief = build_edit_brief("combine", _edit_controls(references=refs), 1)
    assert len(brief.references) == FLUX_KLEIN_9B.max_reference_images == 4


def test_flux_references_carry_index_and_role() -> None:
    brief = build_edit_brief(
        "combine",
        _edit_controls(references=(ReferenceImage(index=1, role="Subject"),)),
        1,
    )
    assert brief.reference_phrases() == ["the subject from image 1"]


def test_edit_system_demands_index_and_role_for_flux() -> None:
    brief = build_edit_brief(
        "combine",
        _edit_controls(references=(ReferenceImage(index=1, role="Subject"),)),
        1,
    )
    system = build_edit_system(brief)
    assert "number and its role" in system


def test_edit_system_explains_the_absence_of_a_mask() -> None:
    system = build_edit_system(build_edit_brief("remove the bins", _edit_controls(), 1))
    assert "no mask" in system


def test_edit_system_forbids_pronouns() -> None:
    system = build_edit_system(build_edit_brief("change her coat", _edit_controls(), 1))
    assert "pronoun" in system


def test_edit_strength_maps_to_denoise() -> None:
    weak = build_edit_brief("x", _edit_controls(edit_strength="Preserve — smallest change that works"), 1)
    strong = build_edit_brief("x", _edit_controls(edit_strength="Transform — commit to the change"), 1)
    assert weak.denoise < strong.denoise


def test_compound_requests_are_flagged() -> None:
    single = build_edit_brief("remove the bins", _edit_controls(), 1)
    compound = build_edit_brief(
        "remove the bins and then add a bench and change the sky to sunset",
        _edit_controls(),
        1,
    )
    assert single.compound_warning is False
    assert compound.compound_warning is True


def test_edit_operation_can_be_chosen_for_you() -> None:
    controls = _edit_controls(operation=CHOOSE)
    settled, outcome = resolve_edit_sentinels(
        "please get rid of the parked cars", controls, 1, use_model=False
    )
    assert settled.operation == "Remove an object"
    assert outcome.source["operation"] == "heuristic"


def test_edit_user_turn_states_the_target_and_the_preservation() -> None:
    turn = build_edit_user_turn(
        build_edit_brief(
            "swap the door",
            _edit_controls(target_element="the wooden door", localization="Left of frame"),
            1,
        )
    )
    assert "the wooden door" in turn
    assert "Must survive" in turn


def test_edit_generation_end_to_end() -> None:
    engine = ImagePromptEngine(ScriptedClient(EDIT))
    result = engine.edit("swap the wooden door for steel", _edit_controls(), use_model_for_choice=False)
    assert isinstance(result, EditGeneration)
    assert result.instruction.startswith("Replace")
    assert result.steps == FLUX_KLEIN_9B.edit_steps
    assert result.as_dict()["task"] == "edit"


def test_edit_status_line_reports_denoise_and_refs() -> None:
    engine = ImagePromptEngine(ScriptedClient(EDIT))
    result = engine.edit(
        "combine",
        _edit_controls(references=(ReferenceImage(1, "Subject"), ReferenceImage(2, "Background"))),
        use_model_for_choice=False,
    )
    assert "denoise" in result.status_line()
    assert "2 refs" in result.status_line()


def test_finish_edit_before_stream_edit_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        ImagePromptEngine(ScriptedClient(EDIT)).finish_edit()


# ---------------------------------------------------------------------------
# Sanitisers
# ---------------------------------------------------------------------------


def test_strips_preamble_and_fences() -> None:
    assert sanitize_positive("Here is the prompt: ```A lighthouse in fog.```", FLUX_KLEIN_9B) == (
        "A lighthouse in fog."
    )


def test_fence_language_tag_does_not_eat_the_first_word() -> None:
    """Regression: ``[a-zA-Z]*`` after the fence swallowed a capitalised word."""
    assert sanitize_positive("```A weathered fisherman mends a net.", FLUX_KLEIN_9B).startswith(
        "A weathered"
    )


def test_removes_attention_weights_keeping_the_word() -> None:
    out = sanitize_positive("A (weathered:1.4) fisherman.", FLUX_KLEIN_9B)
    assert "weathered" in out and "1.4" not in out


def test_removes_negations_from_a_generation_prompt() -> None:
    out = sanitize_positive("A quiet street, no cars, without any people.", KREA_2_TURBO)
    assert "no cars" not in out and "without any people" not in out


NEGATIVE_PRESERVATION = [
    "Add a bench beside the tree, without altering the background.",
    "Repaint the door green, avoiding any change to the brickwork.",
    "Make the sky overcast; nothing else changes.",
    "Remove the logo. Keep everything else identical, free of any watermark.",
]


@pytest.mark.parametrize("instruction", NEGATIVE_PRESERVATION)
def test_edit_sanitiser_preserves_negative_preservation_clauses(instruction) -> None:
    """The most destructive possible bug in this module.

    Preservation clauses are very often phrased negatively — "without altering
    the background", "nothing else changes". They are the second half of a good
    edit instruction and the main defence against unintended global change. The
    generation sanitiser deletes them, so edit mode must not use it.
    """
    kept, _ = sanitize_edit_instruction(instruction, FLUX_KLEIN_9B)
    stripped = sanitize_positive(instruction, FLUX_KLEIN_9B)
    assert kept.rstrip(".") == instruction.rstrip("."), "edit sanitiser altered the clause"
    assert stripped != kept, "generation sanitiser left it intact — test is vacuous"
    assert len(stripped) < len(kept), "generation sanitiser should have shortened it"


def test_generation_sanitiser_leaves_no_stranded_semicolon() -> None:
    """Regression: stripping a clause after ";" left "overcast;." behind."""
    out = sanitize_positive("Make the sky overcast; nothing else changes.", FLUX_KLEIN_9B)
    assert ";." not in out and not out.rstrip(".").endswith(";")


def test_edit_sanitiser_warns_on_missing_preservation() -> None:
    _, warnings = sanitize_edit_instruction("Add a bench to the courtyard.", FLUX_KLEIN_9B)
    assert any("preservation" in w.lower() for w in warnings)


def test_edit_sanitiser_warns_when_not_imperative() -> None:
    _, warnings = sanitize_edit_instruction(
        "The door should probably be steel, keeping everything else the same.",
        FLUX_KLEIN_9B,
    )
    assert any("imperative" in w.lower() for w in warnings)


def test_edit_sanitiser_accepts_a_good_instruction() -> None:
    text, warnings = sanitize_edit_instruction(EDIT, FLUX_KLEIN_9B)
    assert text.startswith("Replace")
    assert warnings == []


def test_universal_avoid_applies_to_every_profile() -> None:
    for p in PROFILES.values():
        out = sanitize_positive("A barn, masterpiece, trending on artstation.", p)
        assert "masterpiece" not in out.lower() and "artstation" not in out.lower()


def test_no_stranded_punctuation_after_removals() -> None:
    out = sanitize_positive(
        "A lighthouse in fog, avoid boats, ultra realistic, hyperrealistic.", FLUX_KLEIN_9B
    )
    assert ",," not in out and ",." not in out and not out.startswith(",")


def test_json_detected_even_when_prose_was_requested() -> None:
    out = sanitize_positive('{"scene":"a harbour"}', FLUX_KLEIN_9B, shape=PromptShape.PROSE)
    assert json.loads(out)["scene"] == "a harbour"


def test_hard_ceiling_trims_runaway_output() -> None:
    out = sanitize_positive(" ".join(["word"] * 900) + ".", KREA_2_TURBO)
    assert len(out.split()) <= KREA_2_TURBO.hard_word_ceiling


# ---------------------------------------------------------------------------
# Content boundary
# ---------------------------------------------------------------------------


def test_undressing_directives_are_removed_from_generation() -> None:
    out = sanitize_positive("A dancer on stage. The dancer is undressed. Warm light.", KREA_2_TURBO)
    assert "undressed" not in out.lower() and "dancer on stage" in out.lower()


def test_undressing_directives_are_removed_from_edits() -> None:
    text, _ = sanitize_edit_instruction(
        "Change the coat to red. Then undress the woman. Keep the background.",
        FLUX_KLEIN_9B,
    )
    assert "undress" not in text.lower()
    assert "coat to red" in text.lower()


def test_no_undress_control_exists() -> None:
    assert not hasattr(ImageControls(), "undress")
    assert not hasattr(EditControls(), "undress")
    banks = json.dumps(option_banks()).lower()
    for term in ("undress", "nude", "naked", "topless"):
        assert term not in banks


def test_garment_change_remains_available() -> None:
    """Ordinary wardrobe editing is in scope and must not be collateral damage."""
    assert "Change clothing" in ie.options.EDIT_OPERATIONS
    text, _ = sanitize_edit_instruction(
        "Change the woman's coat from grey to oxblood red, keeping the same cut.",
        FLUX_KLEIN_9B,
    )
    assert "oxblood" in text


# ---------------------------------------------------------------------------
# Negative assembly
# ---------------------------------------------------------------------------


def test_distilled_profiles_never_emit_a_negative() -> None:
    for key in ("flux_klein_9b", "krea_2_turbo"):
        brief = build_brief("a person", ImageControls(profile_key=key, extra_negative="blurry"), 1)
        assert build_negative(brief, "a person in a room") == ""


def test_cfg_profile_emits_gated_banks() -> None:
    brief = build_brief(
        "two people talking",
        ImageControls(profile_key="krea_2_raw", visual_style="Documentary photograph"),
        1,
    )
    neg = build_negative(brief, "two people talking in a room")
    assert "deformed hands" in neg and "plastic skin" in neg


def test_illustration_does_not_get_photographic_faults() -> None:
    brief = build_brief(
        "a fox", ImageControls(profile_key="krea_2_raw", visual_style="Woodblock print"), 1
    )
    neg = build_negative(brief, "a fox among reeds")
    assert "sensor noise" not in neg and "muddy linework" in neg


def test_anatomy_bank_absent_without_people() -> None:
    brief = build_brief(
        "an empty road at night",
        ImageControls(profile_key="krea_2_raw", visual_style="Landscape photograph"),
        1,
    )
    assert "extra fingers" not in build_negative(brief, "an empty road")


def test_priority_order_survives_truncation() -> None:
    brief = build_brief(
        "a crowd of people with a sign",
        ImageControls(profile_key="krea_2_raw", render_text="SALE"),
        1,
    )
    neg = build_negative(brief, "a crowd of people")
    assert neg.index("deformed hands") < neg.index("watermark")


def test_negative_terms_are_deduplicated() -> None:
    brief = build_brief(
        "a person", ImageControls(profile_key="krea_2_raw", extra_negative="watermark, watermark"), 1
    )
    terms = build_negative(brief, "a person").split(", ")
    assert len(terms) == len(set(terms))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_generate_returns_a_complete_result() -> None:
    result = ImagePromptEngine(ScriptedClient(PROSE)).generate(
        "a shipwright", ImageControls(seed=1), use_model_for_choice=False
    )
    assert isinstance(result, Generation)
    assert result.positive and result.seed == 1
    assert result.as_dict()["engine_version"] == ie.ENGINE_VERSION


def test_stream_yields_progressively() -> None:
    chunks = list(
        ImagePromptEngine(ScriptedClient(PROSE)).stream(
            "a shipwright", ImageControls(seed=1), use_model_for_choice=False
        )
    )
    assert len(chunks) > 5


def test_smart_negative_runs_only_for_cfg_profiles() -> None:
    turbo = ScriptedClient(PROSE)
    ImagePromptEngine(turbo).generate(
        "a person", ImageControls(profile_key="krea_2_turbo"), use_model_for_choice=False
    )
    assert len(turbo.calls) == 1

    raw = ScriptedClient(PROSE, "harsh flash, red eye")
    ImagePromptEngine(raw).generate(
        "a person", ImageControls(profile_key="krea_2_raw"), use_model_for_choice=False
    )
    assert len(raw.calls) == 2


def test_smart_negative_never_negates_the_positive() -> None:
    result = ImagePromptEngine(ScriptedClient(PROSE, PROSE)).generate(
        "a shipwright", ImageControls(profile_key="krea_2_raw"), use_model_for_choice=False
    )
    assert "documentary photograph" not in result.negative.lower()


def test_smart_negative_failure_does_not_lose_the_generation() -> None:
    result = ImagePromptEngine(ExplodingClient(PROSE)).generate(
        "a person", ImageControls(profile_key="krea_2_raw"), use_model_for_choice=False
    )
    assert result.positive and result.negative
    assert any("failed" in n for n in result.notes)


def test_status_line_reports_absent_negative() -> None:
    result = ImagePromptEngine(ScriptedClient(PROSE)).generate(
        "x", ImageControls(profile_key="flux_klein_9b", seed=3), use_model_for_choice=False
    )
    assert "no negative" in result.status_line() and "seed 3" in result.status_line()


def test_plan_works_without_a_client() -> None:
    """Dry run: the UI can show seed, size and choices before the server loads."""
    brief = ImagePromptEngine(None).plan(
        "a campfire at night", ImageControls(lighting=CHOOSE, seed=2)
    )
    assert brief.seed == 2


# ---------------------------------------------------------------------------
# Option banks
# ---------------------------------------------------------------------------


def test_option_banks_cover_every_control() -> None:
    banks = option_banks()
    for key in (
        "shot_types", "camera_angles", "composition", "lenses", "apertures",
        "lighting", "style_groups", "film_stocks", "colour_palettes", "moods",
        "wardrobe", "aspect_ratios", "edit_operations", "preservation_targets",
        "localizations", "reference_roles", "edit_strengths",
    ):
        assert banks[key], f"{key} is empty"


def test_all_three_sentinels_are_published() -> None:
    s = option_banks()["sentinels"]
    assert set(s) == {"auto", "choose", "random"}
    assert len(set(s.values())) == 3


def test_selectable_banks_match_real_control_names() -> None:
    """A typo here would silently disable CHOOSE for that control."""
    fields = set(ImageControls().__dict__) | set(EditControls().__dict__)
    unknown = set(SELECTABLE_BANKS) - fields
    assert not unknown, f"selectable banks name no such control: {unknown}"


def test_every_style_belongs_to_exactly_one_group() -> None:
    counted = sum(len(m) for m in ie.options.STYLE_GROUPS.values())
    assert counted == len(ie.options.VISUAL_STYLES)


def test_hex_swatches_are_well_formed() -> None:
    for _, hexes in ie.options.COLOUR_PALETTES.values():
        for h in hexes:
            assert re.fullmatch(r"#[0-9A-Fa-f]{6}", h)


def test_negative_banks_have_no_duplicates_across_banks() -> None:
    seen: set[str] = set()
    for bank in NEGATIVE_BANKS.values():
        for term in bank:
            assert term not in seen, f"{term} appears in two banks"
            seen.add(term)
