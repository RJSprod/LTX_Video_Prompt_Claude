"""
image_engine — Image Prompt mode for Prompt Master.

A standalone prompt engine that turns an intent (or a terse edit request) into
a finished prompt for a text-to-image model, written by the same local Gemma
model that already runs behind LTX Prompt mode and Conversation mode.

DESIGN CONTRACT
---------------
This package imports NOTHING from ``prompt_master.prompt_engine``. It shares no
state, no constants and no code path with the vendored LTX video engine. That
isolation is deliberate and load-bearing: the video engine is byte-identical to
upstream and is verified as such by ``tools/compare_upstream_engine.py``. An
import here would put this package inside the parity surface. Do not add one.

It also imports nothing from ``prompt_master.inference``. The streaming client
is injected, so the whole package is importable and testable with no Qt, no
llama-server and no network. Standard library only.

TWO TASKS
---------
generate  Text-to-image. An intent becomes a descriptive prompt.
edit      Instruction editing. A terse request becomes a full edit instruction
          with the target pinned down and the preservation contract stated.

THREE SENTINELS
---------------
Every dropdown offers Auto (omit it), Choose for me (engine picks the best fit
for the intent and reports what it picked) and Random (a seeded draw).

SAFETY NOTE
-----------
The video engine ships an "undress" control. It is deliberately not ported.
Wardrobe, costume and garment-change controls are present in a plain
descriptive register; there is no clothing-removal control and no
sexualisation path, and the sanitisers strip such directives if one arrives
through a free-text field.
"""

from .brief import (
    ImageBrief,
    ImageControls,
    build_brief,
    mentions_people,
    resolve_dimensions,
    resolve_sentinels,
)
from .edit import (
    EditBrief,
    EditControls,
    ReferenceImage,
    build_edit_brief,
    resolve_edit_sentinels,
)
from .engine import (
    ENGINE_VERSION,
    ChatClient,
    EditGeneration,
    Generation,
    ImagePromptEngine,
)
from .negative import NEGATIVE_BANKS, build_negative, gates_for
from .options import (
    AUTO,
    CHOOSE,
    RANDOM,
    SELECTABLE_BANKS,
    SENTINELS,
    is_set,
    option_banks,
)
from .profiles import (
    DEFAULT_PROFILE_KEY,
    FLUX_KLEIN_9B,
    KREA_2_RAW,
    KREA_2_TURBO,
    PROFILES,
    EditCapability,
    ModelProfile,
    NegativeMode,
    PromptShape,
    editing_profiles,
    profile_for,
)
from .sanitize import (
    UNIVERSAL_AVOID,
    sanitize_edit_instruction,
    sanitize_positive,
    word_count,
)
from .selection import (
    NO_PREFERENCE,
    SelectionOutcome,
    SelectionPass,
    heuristic_pick,
    random_pick,
    seeded_rng_for,
)
from .system import (
    build_edit_system,
    build_edit_user_turn,
    build_system,
    build_user_turn,
)

__version__ = ENGINE_VERSION

__all__ = [
    "ENGINE_VERSION",
    "__version__",
    # sentinels and banks
    "AUTO",
    "CHOOSE",
    "RANDOM",
    "SENTINELS",
    "SELECTABLE_BANKS",
    "is_set",
    "option_banks",
    # profiles
    "ModelProfile",
    "NegativeMode",
    "PromptShape",
    "EditCapability",
    "FLUX_KLEIN_9B",
    "KREA_2_TURBO",
    "KREA_2_RAW",
    "PROFILES",
    "DEFAULT_PROFILE_KEY",
    "profile_for",
    "editing_profiles",
    # generation
    "ImageControls",
    "ImageBrief",
    "resolve_sentinels",
    "build_brief",
    "resolve_dimensions",
    "mentions_people",
    "build_system",
    "build_user_turn",
    # editing
    "EditControls",
    "EditBrief",
    "ReferenceImage",
    "resolve_edit_sentinels",
    "build_edit_brief",
    "build_edit_system",
    "build_edit_user_turn",
    # selection
    "SelectionPass",
    "SelectionOutcome",
    "NO_PREFERENCE",
    "heuristic_pick",
    "random_pick",
    "seeded_rng_for",
    # post-processing
    "sanitize_positive",
    "sanitize_edit_instruction",
    "word_count",
    "UNIVERSAL_AVOID",
    "NEGATIVE_BANKS",
    "gates_for",
    "build_negative",
    # orchestration
    "ChatClient",
    "Generation",
    "EditGeneration",
    "ImagePromptEngine",
]
