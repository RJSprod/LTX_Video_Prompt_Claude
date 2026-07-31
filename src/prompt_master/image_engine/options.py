"""
image_engine.options — every option bank, and the three sentinels.

This module is the single source of truth for every dropdown in Image Prompt
mode. The UI must build its controls from :func:`option_banks` and must never
hold its own copy of a list. That is the same rule ``prompt_engine/options.py``
enforces for LTX mode, and it is the reason no control there can drift from the
engine that consumes its value.

THE THREE SENTINELS
-------------------
Every dropdown carries three values above its real options:

  AUTO    "Leave it to the writer." The facet is omitted from the brief
          entirely. The writing model may still imply something, but nothing
          is stated and nothing is recorded. This is the quietest option.

  CHOOSE  "Choose for me." The engine actively inspects the intent and picks
          the best-fitting option from this bank, then states it explicitly.
          The resolved value is reported back to the UI so it is visible and
          reusable — the point is to see what was chosen, not to guess.

  RANDOM  "Random." A seeded draw from this bank. With a fixed seed the draw
          is reproducible; with seed -1 it is fresh every generation.

AUTO and RANDOM differ in whether anything is stated. CHOOSE and RANDOM differ
in whether the intent is consulted. All three exist because they answer
different questions: "I don't care", "surprise me", and "you decide, but tell
me what you decided".
"""

from __future__ import annotations

__all__ = [
    "AUTO",
    "CHOOSE",
    "RANDOM",
    "SENTINELS",
    "is_set",
    "SHOT_TYPES",
    "CAMERA_ANGLES",
    "COMPOSITION",
    "LENSES",
    "APERTURES",
    "LIGHTING",
    "TIME_OF_DAY",
    "WEATHER",
    "STYLE_GROUPS",
    "VISUAL_STYLES",
    "STYLE_GROUP_OF",
    "FILM_STOCKS",
    "COLOUR_PALETTES",
    "MATERIALS",
    "SKIN_TEXTURE",
    "MOODS",
    "WARDROBE_REGISTERS",
    "AGE_BANDS",
    "BUILDS",
    "FEATURE_HOOKS",
    "ASPECT_RATIOS",
    "RENDER_DETAIL",
    "EDIT_OPERATIONS",
    "PRESERVATION_TARGETS",
    "LOCALIZATIONS",
    "REFERENCE_ROLES",
    "EDIT_STRENGTHS",
    "SELECTABLE_BANKS",
    "SELECTION_CUES",
    "option_banks",
]


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

AUTO = "Auto — leave it to the writer"
CHOOSE = "Choose for me — best fit for my intent"
RANDOM = "Random — a seeded draw"

SENTINELS: tuple[str, ...] = (AUTO, CHOOSE, RANDOM)


def is_set(value: str | None) -> bool:
    """True when a control carries a real choice rather than a sentinel."""
    return bool(value) and value not in SENTINELS


# ---------------------------------------------------------------------------
# Generation banks
# ---------------------------------------------------------------------------

SHOT_TYPES: dict[str, str] = {
    "Extreme close-up": "an extreme close-up filling the frame with a single detail",
    "Close-up": "a close-up framed tightly on the subject",
    "Medium close-up": "a medium close-up from the chest up",
    "Medium shot": "a medium shot from the waist up",
    "Cowboy shot": "a cowboy shot cut at mid-thigh",
    "Full shot": "a full shot with the whole subject inside the frame",
    "Wide shot": "a wide shot placing the subject in its surroundings",
    "Extreme wide shot": "an extreme wide shot where the subject is small in a large space",
    "Two shot": "a two shot holding both figures in frame",
    "Over-the-shoulder": "an over-the-shoulder framing past a foreground figure",
    "Insert": "a tight insert on an object",
    "Establishing shot": "an establishing shot that reads the whole location",
}

CAMERA_ANGLES: dict[str, str] = {
    "Eye level": "shot at eye level",
    "Low angle": "shot from a low angle looking up",
    "High angle": "shot from a high angle looking down",
    "Worm's eye": "shot from ground level looking steeply up",
    "Bird's eye": "shot from directly overhead",
    "Dutch angle": "shot with the horizon tilted",
    "Three-quarter": "shot from three-quarters on",
    "Profile": "shot square to the subject in profile",
    "Behind": "shot from behind the subject",
    "Top-down flat lay": "shot straight down onto a flat surface",
}

COMPOSITION: dict[str, str] = {
    "Rule of thirds": "the subject placed on a third rather than centred",
    "Centred symmetry": "a deliberately centred, symmetrical composition",
    "Off-centre with negative space": "the subject well off centre with open negative space beside it",
    "Leading lines": "strong leading lines drawing the eye to the subject",
    "Framed within a frame": "the subject framed by an opening in the foreground",
    "Diagonal": "a composition built on a strong diagonal",
    "Foreground layering": "an out-of-focus foreground element layered in front",
    "Golden spiral": "a composition curving along a golden spiral",
    "Tight crop": "a tight crop that cuts the subject at the frame edge",
    "Horizon low": "a low horizon giving most of the frame to the sky",
    "Horizon high": "a high horizon giving most of the frame to the ground",
}

LENSES: dict[str, str] = {
    "14mm ultra-wide": "shot on a 14mm ultra-wide lens with pronounced perspective",
    "24mm wide": "shot on a 24mm wide lens",
    "35mm reportage": "shot on a 35mm lens with natural reportage perspective",
    "50mm normal": "shot on a 50mm lens with perspective close to the eye",
    "85mm portrait": "shot on an 85mm portrait lens with gentle compression",
    "135mm telephoto": "shot on a 135mm telephoto with compressed depth",
    "200mm long telephoto": "shot on a 200mm telephoto flattening the layers",
    "100mm macro": "shot on a 100mm macro lens at close focus",
    "Tilt-shift": "shot on a tilt-shift lens with a narrow slanted plane of focus",
    "Anamorphic": "shot on an anamorphic lens with horizontal flare and oval bokeh",
}

APERTURES: dict[str, str] = {
    "f/1.4 — subject isolated": "at f/1.4, the background dissolved",
    "f/2 — shallow": "at f/2 with a shallow plane of focus",
    "f/2.8 — soft background": "at f/2.8, the background soft but readable",
    "f/4 — moderate": "at f/4 with moderate depth of field",
    "f/5.6 — balanced": "at f/5.6, subject and setting both legible",
    "f/8 — deep": "at f/8 with deep focus front to back",
    "f/11 — landscape": "at f/11, everything sharp from foreground to horizon",
    "f/16 — everything sharp": "at f/16 with the whole frame in focus",
}

LIGHTING: dict[str, str] = {
    "Soft window light": "lit by soft daylight through a large window",
    "Hard sunlight": "lit by hard direct sun with crisp shadow edges",
    "Golden hour": "lit by low golden-hour sun raking across the scene",
    "Blue hour": "lit by the cool residual light after sunset",
    "Overcast": "lit by flat overcast daylight with open shadows",
    "Three-point studio": "lit with a three-point studio setup, key, fill and rim",
    "Rembrandt": "lit in Rembrandt style with a small triangle of light on the shadowed cheek",
    "Rim light": "lit from behind so the subject is edged in light",
    "Backlit silhouette": "backlit until the subject reads as a silhouette",
    "Softbox beauty": "lit by a large softbox close to the lens, shadows nearly closed",
    "Chiaroscuro": "lit in deep chiaroscuro, a single source and heavy falloff",
    "Practical sources": "lit only by practical lights inside the scene",
    "Neon": "lit by coloured neon signage",
    "Firelight": "lit by low warm firelight",
    "Candlelight": "lit by candles, the falloff very fast",
    "Moonlight": "lit by cool moonlight",
    "Fluorescent overhead": "lit by flat overhead fluorescent tubes",
    "Mixed colour temperature": "lit by two sources of different colour temperature",
    "Volumetric shafts": "lit by visible shafts of light through haze",
    "Bounced fill": "lit by light bounced off a nearby pale surface",
}

TIME_OF_DAY: dict[str, str] = {
    "Dawn": "at dawn",
    "Early morning": "in the early morning",
    "Midday": "at midday",
    "Afternoon": "in the afternoon",
    "Golden hour": "during golden hour",
    "Dusk": "at dusk",
    "Night": "at night",
    "Pre-dawn dark": "in the dark before dawn",
}

WEATHER: dict[str, str] = {
    "Clear": "under a clear sky",
    "Overcast": "under heavy overcast",
    "Light rain": "in light rain, surfaces wet and reflective",
    "Downpour": "in a downpour",
    "Fog": "in thick fog that swallows the background",
    "Light haze": "in light haze that separates the distance into planes",
    "Snow": "in falling snow",
    "Dust": "in blowing dust",
    "After rain": "just after rain, everything still wet",
    "Humid heat": "in humid heat with a soft atmospheric shimmer",
}

STYLE_GROUPS: dict[str, dict[str, str]] = {
    "Photographic": {
        "Documentary photograph": "a documentary photograph, unposed and available-light",
        "Editorial portrait": "an editorial portrait, deliberately lit and composed",
        "Fashion editorial": "a fashion editorial photograph",
        "Street photography": "a street photograph caught in passing",
        "Fine-art photograph": "a fine-art photograph with careful tonal control",
        "Product photograph": "a clean product photograph on a controlled background",
        "Architectural photograph": "an architectural photograph with corrected verticals",
        "Landscape photograph": "a landscape photograph with deep detail",
        "Photojournalism": "a photojournalistic frame, plain and unstyled",
        "Analogue film still": "a still frame from analogue motion-picture film",
    },
    "Illustration": {
        "Ink line drawing": "an ink line drawing with visible pen weight",
        "Watercolour": "a watercolour with soft bleeding edges and paper grain",
        "Gouache": "a gouache painting with flat opaque colour",
        "Comic book": "a comic-book panel with bold inked outlines",
        "Manga": "a manga panel in screentone",
        "Children's book": "a children's-book illustration with warm simple shapes",
        "Technical drawing": "a precise technical drawing with clean linework",
        "Woodblock print": "a woodblock print with visible grain and limited colour",
        "Vintage poster": "a mid-century printed poster with limited inks",
        "Storyboard sketch": "a loose storyboard sketch in graphite",
    },
    "Painterly": {
        "Oil painting": "an oil painting with visible brushwork and impasto",
        "Alla prima": "an alla prima oil sketch, wet into wet",
        "Impressionist": "an impressionist painting of broken colour",
        "Tonalist": "a tonalist painting in a narrow value range",
        "Baroque": "a baroque painting with dramatic light and deep shadow",
        "Ukiyo-e": "an ukiyo-e print with flat colour and strong outline",
        "Charcoal": "a charcoal drawing, smudged and rubbed back",
        "Pastel": "a soft pastel drawing on toned paper",
    },
    "Rendered": {
        "Physically based render": "a physically based 3D render with accurate materials",
        "Clay render": "an untextured clay render lit by a single soft source",
        "Isometric render": "an isometric 3D render",
        "Architectural visualisation": "an architectural visualisation with realistic glazing and daylight",
        "Stop-motion miniature": "a photographed stop-motion miniature with real materials",
        "Low-poly": "a low-polygon render with faceted surfaces",
        "Voxel": "a voxel render built from cubic units",
        "Product visualisation": "a studio product visualisation with controlled reflections",
    },
}

VISUAL_STYLES: dict[str, str] = {
    name: phrase for group in STYLE_GROUPS.values() for name, phrase in group.items()
}

STYLE_GROUP_OF: dict[str, str] = {
    name: group for group, members in STYLE_GROUPS.items() for name in members
}

FILM_STOCKS: dict[str, str] = {
    "Kodak Portra 400": "the palette of Kodak Portra 400, warm skin and restrained saturation",
    "Kodak Gold 200": "the palette of Kodak Gold 200, warm and slightly nostalgic",
    "Kodak Tri-X 400": "black and white on Kodak Tri-X with visible grain",
    "Ilford HP5": "black and white on Ilford HP5, open shadows",
    "Fujifilm Pro 400H": "the palette of Fuji Pro 400H, cool greens and soft highlights",
    "Fujifilm Velvia 50": "the palette of Velvia 50, dense saturated colour",
    "Cinestill 800T": "the palette of Cinestill 800T, tungsten balance and haloed highlights",
    "Kodak Vision3 500T": "the palette of Vision3 500T motion-picture stock",
    "Polaroid 600": "the look of a Polaroid 600 print, low contrast and shifted colour",
    "Technicolor three-strip": "the look of three-strip Technicolor, saturated primaries",
}

COLOUR_PALETTES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Muted earth": ("a muted earth palette of clay, ochre and dust", ("#8C6A4A", "#C2A878", "#5E4B3C")),
    "Cool monochrome": ("a cool near-monochrome palette of slate and pale blue", ("#2E3A45", "#7C8B99", "#C9D4DC")),
    "Warm amber": ("a warm amber palette", ("#B5651D", "#E0A458", "#3A2412")),
    "Teal and orange": ("a teal-and-orange split palette", ("#1F5C63", "#D97B34", "#0E2C30")),
    "Pastel": ("a pale pastel palette", ("#F2D5D5", "#D8E2DC", "#EED9C4")),
    "High-contrast black and white": ("black and white with deep blacks and clean whites", ("#000000", "#7A7A7A", "#FFFFFF")),
    "Desaturated green": ("a desaturated green palette, almost olive", ("#4A5240", "#8A9179", "#2A2E24")),
    "Neon night": ("a neon night palette of magenta and cyan against black", ("#FF2D95", "#00E5FF", "#0A0A12")),
    "Sun-bleached": ("a sun-bleached palette, colour washed out of everything", ("#E8E0D0", "#C4B9A5", "#9A8F7C")),
    "Deep jewel": ("a deep jewel palette of oxblood, forest and indigo", ("#5C1A1B", "#1E3D2F", "#232B54")),
    "Sepia": ("a sepia palette", ("#704214", "#A67B5B", "#2B1A0E")),
    "Cold industrial": ("a cold industrial palette of concrete and steel", ("#6E7379", "#A8ADB2", "#3B3F44")),
}

MATERIALS: dict[str, str] = {
    "Brushed metal": "brushed metal with fine directional grain",
    "Raw concrete": "raw concrete with form-tie marks",
    "Aged leather": "aged leather, creased and slightly glossed",
    "Rough linen": "rough linen with a visible weave",
    "Weathered timber": "weathered timber with raised grain",
    "Wet stone": "wet stone with specular highlights",
    "Frosted glass": "frosted glass diffusing what is behind it",
    "Polished marble": "polished marble with deep veining",
    "Oxidised copper": "oxidised copper with green patina",
    "Worn denim": "worn denim, faded at the seams",
    "Cracked paint": "cracked and flaking paint",
    "Matte ceramic": "matte unglazed ceramic",
}

SKIN_TEXTURE: tuple[str, ...] = (
    "visible skin pores and fine lines",
    "natural skin texture with faint blemishes",
    "freckles across the nose and cheeks",
    "slight shine on the forehead and nose",
    "fine vellus hair catching the light",
    "sun-weathered skin with uneven tone",
)

MOODS: dict[str, str] = {
    "Calm": "a calm, unhurried mood",
    "Tense": "a tense, coiled mood",
    "Melancholy": "a melancholy mood",
    "Joyful": "an open, joyful mood",
    "Solemn": "a solemn, formal mood",
    "Intimate": "a quiet, intimate mood",
    "Ominous": "an ominous mood",
    "Nostalgic": "a nostalgic mood",
    "Austere": "an austere, stripped-back mood",
    "Playful": "a playful mood",
    "Weary": "a weary, worn-down mood",
    "Triumphant": "a triumphant mood",
}

WARDROBE_REGISTERS: dict[str, str] = {
    "Everyday casual": "everyday casual clothes, worn and unremarkable",
    "Workwear": "practical workwear with visible use",
    "Tailored formal": "well-cut tailored formalwear",
    "Uniform": "a service uniform, correctly worn",
    "Outdoor technical": "technical outdoor clothing",
    "Period 1920s": "1920s dress, correct in cut and fabric",
    "Period 1950s": "1950s dress, correct in cut and fabric",
    "Period 1970s": "1970s dress, correct in cut and fabric",
    "Period 1990s": "1990s dress, correct in cut and fabric",
    "Traditional regional": "traditional regional dress, respectfully rendered",
    "Athletic": "athletic clothing suited to the activity",
    "Winter layers": "heavy winter layers",
    "Stage costume": "theatrical stage costume",
    "Formal robes": "formal ceremonial robes",
}

AGE_BANDS: tuple[str, ...] = (
    "in their twenties",
    "in their thirties",
    "in their forties",
    "in their fifties",
    "in their sixties",
    "in their seventies",
)

BUILDS: tuple[str, ...] = (
    "slight",
    "lean",
    "average build",
    "broad-shouldered",
    "heavyset",
    "stocky",
    "tall and narrow",
)

FEATURE_HOOKS: tuple[str, ...] = (
    "a broken nose that set slightly crooked",
    "deep-set eyes",
    "a wide jaw",
    "heavy brows",
    "a long face with sharp cheekbones",
    "a round face with soft features",
    "a scar through one eyebrow",
    "close-cropped grey hair",
    "long hair tied back",
    "a shaved head",
    "wire-framed glasses",
    "a gap between the front teeth",
)

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1 square": (1, 1),
    "4:5 portrait": (4, 5),
    "2:3 portrait": (2, 3),
    "3:4 portrait": (3, 4),
    "9:16 tall": (9, 16),
    "4:3 landscape": (4, 3),
    "3:2 landscape": (3, 2),
    "16:9 widescreen": (16, 9),
    "21:9 anamorphic": (21, 9),
}

RENDER_DETAIL: dict[str, str] = {
    "Restrained": "keep detail restrained; describe only what carries the image",
    "Balanced": "describe detail at a natural level",
    "Dense": "describe surface detail densely, material by material",
}


# ---------------------------------------------------------------------------
# Edit-mode banks
#
# Operation names map to the verb-and-shape guidance the writer needs. These
# are the operation types documented by Black Forest Labs or verified in
# community use for FLUX.2 instruction editing. Masked inpainting is absent
# because neither supported model does masks.
# ---------------------------------------------------------------------------

EDIT_OPERATIONS: dict[str, str] = {
    "Change an attribute": (
        "change one property of an existing element — colour, material, age, "
        "condition — leaving the element itself in place"
    ),
    "Replace an object": (
        "swap one element for a different one occupying the same space and "
        "catching the same light"
    ),
    "Add an object": (
        "introduce a new element that was not there, sitting plausibly in the "
        "existing perspective and lighting"
    ),
    "Remove an object": (
        "take an element out and reconstruct what was behind it, so the gap "
        "reads as though the element was never there"
    ),
    "Replace the background": (
        "change the setting behind the subject while the subject itself stays "
        "exactly as it is"
    ),
    "Change the style": (
        "restate the whole image in a different medium or treatment while the "
        "content, layout and subject identity survive"
    ),
    "Relight the scene": (
        "change the light source, its direction or its quality, and let the "
        "shadows and highlights follow it"
    ),
    "Change time of day": (
        "move the scene to a different hour, with the light, shadow length and "
        "colour temperature all consistent with it"
    ),
    "Change the season": (
        "move the scene to a different season, changing foliage, ground cover "
        "and light to match"
    ),
    "Change the weather": (
        "change the weather, with surfaces, visibility and light all responding"
    ),
    "Change clothing": (
        "change a garment for a different one that fits the figure's pose and "
        "the scene's period"
    ),
    "Change expression": (
        "change the facial expression while the same face, angle and lighting "
        "are kept"
    ),
    "Change pose": (
        "change how the figure is standing or holding themselves, keeping the "
        "same person and the same setting"
    ),
    "Change age": (
        "make the person appear older or younger while remaining recognisably "
        "the same person"
    ),
    "Edit text in the image": (
        "change lettering that appears in the scene, keeping the same surface, "
        "typeface weight and perspective"
    ),
    "Place a product": (
        "insert a product into the scene at a plausible scale, with contact "
        "shadows and reflections that match the surface it sits on"
    ),
    "Transfer a subject": (
        "take the subject from one reference and place it into the scene from "
        "another, matching scale, perspective and light"
    ),
    "Colourise or restore": (
        "add colour to a monochrome image, or repair damage, while the original "
        "grain, composition and period character survive"
    ),
    "Extend the frame": (
        "continue the scene beyond its current edges, matching perspective, "
        "lighting and content"
    ),
    "Enhance detail": (
        "sharpen and enrich surface detail without changing content, layout or "
        "colour"
    ),
}

PRESERVATION_TARGETS: dict[str, str] = {
    "Identity": "the same face, the same features and the same expression",
    "Pose": "the same pose and body position",
    "Clothing": "the same clothing, in the same cut and colour",
    "Background": "the background exactly as it is",
    "Composition": "the same framing, camera angle and subject placement",
    "Lighting": "the same light direction, quality and shadow placement",
    "Colour grading": "the same colour grading and overall palette",
    "Style": "the same medium and rendering style",
    "Texture and grain": "the same surface texture and film grain",
    "Text": "any lettering already in the image, unchanged",
    "Scale and perspective": "the same scale relationships and perspective",
}

LOCALIZATIONS: dict[str, str] = {
    "Left of frame": "on the left of the frame",
    "Right of frame": "on the right of the frame",
    "Centre": "in the centre of the frame",
    "Foreground": "in the foreground",
    "Middle distance": "in the middle distance",
    "Background": "in the background",
    "Upper half": "in the upper half of the frame",
    "Lower half": "in the lower half of the frame",
    "Top-left": "in the top-left of the frame",
    "Top-right": "in the top-right of the frame",
    "Bottom-left": "in the bottom-left of the frame",
    "Bottom-right": "in the bottom-right of the frame",
    "Whole image": "across the whole image",
}

REFERENCE_ROLES: dict[str, str] = {
    "Subject": "the subject",
    "Face": "the face",
    "Style": "the style",
    "Background": "the background",
    "Pose": "the pose",
    "Clothing": "the clothing",
    "Product": "the product",
    "Colour palette": "the colour palette",
    "Composition": "the composition",
    "Lighting": "the lighting",
}

EDIT_STRENGTHS: dict[str, str] = {
    "Preserve — smallest change that works": "preserve",
    "Balanced": "balanced",
    "Transform — commit to the change": "transform",
}


# ---------------------------------------------------------------------------
# Choose-for-me support
# ---------------------------------------------------------------------------

SELECTABLE_BANKS: dict[str, dict[str, str]] = {
    "shot_type": SHOT_TYPES,
    "camera_angle": CAMERA_ANGLES,
    "composition": COMPOSITION,
    "lens": LENSES,
    "aperture": APERTURES,
    "lighting": LIGHTING,
    "time_of_day": TIME_OF_DAY,
    "weather": WEATHER,
    "visual_style": VISUAL_STYLES,
    "film_stock": FILM_STOCKS,
    "colour_palette": {k: v[0] for k, v in COLOUR_PALETTES.items()},
    "material_focus": MATERIALS,
    "mood": MOODS,
    "wardrobe": WARDROBE_REGISTERS,
    "render_detail": RENDER_DETAIL,
    "aspect_ratio": {k: k for k in ASPECT_RATIOS},
    # edit mode
    "operation": EDIT_OPERATIONS,
    "localization": LOCALIZATIONS,
    "edit_strength": {k: k for k in EDIT_STRENGTHS},
}
"""Which controls CHOOSE and RANDOM can resolve, and from what.

Keys are :class:`~image_engine.brief.ImageControls` and
:class:`~image_engine.edit.EditControls` field names, so the selection pass can
write straight back onto the controls object.
"""

SELECTION_CUES: dict[str, tuple[str, ...]] = {
    # Extra trigger words for the deterministic fallback, beyond what can be
    # mined from an option's own name and description. Only options whose cues
    # are not obvious from their text need an entry.
    "Extreme close-up": ("detail", "texture", "macro", "tiny", "grain", "eye"),
    "Wide shot": ("landscape", "vista", "panorama", "expanse", "valley", "skyline"),
    "Extreme wide shot": ("vast", "tiny figure", "horizon", "desert", "wilderness"),
    "Two shot": ("conversation", "couple", "pair", "talking", "facing"),
    "Establishing shot": ("city", "town", "building", "arrival", "location"),
    "Low angle": ("heroic", "towering", "monument", "imposing", "power"),
    "Bird's eye": ("overhead", "aerial", "drone", "map", "rooftop"),
    "Top-down flat lay": ("ingredients", "tools", "desk", "arranged", "objects"),
    "Golden hour": ("sunset", "sunrise", "warm", "evening", "dusk", "glow"),
    "Blue hour": ("twilight", "after sunset", "cool", "dim"),
    "Chiaroscuro": ("dramatic", "shadow", "dark", "candle", "baroque", "mystery"),
    "Neon": ("cyberpunk", "night city", "sign", "bar", "arcade", "rain"),
    "Firelight": ("campfire", "hearth", "forge", "bonfire", "flame"),
    "Moonlight": ("night", "moon", "nocturne", "dark sky"),
    "Fluorescent overhead": ("office", "hospital", "corridor", "supermarket", "clinic"),
    "Volumetric shafts": ("forest", "cathedral", "dust", "window light", "beams"),
    "Soft window light": ("interior", "kitchen", "quiet", "morning", "reading"),
    "Hard sunlight": ("noon", "desert", "harsh", "summer", "beach"),
    "100mm macro": ("insect", "flower", "close", "detail", "tiny", "droplet"),
    "14mm ultra-wide": ("interior", "cramped", "sweeping", "architecture"),
    "85mm portrait": ("portrait", "face", "headshot", "person"),
    "200mm long telephoto": ("wildlife", "distant", "sport", "bird", "compressed"),
    "Tilt-shift": ("miniature", "toy", "model", "cityscape"),
    "Anamorphic": ("cinematic", "film", "widescreen", "flare"),
    "f/1.4 — subject isolated": ("bokeh", "dreamy", "isolated", "blurred background"),
    "f/11 — landscape": ("landscape", "sharp", "vista", "sweeping"),
    "f/16 — everything sharp": ("architecture", "detail", "everything sharp"),
    "Documentary photograph": ("real", "candid", "unposed", "reportage", "journal"),
    "Editorial portrait": ("portrait", "profile", "magazine", "posed"),
    "Product photograph": ("product", "packshot", "commercial", "bottle", "shoe"),
    "Architectural photograph": ("building", "facade", "interior", "architecture"),
    "Landscape photograph": ("mountain", "valley", "coast", "field", "wilderness"),
    "Street photography": ("street", "city", "passerby", "market", "crowd"),
    "Analogue film still": ("cinematic", "movie", "film still", "scene"),
    "Ink line drawing": ("line", "sketch", "pen", "drawn", "illustration"),
    "Watercolour": ("soft", "painted", "bleeding", "delicate", "storybook"),
    "Comic book": ("comic", "superhero", "action", "bold ink", "speech bubble"),
    "Manga": ("anime", "manga", "japanese comic"),
    "Children's book": ("children", "storybook", "friendly", "whimsical", "cute"),
    "Technical drawing": ("blueprint", "schematic", "diagram", "exploded", "patent"),
    "Woodblock print": ("japanese", "ukiyo", "print", "traditional", "wave"),
    "Vintage poster": ("poster", "travel", "retro", "advert", "mid-century"),
    "Oil painting": ("painted", "classical", "gallery", "canvas", "brush"),
    "Impressionist": ("monet", "loose", "dappled", "garden", "broken colour"),
    "Baroque": ("dramatic", "classical", "renaissance", "chiaroscuro"),
    "Charcoal": ("sketch", "monochrome", "smudged", "study"),
    "Physically based render": ("3d", "render", "cgi", "raytraced"),
    "Isometric render": ("isometric", "game", "diorama", "tiny world"),
    "Clay render": ("clay", "untextured", "grey", "sculpt", "maquette"),
    "Architectural visualisation": ("archviz", "interior design", "property"),
    "Stop-motion miniature": ("stop motion", "puppet", "miniature", "handmade"),
    "Low-poly": ("low poly", "faceted", "game", "polygon"),
    "Voxel": ("voxel", "minecraft", "blocks", "cubes"),
    "Kodak Portra 400": ("portrait", "skin", "wedding", "natural"),
    "Kodak Tri-X 400": ("black and white", "monochrome", "grain", "documentary"),
    "Ilford HP5": ("black and white", "monochrome", "street"),
    "Cinestill 800T": ("night", "neon", "tungsten", "city", "halation"),
    "Fujifilm Velvia 50": ("saturated", "landscape", "vivid", "slide"),
    "Polaroid 600": ("instant", "snapshot", "vintage", "faded"),
    "Technicolor three-strip": ("classic hollywood", "musical", "saturated"),
    "Neon night": ("cyberpunk", "night", "neon", "city", "arcade"),
    "Sun-bleached": ("desert", "summer", "faded", "coastal", "arid"),
    "Deep jewel": ("rich", "luxurious", "velvet", "opulent"),
    "Cold industrial": ("factory", "warehouse", "concrete", "machinery"),
    "Sepia": ("old", "antique", "historic", "vintage", "archive"),
    "Muted earth": ("rustic", "rural", "clay", "natural", "farm"),
    "Weathered timber": ("boat", "barn", "dock", "cabin", "workshop"),
    "Raw concrete": ("brutalist", "car park", "bunker", "modernist"),
    "Aged leather": ("saddle", "jacket", "armchair", "bookbinding"),
    "Oxidised copper": ("roof", "statue", "verdigris", "patina"),
    "Ominous": ("threat", "danger", "storm", "dread", "menace"),
    "Melancholy": ("sad", "lonely", "rain", "loss", "empty"),
    "Nostalgic": ("memory", "childhood", "old", "remember", "past"),
    "Triumphant": ("victory", "win", "summit", "celebration"),
    "Weary": ("tired", "exhausted", "long day", "worn"),
    "Workwear": ("welder", "mechanic", "farmer", "builder", "shop floor"),
    "Uniform": ("soldier", "nurse", "police", "officer", "service"),
    "Outdoor technical": ("hiking", "climbing", "mountain", "expedition"),
    "Athletic": ("running", "sport", "gym", "training", "match"),
    "Winter layers": ("snow", "cold", "winter", "frost"),
    "Traditional regional": ("festival", "ceremony", "heritage", "folk"),
    # edit operations
    "Remove an object": ("remove", "delete", "get rid", "take out", "erase", "without"),
    "Add an object": ("add", "put", "insert", "include", "place a"),
    "Replace an object": ("replace", "swap", "change the", "instead of", "turn the"),
    "Change an attribute": ("recolour", "recolor", "repaint", "colour the", "color the",
                            "rusty", "brand new", "worn", "shinier", "duller"),
    "Replace the background": ("background", "behind", "setting", "backdrop"),
    "Change the style": ("style", "as a painting", "make it look", "turn into",
                         "turn it into", "restyle", "in the style of", "as a sketch",
                         "oil painting", "watercolour", "watercolor"),
    "Relight the scene": ("light", "lighting", "brighter", "darker", "shadow"),
    "Change time of day": ("sunset", "sunrise", "night", "nighttime", "morning",
                           "daytime", "dusk", "dawn", "noon", "midday", "evening"),
    "Change the season": ("winter", "summer", "autumn", "spring", "snow", "leaves"),
    "Change the weather": ("rain", "fog", "storm", "sunny", "cloudy", "snowing"),
    "Change clothing": ("coat", "jacket", "shirt", "dress", "wearing", "outfit",
                        "trousers", "sweater", "hat", "scarf", "uniform", "suit",
                        "put her in", "put him in", "put them in", "dress her in",
                        "dress him in", "wearing a", "change into"),
    "Change expression": ("smile", "smiling", "frown", "expression", "laughing"),
    "Change pose": ("pose", "standing", "sitting", "arms", "turn to face"),
    "Change age": ("older", "younger", "age", "elderly", "child"),
    "Edit text in the image": ("text", "sign", "words", "lettering", "caption", "says"),
    "Place a product": ("product", "bottle", "can", "package", "brand"),
    "Transfer a subject": ("from image", "second image", "combine", "composite",
                           "into the other", "same person as"),
    "Colourise or restore": ("colourise", "colorize", "restore", "repair", "old photo"),
    "Extend the frame": ("extend", "wider", "outpaint", "zoom out", "more of"),
    "Enhance detail": ("sharpen", "enhance", "upscale", "crisper", "detail"),
}


def option_banks() -> dict[str, object]:
    """Everything the UI needs to build its controls, in one call.

    Build every dropdown from this. A control that carries its own list will
    drift from the engine that consumes its value.
    """
    return {
        "sentinels": {"auto": AUTO, "choose": CHOOSE, "random": RANDOM},
        "shot_types": list(SHOT_TYPES),
        "camera_angles": list(CAMERA_ANGLES),
        "composition": list(COMPOSITION),
        "lenses": list(LENSES),
        "apertures": list(APERTURES),
        "lighting": list(LIGHTING),
        "time_of_day": list(TIME_OF_DAY),
        "weather": list(WEATHER),
        "style_groups": {g: list(m) for g, m in STYLE_GROUPS.items()},
        "film_stocks": list(FILM_STOCKS),
        "colour_palettes": list(COLOUR_PALETTES),
        "materials": list(MATERIALS),
        "moods": list(MOODS),
        "wardrobe": list(WARDROBE_REGISTERS),
        "aspect_ratios": list(ASPECT_RATIOS),
        "render_detail": list(RENDER_DETAIL),
        "edit_operations": list(EDIT_OPERATIONS),
        "preservation_targets": list(PRESERVATION_TARGETS),
        "localizations": list(LOCALIZATIONS),
        "reference_roles": list(REFERENCE_ROLES),
        "edit_strengths": list(EDIT_STRENGTHS),
        "selectable": list(SELECTABLE_BANKS),
    }
