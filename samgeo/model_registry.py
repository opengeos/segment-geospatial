"""Shared model identifiers used across samgeo entry points."""

SAM3_MODEL_IDS = ("facebook/sam3", "facebook/sam3.1")

DEFAULT_MODEL_IDS = {
    "sam": "vit_h",
    "sam2": "sam2-hiera-large",
    "sam3": "facebook/sam3",
}

AVAILABLE_MODELS = {
    "sam": ["vit_h", "vit_l", "vit_b"],
    "sam2": [
        "sam2-hiera-tiny",
        "sam2-hiera-small",
        "sam2-hiera-base-plus",
        "sam2-hiera-large",
    ],
    "sam3": list(SAM3_MODEL_IDS),
}

EXTRAS_MAP = {
    "sam": "samgeo",
    "sam2": "samgeo2",
    "sam3": "samgeo3",
}
