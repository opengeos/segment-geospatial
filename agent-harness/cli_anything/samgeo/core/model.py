"""Model management: list, download, and inspect SAM models."""

import os

MODEL_REGISTRY = {
    "sam": {
        "display_name": "SAM (v1)",
        "description": "Original Segment Anything Model",
        "install": "pip install segment-geospatial",
        "models": {
            "vit_h": {
                "name": "ViT-H (default)",
                "params": "636M",
                "description": "Largest, most accurate",
            },
            "vit_l": {
                "name": "ViT-L",
                "params": "308M",
                "description": "Large model, good balance",
            },
            "vit_b": {
                "name": "ViT-B",
                "params": "91M",
                "description": "Smallest, fastest",
            },
        },
    },
    "sam2": {
        "display_name": "SAM 2",
        "description": "Segment Anything Model 2 with video support",
        "install": "pip install segment-geospatial[samgeo2]",
        "models": {
            "sam2-hiera-tiny": {
                "name": "Hiera-Tiny",
                "params": "38.9M",
                "description": "Smallest, fastest inference",
            },
            "sam2-hiera-small": {
                "name": "Hiera-Small",
                "params": "46M",
                "description": "Small model",
            },
            "sam2-hiera-base-plus": {
                "name": "Hiera-Base+",
                "params": "80.8M",
                "description": "Base model with extras",
            },
            "sam2-hiera-large": {
                "name": "Hiera-Large (default)",
                "params": "224.4M",
                "description": "Largest, most accurate",
            },
        },
    },
    "sam3": {
        "display_name": "SAM 3",
        "description": "Segment Anything Model 3 with text support",
        "install": "pip install segment-geospatial[samgeo3]",
        "models": {
            "facebook/sam3": {
                "name": "SAM3 (default)",
                "params": "~2B",
                "description": "Full SAM3 model with text understanding",
            },
        },
    },
    "fast_sam": {
        "display_name": "FastSAM",
        "description": "Fast Segment Anything Model (YOLO-based)",
        "install": "pip install segment-geospatial[fast]",
        "models": {
            "FastSAM-x.pt": {
                "name": "FastSAM-X (default)",
                "params": "138M",
                "description": "Larger, more accurate",
            },
            "FastSAM-s.pt": {
                "name": "FastSAM-S",
                "params": "11.8M",
                "description": "Smaller, faster",
            },
        },
    },
    "hq_sam": {
        "display_name": "HQ-SAM",
        "description": "High-Quality Segment Anything Model",
        "install": "pip install segment-geospatial[hq]",
        "models": {
            "vit_h": {
                "name": "ViT-H (default)",
                "params": "636M",
                "description": "Largest HQ model",
            },
            "vit_l": {
                "name": "ViT-L",
                "params": "308M",
                "description": "Large HQ model",
            },
            "vit_b": {
                "name": "ViT-B",
                "params": "91M",
                "description": "Base HQ model",
            },
            "vit_tiny": {
                "name": "ViT-Tiny",
                "params": "~10M",
                "description": "Tiny HQ model",
            },
        },
    },
    "text_sam": {
        "display_name": "LangSAM",
        "description": "Language-guided SAM (GroundingDINO + SAM)",
        "install": "pip install segment-geospatial[text]",
        "models": {
            "vit_h": {
                "name": "ViT-H (default)",
                "params": "636M",
                "description": "SAM1 ViT-H backend for mask generation",
            },
            "sam2-hiera-large": {
                "name": "SAM2 Hiera-Large",
                "params": "224.4M",
                "description": "SAM2 backend for mask generation",
            },
        },
    },
}


def list_models():
    """List all available model types and their variants.

    Returns:
        list: List of model info dicts.
    """
    result = []
    for type_key, type_info in MODEL_REGISTRY.items():
        for model_id, model_info in type_info["models"].items():
            result.append(
                {
                    "type": type_key,
                    "type_name": type_info["display_name"],
                    "model_id": model_id,
                    "name": model_info["name"],
                    "params": model_info["params"],
                    "description": model_info["description"],
                    "install": type_info["install"],
                }
            )
    return result


def get_model_info(model_type, model_id=None):
    """Get detailed info about a specific model.

    Args:
        model_type: Model type key (e.g., 'sam2').
        model_id: Model identifier. If None, returns info about the type.

    Returns:
        dict: Model information.

    Raises:
        ValueError: If the model type or ID is not found.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {', '.join(MODEL_REGISTRY.keys())}"
        )

    type_info = MODEL_REGISTRY[model_type]

    if model_id is None:
        return {
            "type": model_type,
            "display_name": type_info["display_name"],
            "description": type_info["description"],
            "install": type_info["install"],
            "models": list(type_info["models"].keys()),
            "model_count": len(type_info["models"]),
        }

    if model_id not in type_info["models"]:
        raise ValueError(
            f"Unknown model ID '{model_id}' for type '{model_type}'. "
            f"Available: {', '.join(type_info['models'].keys())}"
        )

    model_info = type_info["models"][model_id]
    return {
        "type": model_type,
        "type_name": type_info["display_name"],
        "model_id": model_id,
        "name": model_info["name"],
        "params": model_info["params"],
        "description": model_info["description"],
        "install": type_info["install"],
    }


def get_model_types():
    """Get list of available model types.

    Returns:
        list: List of model type keys.
    """
    return list(MODEL_REGISTRY.keys())


def get_default_model_id(model_type):
    """Get the default model ID for a given type.

    Args:
        model_type: Model type key.

    Returns:
        str: Default model ID.
    """
    defaults = {
        "sam": "vit_h",
        "sam2": "sam2-hiera-large",
        "sam3": "facebook/sam3",
        "fast_sam": "FastSAM-x.pt",
        "hq_sam": "vit_h",
        "text_sam": "vit_h",
    }
    return defaults.get(model_type, model_type)


def check_model_available(model_type):
    """Check if the required package for a model type is installed.

    Args:
        model_type: Model type key.

    Returns:
        dict: Availability info with 'available' bool and 'message' str.
    """
    import_map = {
        "sam": "segment_anything",
        "sam2": "sam2",
        "sam3": "sam3",
        "fast_sam": "ultralytics",
        "hq_sam": "segment_anything_hq",
        "text_sam": "groundingdino",
    }

    pkg = import_map.get(model_type)
    if pkg is None:
        return {"available": False, "message": f"Unknown model type: {model_type}"}

    try:
        __import__(pkg)
        return {"available": True, "message": f"{model_type} is available"}
    except ImportError:
        install_cmd = MODEL_REGISTRY.get(model_type, {}).get("install", "")
        return {
            "available": False,
            "message": f"{model_type} not installed. Install with: {install_cmd}",
        }
