"""Project management: create, open, save, and inspect segmentation projects.

A project is a JSON file that tracks the source image, model configuration,
generated masks, vector outputs, and operation history.
"""

import json
import os
import time
from datetime import datetime, timezone


def create_project(
    name,
    source_path=None,
    model_type="sam2",
    model_id=None,
    device=None,
    output_path=None,
):
    """Create a new segmentation project.

    Args:
        name: Project name.
        source_path: Path to the source image (GeoTIFF, PNG, etc.).
        model_type: SAM model type ('sam', 'sam2', 'sam3', etc.).
        model_id: Model identifier.
        device: Compute device.
        output_path: Path to save the project JSON. If None, not saved to disk.

    Returns:
        dict: The project state dictionary.
    """
    now = datetime.now(timezone.utc).isoformat()
    default_ids = {
        "sam": "vit_h",
        "sam2": "sam2-hiera-large",
        "sam3": "facebook/sam3",
        "fast_sam": "FastSAM-x.pt",
        "hq_sam": "vit_h",
        "text_sam": "vit_h",
    }

    project = {
        "name": name,
        "version": "1.0",
        "created": now,
        "modified": now,
        "model": {
            "type": model_type,
            "id": model_id or default_ids.get(model_type, model_type),
            "device": device,
        },
        "source": None,
        "masks": None,
        "vectors": None,
        "parameters": {
            "points_per_side": 32,
            "points_per_batch": 64,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "box_nms_thresh": 0.7,
            "min_mask_region_area": 0,
        },
        "history": [],
    }

    if source_path:
        project["source"] = _resolve_source(source_path)

    if output_path:
        save_project(project, output_path)

    return project


def open_project(path):
    """Open an existing project from a JSON file.

    Args:
        path: Path to the project JSON file.

    Returns:
        dict: The project state dictionary.

    Raises:
        FileNotFoundError: If the project file does not exist.
        ValueError: If the file is not a valid project.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Project file not found: {path}")

    with open(path, "r") as f:
        project = json.load(f)

    if "name" not in project or "version" not in project:
        raise ValueError(f"Invalid project file: {path}")

    project["_path"] = path
    return project


def save_project(project, path=None):
    """Save a project to disk.

    Args:
        project: The project state dictionary.
        path: Path to save to. If None, uses the project's stored path.

    Returns:
        str: The path the project was saved to.
    """
    path = path or project.get("_path")
    if not path:
        raise ValueError("No path specified and project has no stored path.")

    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    project["modified"] = datetime.now(timezone.utc).isoformat()

    save_data = {k: v for k, v in project.items() if not k.startswith("_")}

    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)

    project["_path"] = path
    return path


def get_project_info(project):
    """Get a summary of the project state.

    Args:
        project: The project state dictionary.

    Returns:
        dict: Summary information.
    """
    info = {
        "name": project["name"],
        "version": project["version"],
        "created": project["created"],
        "modified": project["modified"],
        "model_type": project["model"]["type"],
        "model_id": project["model"]["id"],
        "device": project["model"]["device"],
        "has_source": project["source"] is not None,
        "has_masks": project["masks"] is not None,
        "has_vectors": project["vectors"] is not None,
        "history_count": len(project.get("history", [])),
    }

    if project["source"]:
        info["source_path"] = project["source"]["path"]
        info["source_crs"] = project["source"].get("crs")
        info["source_size"] = project["source"].get("size")

    if project["masks"]:
        info["mask_path"] = project["masks"]["path"]
        info["mask_count"] = project["masks"].get("count")

    if project["vectors"]:
        info["vector_path"] = project["vectors"]["path"]
        info["feature_count"] = project["vectors"].get("feature_count")

    return info


def add_history_entry(project, action, params=None, result=None):
    """Add an entry to the project history.

    Args:
        project: The project state dictionary.
        action: Action name (e.g., 'generate', 'predict', 'export').
        params: Parameters used for the action.
        result: Result summary.
    """
    entry = {
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params or {},
        "result": result or {},
    }
    project.setdefault("history", []).append(entry)


def set_source(project, source_path):
    """Set or update the source image for a project.

    Args:
        project: The project state dictionary.
        source_path: Path to the source image.

    Returns:
        dict: The updated source info.
    """
    project["source"] = _resolve_source(source_path)
    return project["source"]


def set_masks(project, mask_path, count=None):
    """Record generated masks in the project.

    Args:
        project: The project state dictionary.
        mask_path: Path to the mask file.
        count: Number of masks generated.

    Returns:
        dict: The mask info.
    """
    project["masks"] = {
        "path": os.path.abspath(mask_path),
        "count": count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return project["masks"]


def set_vectors(project, vector_path, feature_count=None):
    """Record generated vectors in the project.

    Args:
        project: The project state dictionary.
        vector_path: Path to the vector file.
        feature_count: Number of vector features.

    Returns:
        dict: The vector info.
    """
    project["vectors"] = {
        "path": os.path.abspath(vector_path),
        "feature_count": feature_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return project["vectors"]


def _resolve_source(source_path):
    """Build source info dict from a file path.

    Args:
        source_path: Path to the source image.

    Returns:
        dict: Source information.
    """
    source_path = os.path.abspath(source_path)
    info = {"path": source_path, "crs": None, "bounds": None, "size": None}

    if os.path.exists(source_path):
        try:
            import rasterio

            with rasterio.open(source_path) as src:
                info["crs"] = str(src.crs) if src.crs else None
                info["bounds"] = list(src.bounds)
                info["size"] = [src.width, src.height]
        except Exception:
            pass

    return info
