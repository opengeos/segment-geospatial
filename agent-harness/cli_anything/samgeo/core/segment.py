"""Segmentation operations: automatic, point, box, and text-based segmentation.

All functions call the real samgeo library — no reimplementation.
"""

import os
from datetime import datetime, timezone

from cli_anything.samgeo.core.project import add_history_entry, set_masks
from cli_anything.samgeo.utils.samgeo_backend import get_sam_model


def automatic_segment(
    project,
    output,
    foreground=True,
    unique=True,
    erosion_kernel=None,
    mask_multiplier=255,
    min_size=0,
    max_size=None,
    batch=False,
    **kwargs,
):
    """Run automatic mask generation on the source image.

    Args:
        project: The project state dict.
        output: Output mask file path.
        foreground: Whether to generate foreground masks.
        unique: Whether to assign unique values to each mask.
        erosion_kernel: Erosion kernel size tuple.
        mask_multiplier: Mask value multiplier.
        min_size: Minimum mask area in pixels.
        max_size: Maximum mask area in pixels.
        batch: Whether to use batch mode for large images.
        **kwargs: Additional kwargs for the model.

    Returns:
        dict: Result with output path and mask count.
    """
    source = _get_source_path(project)
    model_cfg = project["model"]

    output = os.path.abspath(output)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if model_cfg["type"] == "sam3":
        # SAM3 uses set_image() + generate_masks(prompt) + save_masks(output)
        model = get_sam_model(
            model_type="sam3",
            model_id=model_cfg["id"],
            device=model_cfg.get("device"),
        )
        model.set_image(source)
        masks = model.generate_masks(
            prompt="",
            min_size=min_size,
            max_size=max_size,
            **kwargs,
        )
        model.save_masks(output=output, unique=unique)
    else:
        model = get_sam_model(
            model_type=model_cfg["type"],
            model_id=model_cfg["id"],
            device=model_cfg.get("device"),
            automatic=True,
            **{k: v for k, v in project.get("parameters", {}).items() if v is not None},
        )
        masks = model.generate(
            source,
            output=output,
            foreground=foreground,
            unique=unique,
            erosion_kernel=erosion_kernel,
            mask_multiplier=mask_multiplier,
            min_size=min_size,
            max_size=max_size,
            batch=batch,
            **kwargs,
        )

    mask_count = len(masks) if isinstance(masks, list) else None
    set_masks(project, output, count=mask_count)

    result = {
        "output": output,
        "mask_count": mask_count,
        "file_size": os.path.getsize(output) if os.path.exists(output) else 0,
        "model_type": model_cfg["type"],
        "model_id": model_cfg["id"],
    }

    add_history_entry(
        project,
        "automatic_segment",
        params={
            "foreground": foreground,
            "unique": unique,
            "min_size": min_size,
            "max_size": max_size,
        },
        result=result,
    )

    return result


def predict_points(
    project,
    point_coords,
    point_labels,
    output,
    multimask_output=False,
    mask_multiplier=255,
    **kwargs,
):
    """Run interactive prediction with point prompts.

    Args:
        project: The project state dict.
        point_coords: List of [x, y] coordinates.
        point_labels: List of labels (1=foreground, 0=background).
        output: Output mask file path.
        multimask_output: Whether to output multiple masks.
        mask_multiplier: Mask value multiplier.
        **kwargs: Additional kwargs.

    Returns:
        dict: Result with output path.
    """
    source = _get_source_path(project)
    model_cfg = project["model"]

    output = os.path.abspath(output)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    import numpy as np

    coords = np.array(point_coords)
    labels = np.array(point_labels)

    if model_cfg["type"] == "sam3":
        # SAM3 interactive uses predict_inst() and requires enable_inst_interactivity
        model = get_sam_model(
            model_type="sam3",
            model_id=model_cfg["id"],
            device=model_cfg.get("device"),
            enable_inst_interactivity=True,
        )
        model.set_image(source)
        model.predict_inst(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask_output,
            **kwargs,
        )
        model.save_masks(output=output)
    else:
        model = get_sam_model(
            model_type=model_cfg["type"],
            model_id=model_cfg["id"],
            device=model_cfg.get("device"),
            automatic=False,
        )
        model.set_image(source)
        model.predict(
            point_coords=coords,
            point_labels=labels,
            output=output,
            multimask_output=multimask_output,
            mask_multiplier=mask_multiplier,
            **kwargs,
        )

    set_masks(project, output)

    result = {
        "output": output,
        "point_count": len(point_coords),
        "file_size": os.path.getsize(output) if os.path.exists(output) else 0,
        "model_type": model_cfg["type"],
    }

    add_history_entry(
        project,
        "predict_points",
        params={
            "point_coords": [list(c) for c in point_coords],
            "point_labels": list(point_labels),
        },
        result=result,
    )

    return result


def predict_boxes(
    project,
    boxes,
    output,
    multimask_output=False,
    mask_multiplier=255,
    **kwargs,
):
    """Run interactive prediction with bounding box prompts.

    Args:
        project: The project state dict.
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        output: Output mask file path.
        multimask_output: Whether to output multiple masks.
        mask_multiplier: Mask value multiplier.
        **kwargs: Additional kwargs.

    Returns:
        dict: Result with output path.
    """
    source = _get_source_path(project)
    model_cfg = project["model"]

    output = os.path.abspath(output)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    import numpy as np

    box_array = np.array(boxes)

    if model_cfg["type"] == "sam3":
        # SAM3 interactive uses predict_inst() and requires enable_inst_interactivity
        model = get_sam_model(
            model_type="sam3",
            model_id=model_cfg["id"],
            device=model_cfg.get("device"),
            enable_inst_interactivity=True,
        )
        model.set_image(source)
        model.predict_inst(
            box=box_array,
            multimask_output=multimask_output,
            **kwargs,
        )
        model.save_masks(output=output)
    else:
        model = get_sam_model(
            model_type=model_cfg["type"],
            model_id=model_cfg["id"],
            device=model_cfg.get("device"),
            automatic=False,
        )
        model.set_image(source)
        model.predict(
            boxes=box_array,
            output=output,
            multimask_output=multimask_output,
            mask_multiplier=mask_multiplier,
            **kwargs,
        )

    set_masks(project, output)

    result = {
        "output": output,
        "box_count": len(boxes),
        "file_size": os.path.getsize(output) if os.path.exists(output) else 0,
        "model_type": model_cfg["type"],
    }

    add_history_entry(
        project,
        "predict_boxes",
        params={"boxes": [list(b) for b in boxes]},
        result=result,
    )

    return result


def text_segment(
    project,
    text_prompt,
    output,
    box_threshold=0.24,
    text_threshold=0.24,
    **kwargs,
):
    """Run text-based segmentation using LangSAM.

    Args:
        project: The project state dict.
        text_prompt: Text description of objects to segment.
        output: Output mask file path.
        box_threshold: Box confidence threshold.
        text_threshold: Text confidence threshold.
        **kwargs: Additional kwargs.

    Returns:
        dict: Result with output path and detection count.
    """
    source = _get_source_path(project)

    from cli_anything.samgeo.utils.samgeo_backend import get_sam_model

    model = get_sam_model(
        model_type="text_sam",
        model_id=project["model"].get("id"),
        device=project["model"].get("device"),
    )

    output = os.path.abspath(output)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    model.predict(
        source,
        text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output=output,
        **kwargs,
    )

    set_masks(project, output)

    result = {
        "output": output,
        "text_prompt": text_prompt,
        "file_size": os.path.getsize(output) if os.path.exists(output) else 0,
    }

    add_history_entry(
        project,
        "text_segment",
        params={"text_prompt": text_prompt},
        result=result,
    )

    return result


def _get_source_path(project):
    """Extract and validate the source path from a project.

    Args:
        project: The project state dict.

    Returns:
        str: The source image path.

    Raises:
        ValueError: If no source is set.
    """
    if not project.get("source") or not project["source"].get("path"):
        raise ValueError(
            "No source image set. Use 'project new --source <path>' or "
            "'project set-source <path>' first."
        )
    path = project["source"]["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source image not found: {path}")
    return path
