"""REST API for segment-geospatial.

Provides FastAPI endpoints for image segmentation using SAM, SAM2, and SAM3
models. Install with: pip install segment-geospatial[api]

Usage:
    samgeo-api                          # Start on default port 8000
    samgeo-api --port 9000              # Custom port
    samgeo-api --preload sam2:sam2-hiera-large  # Preload a model
    uvicorn samgeo.api:app              # Direct uvicorn usage
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Optional

try:
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI dependencies are required. "
        "Install with: pip install segment-geospatial[api]"
    )

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from samgeo import __version__

logger = logging.getLogger("uvicorn.error")

_VALID_OUTPUT_FORMATS = {"geojson", "geotiff", "png", "detections", "json"}


def _normalize_max_size(max_size: Optional[int]) -> Optional[int]:
    """Treat max_size of 0 or negative as None (no limit).

    Args:
        max_size: The max_size value from the form.

    Returns:
        None if max_size is 0 or negative, otherwise the original value.
    """
    if max_size is not None and max_size <= 0:
        return None
    return max_size


app = FastAPI(
    title="samgeo API",
    description="REST API for geospatial image segmentation with SAM models.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache: (model_version, model_id) -> (model_instance, lock)
_model_cache: dict = {}
_model_cache_lock = threading.Lock()

# Image encoding cache: maps model cache key -> hash of last encoded image.
# When the same image is sent again, we skip the expensive set_image() call.
_image_hash_cache: dict = {}

# Default model IDs per version
_DEFAULT_MODEL_IDS = {
    "sam": "vit_h",
    "sam2": "sam2-hiera-large",
    "sam3": "facebook/sam3",
}

_AVAILABLE_MODELS = {
    "sam": ["vit_h", "vit_l", "vit_b"],
    "sam2": [
        "sam2-hiera-tiny",
        "sam2-hiera-small",
        "sam2-hiera-base-plus",
        "sam2-hiera-large",
    ],
    "sam3": ["facebook/sam3"],
}

# Maps model_version to the correct pip extra name
_EXTRAS_MAP = {
    "sam": "samgeo",
    "sam2": "samgeo2",
    "sam3": "samgeo3",
}


def get_model(model_version: str, model_id: Optional[str] = None, **kwargs):
    """Get or create a cached model instance.

    Args:
        model_version: One of "sam", "sam2", "sam3".
        model_id: Specific model identifier. Uses default if None.
        **kwargs: Additional keyword arguments for model initialization.

    Returns:
        tuple: (model_instance, threading.Lock)

    Raises:
        HTTPException: If model_version or model_id is invalid, or
            dependencies are missing.
    """
    if model_version not in _DEFAULT_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model_version '{model_version}'. "
                f"Must be one of: {list(_DEFAULT_MODEL_IDS.keys())}"
            ),
        )

    if not model_id:
        model_id = _DEFAULT_MODEL_IDS[model_version]

    valid_ids = _AVAILABLE_MODELS[model_version]
    if model_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model_id '{model_id}' for {model_version}. "
                f"Must be one of: {valid_ids}"
            ),
        )

    key = (model_version, model_id)
    with _model_cache_lock:
        if key in _model_cache:
            logger.info("Model cache hit for %s", key)
            return _model_cache[key]

        logger.info("Loading model %s", key)
        extra = _EXTRAS_MAP.get(model_version, model_version)
        try:
            if model_version == "sam":
                from samgeo.samgeo import SamGeo

                model = SamGeo(model_type=model_id, **kwargs)
            elif model_version == "sam2":
                from samgeo.samgeo2 import SamGeo2

                model = SamGeo2(model_id=model_id, **kwargs)
            elif model_version == "sam3":
                from samgeo.samgeo3 import SamGeo3

                kwargs.setdefault("enable_inst_interactivity", True)
                model = SamGeo3(**kwargs)
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Dependencies for {model_version} are not installed. "
                    f"Install with: pip install segment-geospatial[{extra}]. "
                    f"Error: {e}"
                ),
            )
        _model_cache[key] = (model, threading.Lock())
        return _model_cache[key]


def _set_image_cached(
    model, model_key: tuple, image_path: str, image_hash: str
) -> bool:
    """Call model.set_image() only if the image has changed since last call.

    Compares the provided hash to the last image encoded on this model.
    Skips the expensive image-encoder forward pass when the hash matches.

    Args:
        model: The SAM model instance.
        model_key: Cache key for the model, e.g. ("sam3", "facebook/sam3").
        image_path: Path to the saved image file.
        image_hash: SHA-256 hex digest of the uploaded file bytes.

    Returns:
        True if set_image was called (new image), False if skipped (cache hit).
    """
    if _image_hash_cache.get(model_key) == image_hash:
        logger.info("Image cache hit for model %s, skipping set_image()", model_key)
        # Update source path so save_masks() can find GeoTIFF metadata
        # even though we skip the expensive encoding step.
        model.source = image_path
        return False
    logger.info("Encoding new image for model %s", model_key)
    model.set_image(image_path)
    _image_hash_cache[model_key] = image_hash
    return True


async def _save_upload(file: UploadFile, tmpdir: str) -> tuple:
    """Save an uploaded file to a temporary directory using chunked streaming.

    Streams the file to disk in 1 MB chunks to avoid loading the entire
    file into memory at once, which is important for large raster files.

    Args:
        file: The uploaded file.
        tmpdir: The temporary directory path.

    Returns:
        tuple: (path to saved file, SHA-256 hex digest of file content).
    """
    suffix = os.path.splitext(file.filename or "image.tif")[1] or ".tif"
    path = os.path.join(tmpdir, f"input{suffix}")
    sha = hashlib.sha256()
    with open(path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            sha.update(chunk)
            f.write(chunk)
    return path, sha.hexdigest()


def _validate_output_format(output_format: str) -> None:
    """Validate the output format before processing.

    Args:
        output_format: The requested output format.

    Raises:
        HTTPException: If the format is not valid.
    """
    if output_format not in _VALID_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output_format '{output_format}'. "
                f"Must be one of: {', '.join(sorted(_VALID_OUTPUT_FORMATS))}"
            ),
        )


def _format_response(raster_path: str, output_format: str, tmpdir: str):
    """Convert a raster mask to the requested output format.

    Args:
        raster_path: Path to the raster mask file.
        output_format: One of "geojson", "geotiff", "png".
        tmpdir: Temporary directory for intermediate files.

    Returns:
        FastAPI response object.

    Raises:
        HTTPException: If raster file not found or conversion fails.
    """
    if not os.path.exists(raster_path):
        raise HTTPException(
            status_code=500, detail="Segmentation produced no output."
        )

    if output_format == "geojson":
        from samgeo.common import raster_to_geojson

        geojson_path = os.path.join(tmpdir, "output.geojson")
        raster_to_geojson(raster_path, geojson_path)
        with open(geojson_path) as f:
            data = json.load(f)
        _cleanup_tmpdir(tmpdir)
        return JSONResponse(content=data)

    elif output_format == "geotiff":
        return FileResponse(
            raster_path, media_type="image/tiff", filename="mask.tif"
        )

    elif output_format == "png":
        from PIL import Image

        img = Image.open(raster_path)
        png_path = os.path.join(tmpdir, "output.png")
        img.save(png_path)
        return FileResponse(
            png_path, media_type="image/png", filename="mask.png"
        )

    elif output_format in ("json", "detections"):
        data = _extract_bboxes_from_raster(raster_path, output_format)
        _cleanup_tmpdir(tmpdir)
        return JSONResponse(content=data)


def _extract_bboxes_from_raster(raster_path: str, output_format: str) -> dict:
    """Extract bounding boxes from a raster mask file.

    Each unique non-zero value in the raster is treated as a separate object.
    Bounding boxes are computed from the pixel regions of each object.

    Args:
        raster_path: Path to the raster mask file.
        output_format: Either "json" for pixel-coordinate bboxes or
            "detections" for a GeoJSON FeatureCollection with geographic
            coordinates.

    Returns:
        A dict with bounding box information in the requested format.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
        has_georef = crs is not None

    unique_vals = np.unique(mask)
    unique_vals = unique_vals[unique_vals != 0]

    if output_format == "json":
        detections = []
        for i, val in enumerate(unique_vals):
            rows, cols = np.where(mask == val)
            x1, y1, x2, y2 = int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())
            detections.append({
                "id": i + 1,
                "value": int(val),
                "bbox": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1,
            })
        return {
            "image_width": int(mask.shape[1]),
            "image_height": int(mask.shape[0]),
            "num_detections": len(detections),
            "detections": detections,
        }

    else:  # detections (GeoJSON)
        features = []
        for i, val in enumerate(unique_vals):
            rows, cols = np.where(mask == val)
            x1, y1, x2, y2 = float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())

            if has_georef:
                geo_x1, geo_y1 = transform * (x1, y1)
                geo_x2, geo_y2 = transform * (x2, y2)
            else:
                geo_x1, geo_y1 = x1, y1
                geo_x2, geo_y2 = x2, y2

            coords = [
                [geo_x1, geo_y1],
                [geo_x2, geo_y1],
                [geo_x2, geo_y2],
                [geo_x1, geo_y2],
                [geo_x1, geo_y1],
            ]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "id": i + 1,
                    "value": int(val),
                    "bbox_pixel": [x1, y1, x2, y2],
                },
            })
        result = {
            "type": "FeatureCollection",
            "features": features,
            "num_detections": len(features),
        }
        if crs:
            result["crs"] = crs
        return result


def _build_detections_json(model) -> dict:
    """Build a plain JSON response with bounding boxes and scores in pixel coordinates.

    Suitable for non-georeferenced images where geographic coordinates are
    not available.

    Args:
        model: The SAM model instance with boxes, scores, and image
            dimension attributes.

    Returns:
        A dict with image dimensions and a list of detections, each containing
        id, bbox (pixel coords), and score.
    """
    boxes = model.boxes if model.boxes is not None else []
    scores = model.scores if model.scores is not None else []

    detections = []
    for i, box in enumerate(boxes):
        box_np = np.asarray(box).flatten()
        x1, y1, x2, y2 = int(round(box_np[0])), int(round(box_np[1])), int(round(box_np[2])), int(round(box_np[3]))
        det = {
            "id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "width": x2 - x1,
            "height": y2 - y1,
        }
        if i < len(scores):
            score_val = scores[i]
            det["score"] = float(score_val) if not isinstance(score_val, float) else score_val
        detections.append(det)

    return {
        "image_width": model.image_width,
        "image_height": model.image_height,
        "num_detections": len(detections),
        "detections": detections,
    }


def _build_detections_geojson(model, source_path: str) -> dict:
    """Build a GeoJSON FeatureCollection from detected bounding boxes and scores.

    Converts pixel-coordinate bounding boxes to geographic coordinates when
    the source image has georeferencing information. Falls back to pixel
    coordinates otherwise.

    Args:
        model: The SAM model instance with boxes and scores attributes.
        source_path: Path to the source image file.

    Returns:
        A GeoJSON FeatureCollection dict with one polygon feature per detection.
    """
    import rasterio

    boxes = model.boxes if model.boxes is not None else []
    scores = model.scores if model.scores is not None else []

    has_georef = False
    transform = None
    crs = None
    try:
        if source_path and source_path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(source_path) as src:
                if src.crs is not None:
                    transform = src.transform
                    crs = src.crs.to_string()
                    has_georef = True
    except Exception:
        pass

    features = []
    for i, box in enumerate(boxes):
        box_np = np.asarray(box).flatten()
        x1, y1, x2, y2 = float(box_np[0]), float(box_np[1]), float(box_np[2]), float(box_np[3])

        if has_georef:
            # Convert pixel coords to geographic coords
            geo_x1, geo_y1 = transform * (x1, y1)
            geo_x2, geo_y2 = transform * (x2, y2)
        else:
            geo_x1, geo_y1 = x1, y1
            geo_x2, geo_y2 = x2, y2

        # Build bbox polygon (clockwise)
        coords = [
            [geo_x1, geo_y1],
            [geo_x2, geo_y1],
            [geo_x2, geo_y2],
            [geo_x1, geo_y2],
            [geo_x1, geo_y1],
        ]

        props = {"id": i + 1}
        if i < len(scores):
            score_val = scores[i]
            props["score"] = float(score_val) if not isinstance(score_val, float) else score_val
        props["bbox_pixel"] = [x1, y1, x2, y2]

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": props,
            }
        )

    result = {
        "type": "FeatureCollection",
        "features": features,
        "num_detections": len(features),
    }
    if crs:
        result["crs"] = crs

    return result


def _cleanup_tmpdir(tmpdir: str) -> None:
    """Remove a temporary directory, ignoring errors.

    Args:
        tmpdir: Path to the temporary directory to remove.
    """
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.get("/models")
def list_models():
    """List available and currently loaded models."""
    loaded = [list(key) for key in _model_cache]
    return {"models": _AVAILABLE_MODELS, "loaded": loaded}


@app.delete("/models")
def clear_models():
    """Clear the model cache and free GPU memory."""
    _model_cache.clear()
    _image_hash_cache.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {"status": "cleared"}


@app.post("/segment/automatic")
async def segment_automatic(
    file: UploadFile = File(...),
    model_version: str = Form("sam2"),
    model_id: Optional[str] = Form(None),
    output_format: str = Form("geojson"),
    foreground: bool = Form(True),
    unique: bool = Form(True),
    min_size: int = Form(0),
    max_size: Optional[int] = Form(None),
    points_per_side: int = Form(32),
    pred_iou_thresh: float = Form(0.8),
    stability_score_thresh: float = Form(0.95),
):
    """Run automatic mask generation on an uploaded image.

    Args:
        file: Image file (TIFF, PNG, JPEG).
        model_version: One of "sam", "sam2", "sam3".
        model_id: Specific model identifier.
        output_format: One of "geojson", "geotiff", "png".
        foreground: Whether to extract foreground objects only.
        unique: Whether to assign unique IDs to each object.
        min_size: Minimum mask size in pixels.
        max_size: Maximum mask size in pixels.
        points_per_side: Number of points sampled per side (SAM/SAM2).
        pred_iou_thresh: IoU threshold for filtering masks.
        stability_score_thresh: Stability score threshold for filtering.

    Returns:
        Segmentation result in the requested format.
    """
    _validate_output_format(output_format)
    max_size = _normalize_max_size(max_size)
    tmpdir = tempfile.mkdtemp()
    try:
        input_path, image_hash = await _save_upload(file, tmpdir)
        output_path = os.path.join(tmpdir, "mask.tif")

        t_start = time.time()
        if model_version == "sam3":
            model, lock = get_model(model_version, model_id)
            model_key = (model_version, model_id or _DEFAULT_MODEL_IDS[model_version])
            with lock:
                _set_image_cached(model, model_key, input_path, image_hash)
                model.generate_masks(
                    prompt="everything",
                    min_size=min_size,
                    max_size=max_size,
                )
                model.save_masks(output=output_path, unique=unique)
        else:
            sam_kwargs = {
                "points_per_side": points_per_side,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
            }
            if model_version == "sam":
                model, lock = get_model(
                    model_version, model_id, sam_kwargs=sam_kwargs
                )
            else:
                model, lock = get_model(model_version, model_id, **sam_kwargs)

            with lock:
                model.generate(
                    source=input_path,
                    output=output_path,
                    foreground=foreground,
                    unique=unique,
                    min_size=min_size,
                    max_size=max_size,
                )

        t_inference = time.time() - t_start
        logger.info(
            "Automatic segmentation completed in %.2fs (model: %s)",
            t_inference,
            model_version,
        )
        return _format_response(output_path, output_format, tmpdir)
    except HTTPException:
        _cleanup_tmpdir(tmpdir)
        raise
    except Exception as e:
        _cleanup_tmpdir(tmpdir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/predict")
async def segment_predict(
    file: UploadFile = File(...),
    model_version: str = Form("sam3"),
    model_id: Optional[str] = Form(None),
    output_format: str = Form("geojson"),
    point_coords: Optional[str] = Form(None),
    point_labels: Optional[str] = Form(None),
    boxes: Optional[str] = Form(None),
    point_crs: Optional[str] = Form(None),
    multimask_output: bool = Form(False),
    min_size: int = Form(0),
    max_size: Optional[int] = Form(None),
):
    """Run prompt-based segmentation with points or bounding boxes.

    For SAM3 with bounding box prompts, the model finds all similar objects
    in the image (not just the object inside the box). Point prompts with
    SAM3 segment the specific object at the point location.

    Args:
        file: Image file (TIFF, PNG, JPEG).
        model_version: One of "sam", "sam2", "sam3".
        model_id: Specific model identifier.
        output_format: One of "geojson", "geotiff", "png", "json", "detections".
        point_coords: JSON string of [[x, y], ...] coordinate pairs.
        point_labels: JSON string of [1, 0, ...] labels (1=foreground,
            0=background).
        boxes: JSON string of [[xmin, ymin, xmax, ymax], ...] bounding boxes.
        point_crs: CRS string (e.g., "EPSG:4326") for point/box coordinates.
        multimask_output: Whether to return multiple masks per prompt.
        min_size: Minimum mask size in pixels.
        max_size: Maximum mask size in pixels.

    Returns:
        Segmentation result in the requested format.
    """
    _validate_output_format(output_format)

    # Swagger UI sends empty strings for unfilled optional fields
    if not point_coords:
        point_coords = None
    if not point_labels:
        point_labels = None
    if not boxes:
        boxes = None
    if not point_crs:
        point_crs = None

    if point_coords is None and boxes is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of point_coords or boxes must be provided.",
        )

    max_size = _normalize_max_size(max_size)
    tmpdir = tempfile.mkdtemp()
    try:
        input_path, image_hash = await _save_upload(file, tmpdir)
        output_path = os.path.join(tmpdir, "mask.tif")

        # Parse JSON prompt fields
        parsed_coords = None
        parsed_labels = None
        parsed_boxes = None

        if point_coords is not None:
            parsed_coords = np.array(json.loads(point_coords))
        if point_labels is not None:
            parsed_labels = np.array(json.loads(point_labels))
        if boxes is not None:
            parsed_boxes = np.array(json.loads(boxes))

        t_start = time.time()

        if model_version == "sam3":
            model, lock = get_model(model_version, model_id)
            model_key = (
                model_version,
                model_id or _DEFAULT_MODEL_IDS[model_version],
            )
            with lock:
                _set_image_cached(model, model_key, input_path, image_hash)
                if parsed_boxes is not None:
                    # Use generate_masks_by_boxes to find all similar objects
                    box_list = parsed_boxes.tolist()
                    if parsed_boxes.ndim == 1:
                        box_list = [box_list]
                    model.generate_masks_by_boxes(
                        boxes=box_list,
                        box_crs=point_crs,
                        min_size=min_size,
                        max_size=max_size,
                    )
                else:
                    # Use predict_inst for point-only prompts
                    model.predict_inst(
                        point_coords=parsed_coords,
                        point_labels=parsed_labels,
                        multimask_output=multimask_output,
                        point_crs=point_crs,
                    )
                if model.masks is None or len(model.masks) == 0:
                    _cleanup_tmpdir(tmpdir)
                    raise HTTPException(
                        status_code=404,
                        detail="No objects found for the given prompts.",
                    )
                model.save_masks(
                    output=output_path,
                    min_size=min_size,
                    max_size=max_size,
                )
        else:
            model, lock = get_model(model_version, model_id, automatic=False)
            model_key = (
                model_version,
                model_id or _DEFAULT_MODEL_IDS[model_version],
            )
            with lock:
                _set_image_cached(model, model_key, input_path, image_hash)
                model.predict(
                    point_coords=parsed_coords,
                    point_labels=parsed_labels,
                    boxes=parsed_boxes,
                    point_crs=point_crs,
                    multimask_output=multimask_output,
                    output=output_path,
                )

        t_inference = time.time() - t_start
        logger.info(
            "Prompt segmentation completed in %.2fs (model: %s)",
            t_inference,
            model_version,
        )
        return _format_response(output_path, output_format, tmpdir)
    except HTTPException:
        _cleanup_tmpdir(tmpdir)
        raise
    except json.JSONDecodeError as e:
        _cleanup_tmpdir(tmpdir)
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in prompt fields: {e}"
        )
    except Exception as e:
        _cleanup_tmpdir(tmpdir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/text")
async def segment_text(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    model_id: Optional[str] = Form(None),
    backend: str = Form("meta"),
    output_format: str = Form("geojson"),
    confidence_threshold: float = Form(0.5),
    min_size: int = Form(0),
    max_size: Optional[int] = Form(None),
):
    """Run text-prompt segmentation using SAM3.

    Args:
        file: Image file (TIFF, PNG, JPEG).
        prompt: Text description of objects to segment (e.g., "building").
        model_id: SAM3 model identifier.
        backend: SAM3 backend, one of "meta" or "transformers".
        output_format: One of "geojson", "geotiff", "png", "detections", "json".
            Use "detections" to get a GeoJSON FeatureCollection of bounding
            box polygons in geographic coordinates with confidence scores.
            Use "json" for a plain JSON array of bounding boxes in pixel
            coordinates, suitable for non-georeferenced images.
        confidence_threshold: Confidence threshold for detections.
        min_size: Minimum mask size in pixels.
        max_size: Maximum mask size in pixels.

    Returns:
        Segmentation result in the requested format.
    """
    _validate_output_format(output_format)
    max_size = _normalize_max_size(max_size)
    tmpdir = tempfile.mkdtemp()
    try:
        input_path, image_hash = await _save_upload(file, tmpdir)
        output_path = os.path.join(tmpdir, "mask.tif")

        model, lock = get_model(
            "sam3",
            model_id,
            backend=backend,
            confidence_threshold=confidence_threshold,
        )
        t_start = time.time()
        model_key = ("sam3", model_id or _DEFAULT_MODEL_IDS["sam3"])
        with lock:
            _set_image_cached(model, model_key, input_path, image_hash)
            model.generate_masks(
                prompt=prompt,
                min_size=min_size,
                max_size=max_size,
            )
            if model.masks is None or len(model.masks) == 0:
                _cleanup_tmpdir(tmpdir)
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "No objects found for the given prompt. "
                        "Please try a different prompt or adjust parameters."
                    ),
                )
            if output_format in ("detections", "json"):
                if output_format == "detections":
                    det_result = _build_detections_geojson(model, input_path)
                else:
                    det_result = _build_detections_json(model)
            else:
                model.save_masks(output=output_path)

        t_inference = time.time() - t_start
        logger.info(
            "Text segmentation completed in %.2fs (prompt: '%s')",
            t_inference,
            prompt,
        )
        if output_format in ("detections", "json"):
            _cleanup_tmpdir(tmpdir)
            return JSONResponse(content=det_result)
        return _format_response(output_path, output_format, tmpdir)
    except HTTPException:
        _cleanup_tmpdir(tmpdir)
        raise
    except Exception as e:
        _cleanup_tmpdir(tmpdir)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for the samgeo-api console script."""
    parser = argparse.ArgumentParser(
        description="Run the samgeo REST API server."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--preload",
        type=str,
        default=None,
        help="Preload a model at startup, e.g. 'sam2:sam2-hiera-large'",
    )
    args = parser.parse_args()

    if args.preload:
        if ":" not in args.preload:
            parser.error(
                "Invalid --preload format. "
                "Expected 'model_version:model_id', e.g. 'sam2:sam2-hiera-large'"
            )
        version, mid = args.preload.split(":", 1)
        get_model(version, mid)

    uvicorn.run("samgeo.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
