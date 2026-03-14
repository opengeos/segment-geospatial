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
import os
import tempfile
import threading
from typing import Optional

try:
    import fastapi
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI dependencies are required. "
        "Install with: pip install segment-geospatial[api]"
    )

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from samgeo import __version__


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

# Model cache: (model_version, model_id) -> (model_instance, lock)
_model_cache: dict = {}

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


def get_model(model_version: str, model_id: Optional[str] = None, **kwargs):
    """Get or create a cached model instance.

    Args:
        model_version: One of "sam", "sam2", "sam3".
        model_id: Specific model identifier. Uses default if None.
        **kwargs: Additional keyword arguments for model initialization.

    Returns:
        tuple: (model_instance, threading.Lock)

    Raises:
        HTTPException: If model_version is invalid or dependencies are missing.
    """
    if model_version not in _DEFAULT_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model_version '{model_version}'. "
                f"Must be one of: {list(_DEFAULT_MODEL_IDS.keys())}"
            ),
        )

    if model_id is None:
        model_id = _DEFAULT_MODEL_IDS[model_version]

    key = (model_version, model_id)
    if key not in _model_cache:
        try:
            if model_version == "sam":
                from samgeo.samgeo import SamGeo

                model = SamGeo(model_type=model_id, **kwargs)
            elif model_version == "sam2":
                from samgeo.samgeo2 import SamGeo2

                model = SamGeo2(model_id=model_id, **kwargs)
            elif model_version == "sam3":
                from samgeo.samgeo3 import SamGeo3

                model = SamGeo3(**kwargs)
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Dependencies for {model_version} are not installed. "
                    f"Install with: pip install segment-geospatial[{model_version}]. "
                    f"Error: {e}"
                ),
            )
        _model_cache[key] = (model, threading.Lock())

    return _model_cache[key]


def _set_image_cached(
    model, model_key: tuple, image_path: str, image_bytes: bytes
) -> bool:
    """Call model.set_image() only if the image has changed since last call.

    Computes a SHA-256 hash of the raw upload bytes and compares it to the
    last image encoded on this model. Skips the expensive image-encoder
    forward pass when the hash matches.

    Args:
        model: The SAM model instance.
        model_key: Cache key for the model, e.g. ("sam3", "facebook/sam3").
        image_path: Path to the saved image file.
        image_bytes: Raw bytes of the uploaded file (used for hashing).

    Returns:
        True if set_image was called (new image), False if skipped (cache hit).
    """
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    if _image_hash_cache.get(model_key) == img_hash:
        return False
    model.set_image(image_path)
    _image_hash_cache[model_key] = img_hash
    return True


async def _save_upload(file: UploadFile, tmpdir: str) -> tuple:
    """Save an uploaded file to a temporary directory.

    Args:
        file: The uploaded file.
        tmpdir: The temporary directory path.

    Returns:
        tuple: (path to saved file, raw file bytes).
    """
    suffix = os.path.splitext(file.filename or "image.tif")[1] or ".tif"
    path = os.path.join(tmpdir, f"input{suffix}")
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    return path, content


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
            return JSONResponse(content=json.load(f))

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

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output_format '{output_format}'. "
                "Must be one of: geojson, geotiff, png"
            ),
        )


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
    max_size = _normalize_max_size(max_size)
    tmpdir = tempfile.mkdtemp()
    try:
        input_path, image_bytes = await _save_upload(file, tmpdir)
        output_path = os.path.join(tmpdir, "mask.tif")

        if model_version == "sam3":
            model, lock = get_model(model_version, model_id)
            model_key = (model_version, model_id or _DEFAULT_MODEL_IDS[model_version])
            with lock:
                _set_image_cached(model, model_key, input_path, image_bytes)
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

        return _format_response(output_path, output_format, tmpdir)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/predict")
async def segment_predict(
    file: UploadFile = File(...),
    model_version: str = Form("sam2"),
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

    Args:
        file: Image file (TIFF, PNG, JPEG).
        model_version: One of "sam", "sam2".
        model_id: Specific model identifier.
        output_format: One of "geojson", "geotiff", "png".
        point_coords: JSON string of [[x, y], ...] coordinate pairs.
        point_labels: JSON string of [1, 0, ...] labels (1=foreground,
            0=background).
        boxes: JSON string of [[xmin, ymin, xmax, ymax], ...] bounding boxes.
        point_crs: CRS string (e.g., "EPSG:4326") for point coordinates.
        multimask_output: Whether to return multiple masks per prompt.
        min_size: Minimum mask size in pixels.
        max_size: Maximum mask size in pixels.

    Returns:
        Segmentation result in the requested format.
    """
    if model_version == "sam3":
        raise HTTPException(
            status_code=400,
            detail="Use /segment/text for SAM3 text-based segmentation.",
        )

    if point_coords is None and boxes is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of point_coords or boxes must be provided.",
        )

    max_size = _normalize_max_size(max_size)
    tmpdir = tempfile.mkdtemp()
    try:
        input_path, image_bytes = await _save_upload(file, tmpdir)
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

        model, lock = get_model(model_version, model_id, automatic=False)
        model_key = (model_version, model_id or _DEFAULT_MODEL_IDS[model_version])
        with lock:
            _set_image_cached(model, model_key, input_path, image_bytes)
            model.predict(
                point_coords=parsed_coords,
                point_labels=parsed_labels,
                boxes=parsed_boxes,
                point_crs=point_crs,
                multimask_output=multimask_output,
                output=output_path,
            )

        return _format_response(output_path, output_format, tmpdir)
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in prompt fields: {e}"
        )
    except Exception as e:
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
        output_format: One of "geojson", "geotiff", "png".
        confidence_threshold: Confidence threshold for detections.
        min_size: Minimum mask size in pixels.
        max_size: Maximum mask size in pixels.

    Returns:
        Segmentation result in the requested format.
    """
    max_size = _normalize_max_size(max_size)
    tmpdir = tempfile.mkdtemp()
    try:
        input_path, image_bytes = await _save_upload(file, tmpdir)
        output_path = os.path.join(tmpdir, "mask.tif")

        model, lock = get_model(
            "sam3",
            model_id,
            backend=backend,
            confidence_threshold=confidence_threshold,
        )
        model_key = ("sam3", model_id or _DEFAULT_MODEL_IDS["sam3"])
        with lock:
            _set_image_cached(model, model_key, input_path, image_bytes)
            model.generate_masks(
                prompt=prompt,
                min_size=min_size,
                max_size=max_size,
            )
            model.save_masks(output=output_path)

        return _format_response(output_path, output_format, tmpdir)
    except HTTPException:
        raise
    except Exception as e:
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
        version, mid = args.preload.split(":", 1)
        get_model(version, mid)

    uvicorn.run("samgeo.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
