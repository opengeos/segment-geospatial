"""Pure helper functions for the SamGeo plugin.

This module must not import ``qgis`` or any Qt binding, so its logic can be
unit tested without a running QGIS instance.
"""

import os
import re

# Model family keys used by the plugin.
FAMILY_SAM3 = "sam3"
FAMILY_SAM2 = "sam2"
FAMILY_SAM1 = "sam1"

# Weight choices for each model family as (label, value) pairs. The values
# are what the matching samgeo constructor accepts: ``SamGeo3(model_id=...)``,
# ``SamGeo2(model_id=...)`` and ``SamGeo(model_type=...)``. The first entry of
# each list is the default and matches samgeo's own default.
MODEL_WEIGHTS = {
    FAMILY_SAM3: [
        ("SAM 3", "facebook/sam3"),
        ("SAM 3.1 (meta backend only)", "facebook/sam3.1"),
    ],
    FAMILY_SAM2: [
        ("Large (hiera-large)", "sam2-hiera-large"),
        ("Base+ (hiera-base-plus)", "sam2-hiera-base-plus"),
        ("Small (hiera-small)", "sam2-hiera-small"),
        ("Tiny (hiera-tiny)", "sam2-hiera-tiny"),
    ],
    FAMILY_SAM1: [
        ("ViT-H (huge)", "vit_h"),
        ("ViT-L (large)", "vit_l"),
        ("ViT-B (base)", "vit_b"),
    ],
}

# Attribute fields added to saved vector masks.
SOURCE_FIELD = "source"
PROMPT_FIELD = "prompt"

# Prompt types recorded in the ``prompt`` field.
PROMPT_TEXT = "text"
PROMPT_POINT = "point"
PROMPT_BOX = "box"
PROMPT_BATCH_POINT = "batch_point"

# Output file extensions keyed by a keyword of the Output tab format combo.
_FORMAT_EXTENSIONS = (
    ("GeoPackage", ".gpkg"),
    ("Shapefile", ".shp"),
    ("Raster", ".tif"),
)


def model_family(label):
    """Return the model family key for a model combo label.

    Args:
        label: Text of the model combo, e.g. ``"SamGeo2 (SAM2)"``.

    Returns:
        One of ``"sam3"``, ``"sam2"`` or ``"sam1"``. Unknown labels map to
        ``"sam1"``, matching the plugin's historical fallback to ``SamGeo``.
    """
    if "SamGeo3" in label:
        return FAMILY_SAM3
    if "SamGeo2" in label:
        return FAMILY_SAM2
    return FAMILY_SAM1


def model_weight_options(family):
    """Return the weight choices for a model family.

    Args:
        family: A model family key such as ``"sam2"``.

    Returns:
        A list of ``(label, value)`` tuples, default first. Unknown families
        return an empty list.
    """
    return list(MODEL_WEIGHTS.get(family, []))


def default_model_weight(family):
    """Return the default weight value for a model family.

    Args:
        family: A model family key such as ``"sam1"``.

    Returns:
        The default value (e.g. ``"vit_h"``), or None for unknown families.
    """
    options = MODEL_WEIGHTS.get(family)
    return options[0][1] if options else None


def model_constructor_kwargs(family, weight):
    """Build the weight keyword argument for a samgeo model constructor.

    Args:
        family: A model family key such as ``"sam2"``.
        weight: The selected weight value, or None/empty for the default.

    Returns:
        A dict to splat into the constructor: ``{"model_id": ...}`` for SAM3
        and SAM2, ``{"model_type": ...}`` for SAM1.

    Raises:
        ValueError: If ``weight`` is not a valid choice for ``family``.
    """
    if not weight:
        weight = default_model_weight(family)
    valid = [value for _, value in MODEL_WEIGHTS.get(family, [])]
    if weight not in valid:
        raise ValueError(
            f"Invalid model weights {weight!r} for {family}. "
            f"Choose one of: {', '.join(valid)}"
        )
    key = "model_type" if family == FAMILY_SAM1 else "model_id"
    return {key: weight}


def layer_file_path(source):
    """Return the file path of a QGIS layer source string.

    QGIS appends provider options after a ``|`` (for example
    ``image.tif|layername=x``); those are stripped.

    Args:
        source: A QGIS layer ``source()`` string.

    Returns:
        The file path part of ``source``.
    """
    return str(source).split("|", 1)[0].strip()


def source_name(source_path):
    """Return the file name of a source raster for the ``source`` field.

    QGIS layer sources may carry provider options after a ``|`` (for example
    ``image.tif|layername=x``); those are dropped. Both ``/`` and ``\\`` are
    treated as separators so Windows paths work on any platform.

    Args:
        source_path: Path or QGIS layer source of the raster, or None.

    Returns:
        The file name, e.g. ``"my image.tif"``, or an empty string.
    """
    if not source_path:
        return ""
    path = str(source_path).split("|", 1)[0].strip()
    path = path.rstrip("/\\")
    return re.split(r"[/\\]", path)[-1]


def mask_attributes(source_path, prompt_type):
    """Return the attribute values written to every saved vector mask.

    Args:
        source_path: Path of the segmented source raster.
        prompt_type: Prompt type, e.g. ``"text"``, ``"point"`` or ``"box"``.

    Returns:
        A dict mapping field name to value, e.g.
        ``{"source": "image.tif", "prompt": "text"}``.
    """
    return {
        SOURCE_FIELD: source_name(source_path),
        PROMPT_FIELD: prompt_type or "",
    }


def output_extension(format_text):
    """Return the file extension for an Output tab format label.

    Args:
        format_text: Text of the output format combo, e.g.
            ``"Vector (GeoPackage)"``.

    Returns:
        ``".gpkg"``, ``".shp"`` or ``".tif"`` (the default).
    """
    for keyword, extension in _FORMAT_EXTENSIONS:
        if keyword in format_text:
            return extension
    return ".tif"


def source_directory(source_path):
    """Return the directory of a source raster if it exists.

    Args:
        source_path: Path or QGIS layer source of the raster, or None.

    Returns:
        The absolute directory path, or an empty string when the source is
        unset or its directory does not exist.
    """
    if not source_path:
        return ""
    path = str(source_path).split("|", 1)[0].strip()
    directory = os.path.dirname(os.path.abspath(path))
    return directory if os.path.isdir(directory) else ""


def _safe_token(text):
    """Make a short, file-name-safe token from free text.

    Args:
        text: Arbitrary text such as a prompt type.

    Returns:
        The text with runs of unsafe characters replaced by ``_``.
    """
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_")


def default_output_path(source_path, prompt_type, extension, ensure_unique=True):
    """Suggest an output path next to the source raster.

    The name is ``<source stem>_<prompt>_masks<extension>``; with
    ``ensure_unique`` a numeric suffix is added so an existing file is never
    overwritten.

    Args:
        source_path: Path of the segmented source raster.
        prompt_type: Prompt type used for the masks, or None.
        extension: Output extension including the dot, e.g. ``".gpkg"``.
        ensure_unique: Append ``_1``, ``_2``, ... if the path already exists.

    Returns:
        The suggested path, or an empty string when the source directory is
        unknown.
    """
    directory = source_directory(source_path)
    if not directory:
        return ""
    stem = os.path.splitext(source_name(source_path))[0] or "samgeo"
    parts = [stem]
    token = _safe_token(prompt_type) if prompt_type else ""
    if token:
        parts.append(token)
    parts.append("masks")
    base = os.path.join(directory, "_".join(parts))
    candidate = base + extension
    if ensure_unique:
        index = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{index}{extension}"
            index += 1
    return candidate


# Tiled text segmentation limits (pixels / percent).
TILE_SIZE_DEFAULT = 1024
TILE_OVERLAP_DEFAULT = 128
SCALE_MIN = 10
SCALE_MAX = 200


def is_geotiff(path):
    """Return True if a path looks like a GeoTIFF file.

    ``SamGeo3.generate_masks_tiled`` only accepts ``.tif``/``.tiff`` files.

    Args:
        path: File path, or None.

    Returns:
        True for a ``.tif`` or ``.tiff`` extension (any case).
    """
    if not path:
        return False
    return str(path).split("|", 1)[0].strip().lower().endswith((".tif", ".tiff"))


def validate_tile_settings(tile_size, overlap, scale=100):
    """Check tiled-mode settings before running a long job.

    Args:
        tile_size: Tile edge length in pixels.
        overlap: Overlap between neighboring tiles in pixels.
        scale: Processing scale in percent of the native resolution.

    Returns:
        An error message, or None when the settings are valid.
    """
    if tile_size <= 0:
        return "Tile size must be positive."
    if overlap < 0:
        return "Tile overlap cannot be negative."
    if tile_size <= overlap:
        return "Tile size must be greater than the tile overlap."
    if not SCALE_MIN <= scale <= SCALE_MAX:
        return f"Processing scale must be between {SCALE_MIN}% and {SCALE_MAX}%."
    return None


def tile_count(width, height, tile_size, overlap):
    """Return how many tiles ``generate_masks_tiled`` processes for an image.

    Mirrors the tiling formula in ``SamGeo3.generate_masks_tiled``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        tile_size: Tile edge length in pixels.
        overlap: Overlap between neighboring tiles in pixels.

    Returns:
        The number of tiles (at least 1).
    """
    step = tile_size - overlap
    n_x = max(1, (width - overlap + step - 1) // step)
    n_y = max(1, (height - overlap + step - 1) // step)
    return n_x * n_y


def scaled_size(width, height, scale):
    """Return the image size after resampling to a processing scale.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Processing scale in percent of the native resolution.

    Returns:
        A ``(width, height)`` tuple, each at least 1 pixel.
    """
    factor = scale / 100.0
    return max(1, round(width * factor)), max(1, round(height * factor))


def failed_tile_messages(log_text):
    """Extract per-tile failure messages from ``generate_masks_tiled`` output.

    ``SamGeo3.generate_masks_tiled`` catches errors per tile, prints
    ``Warning: Failed to process tile (x, y): <error>`` and carries on, so
    its captured output is the only record of skipped tiles.

    Args:
        log_text: Captured stdout of a ``generate_masks_tiled(verbose=True)``
            call, or None.

    Returns:
        A list of failure messages, one per failed tile.
    """
    if not log_text:
        return []
    return [
        _strip_prefix(line.strip(), "Warning: ")
        for line in log_text.splitlines()
        if "Failed to process tile" in line
    ]


def _strip_prefix(text, prefix):
    """Remove ``prefix`` from the start of ``text`` if present.

    ``str.removeprefix`` needs Python 3.9, and some QGIS 3.22 builds still
    ship Python 3.8.

    Args:
        text: The string to strip.
        prefix: The prefix to remove.

    Returns:
        ``text`` without ``prefix``.
    """
    return text[len(prefix) :] if text.startswith(prefix) else text


def geotransform_from_extent(x_min, y_max, extent_width, extent_height, width, height):
    """Build a north-up GDAL geotransform from a raster's extent and size.

    Used as a fallback when the raster's own geotransform cannot be read.

    Args:
        x_min: Left edge of the extent in map units.
        y_max: Top edge of the extent in map units.
        extent_width: Extent width in map units.
        extent_height: Extent height in map units.
        width: Raster width in pixels.
        height: Raster height in pixels.

    Returns:
        A 6-tuple GDAL geotransform.
    """
    return (x_min, extent_width / width, 0.0, y_max, 0.0, -extent_height / height)


def invert_geotransform(gt):
    """Invert a GDAL geotransform, including rotated or sheared ones.

    Args:
        gt: A 6-tuple ``(x0, a, b, y0, d, e)`` where
            ``x = x0 + a*col + b*row`` and ``y = y0 + d*col + e*row``.

    Returns:
        The inverse 6-tuple, mapping map coordinates to ``(col, row)``.

    Raises:
        ValueError: If the geotransform is not invertible.
    """
    x0, a, b, y0, d, e = gt
    det = a * e - b * d
    if det == 0:
        raise ValueError("Geotransform is not invertible.")
    ia, ib = e / det, -b / det
    id_, ie = -d / det, a / det
    return (-(ia * x0 + ib * y0), ia, ib, -(id_ * x0 + ie * y0), id_, ie)


def map_to_pixel(inv_gt, x, y):
    """Convert map coordinates to fractional pixel coordinates.

    Args:
        inv_gt: An inverse geotransform from :func:`invert_geotransform`.
        x: Map x coordinate in the raster's CRS.
        y: Map y coordinate in the raster's CRS.

    Returns:
        A ``[col, row]`` list of floats.
    """
    c0, ca, cb, r0, rd, re_ = inv_gt
    return [c0 + ca * x + cb * y, r0 + rd * x + re_ * y]


def rect_to_pixel_box(inv_gt, x_min, y_min, x_max, y_max):
    """Convert a map rectangle to a pixel-space XYXY box.

    All four corners are transformed, so a rotated raster still gets the box
    that encloses the drawn rectangle.

    Args:
        inv_gt: An inverse geotransform from :func:`invert_geotransform`.
        x_min: Rectangle left edge in map units.
        y_min: Rectangle bottom edge in map units.
        x_max: Rectangle right edge in map units.
        y_max: Rectangle top edge in map units.

    Returns:
        ``[col_min, row_min, col_max, row_max]``.
    """
    corners = [
        map_to_pixel(inv_gt, x, y) for x in (x_min, x_max) for y in (y_min, y_max)
    ]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    return [min(cols), min(rows), max(cols), max(rows)]
