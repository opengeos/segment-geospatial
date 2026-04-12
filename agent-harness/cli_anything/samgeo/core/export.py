"""Export operations: save masks and vectors to various formats.

Calls the real samgeo library for all exports.
"""

import os

from cli_anything.samgeo.core.project import add_history_entry, set_vectors
from cli_anything.samgeo.utils.samgeo_backend import get_common_module, get_rasterio


EXPORT_FORMATS = {
    "geotiff": {
        "extension": ".tif",
        "description": "GeoTIFF raster format",
        "type": "raster",
    },
    "cog": {
        "extension": ".tif",
        "description": "Cloud-Optimized GeoTIFF",
        "type": "raster",
    },
    "gpkg": {
        "extension": ".gpkg",
        "description": "GeoPackage vector format",
        "type": "vector",
    },
    "shp": {
        "extension": ".shp",
        "description": "ESRI Shapefile",
        "type": "vector",
    },
    "geojson": {
        "extension": ".geojson",
        "description": "GeoJSON vector format",
        "type": "vector",
    },
    "png": {
        "extension": ".png",
        "description": "PNG image (no georeferencing)",
        "type": "image",
    },
}


def list_formats():
    """List available export formats.

    Returns:
        list: List of format info dicts.
    """
    return [{"format": key, **value} for key, value in EXPORT_FORMATS.items()]


def export_masks(
    project, output, fmt="geotiff", simplify_tolerance=None, overwrite=False
):
    """Export project masks to the specified format.

    Args:
        project: The project state dict.
        output: Output file path.
        fmt: Export format key.
        simplify_tolerance: Geometry simplification tolerance (vector formats only).
        overwrite: Whether to overwrite existing files.

    Returns:
        dict: Result with output path, format, and file size.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(
            f"Unknown export format: {fmt}. "
            f"Available: {', '.join(EXPORT_FORMATS.keys())}"
        )

    if not project.get("masks") or not project["masks"].get("path"):
        raise ValueError(
            "No masks generated. Run segmentation first "
            "(e.g., 'segment automatic -o masks.tif')."
        )

    mask_path = project["masks"]["path"]
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    output = os.path.abspath(output)
    if os.path.exists(output) and not overwrite:
        raise FileExistsError(
            f"Output file exists: {output}. Use --overwrite to replace."
        )

    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    format_info = EXPORT_FORMATS[fmt]

    if format_info["type"] == "raster":
        _export_raster(mask_path, output, fmt)
    elif format_info["type"] == "vector":
        _export_vector(project, mask_path, output, fmt, simplify_tolerance)
    elif format_info["type"] == "image":
        _export_image(mask_path, output)

    file_size = os.path.getsize(output) if os.path.exists(output) else 0

    result = {
        "output": output,
        "format": fmt,
        "file_size": file_size,
        "source_masks": mask_path,
    }

    add_history_entry(
        project, "export", params={"format": fmt, "output": output}, result=result
    )

    return result


def _export_raster(mask_path, output, fmt):
    """Export masks as a raster format.

    Args:
        mask_path: Source mask raster.
        output: Output path.
        fmt: Format key ('geotiff' or 'cog').
    """
    common = get_common_module()
    rasterio = get_rasterio()

    if fmt == "cog":
        common.image_to_cog(source=mask_path, dst_path=output)
    else:
        # Copy with rasterio
        with rasterio.open(mask_path) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff")
            with rasterio.open(output, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)


def _export_vector(project, mask_path, output, fmt, simplify_tolerance):
    """Export masks as a vector format.

    Args:
        project: The project state dict.
        mask_path: Source mask raster.
        output: Output vector path.
        fmt: Export format key ('gpkg', 'shp', 'geojson').
        simplify_tolerance: Geometry simplification tolerance.
    """
    common = get_common_module()

    # Use the requested format, not the file extension, to pick the writer
    if fmt == "gpkg":
        common.raster_to_gpkg(mask_path, output, simplify_tolerance=simplify_tolerance)
    elif fmt == "shp":
        common.raster_to_shp(mask_path, output, simplify_tolerance=simplify_tolerance)
    elif fmt == "geojson":
        common.raster_to_geojson(
            mask_path, output, simplify_tolerance=simplify_tolerance
        )
    else:
        common.raster_to_vector(
            mask_path, output, simplify_tolerance=simplify_tolerance
        )

    # Update project vectors
    try:
        import geopandas as gpd

        gdf = gpd.read_file(output)
        set_vectors(project, output, feature_count=len(gdf))
    except Exception:
        set_vectors(project, output)


def _export_image(mask_path, output):
    """Export masks as a PNG image.

    Args:
        mask_path: Source mask raster.
        output: Output image path.
    """
    rasterio = get_rasterio()
    import numpy as np

    with rasterio.open(mask_path) as src:
        data = src.read(1)

    from PIL import Image

    if data.max() > 0:
        normalized = (data / data.max() * 255).astype(np.uint8)
    else:
        normalized = data.astype(np.uint8)

    img = Image.fromarray(normalized)
    img.save(output)
