"""Data operations: download tiles, inspect rasters, reproject, split."""

import os

from cli_anything.samgeo.utils.samgeo_backend import (
    get_common_module,
    get_rasterio,
)


def raster_info(path):
    """Get information about a raster file.

    Args:
        path: Path to the raster file.

    Returns:
        dict: Raster metadata.
    """
    rasterio = get_rasterio()
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        info = {
            "path": path,
            "driver": src.driver,
            "crs": str(src.crs) if src.crs else None,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]) if src.dtypes else None,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "transform": list(src.transform)[:6],
            "nodata": src.nodata,
            "file_size": os.path.getsize(path),
        }

    return info


def download_tiles(
    output,
    bbox,
    zoom=None,
    resolution=None,
    source="OpenStreetMap",
    crs="EPSG:3857",
    to_cog=False,
    **kwargs,
):
    """Download TMS tiles as a GeoTIFF.

    Args:
        output: Output file path.
        bbox: Bounding box as [west, south, east, north].
        zoom: Zoom level. If None, auto-detected from resolution.
        resolution: Resolution in meters. Alternative to zoom.
        source: Tile source name or URL.
        crs: Output CRS.
        to_cog: Whether to convert to Cloud-Optimized GeoTIFF.
        **kwargs: Additional kwargs.

    Returns:
        dict: Result with output path and file size.
    """
    common = get_common_module()
    output = os.path.abspath(output)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    common.tms_to_geotiff(
        output=output,
        bbox=bbox,
        zoom=zoom,
        resolution=resolution,
        source=source,
        crs=crs,
        to_cog=to_cog,
        **kwargs,
    )

    result = {
        "output": output,
        "file_size": os.path.getsize(output) if os.path.exists(output) else 0,
        "bbox": bbox,
        "zoom": zoom,
        "source": source,
        "crs": crs,
    }

    return result


def reproject_raster(
    input_path, output_path, dst_crs="EPSG:4326", resampling="nearest", to_cog=True
):
    """Reproject a raster file to a new CRS.

    Args:
        input_path: Input raster path.
        output_path: Output raster path.
        dst_crs: Target CRS.
        resampling: Resampling method.
        to_cog: Whether to output as COG.

    Returns:
        dict: Result with output path and new CRS.
    """
    common = get_common_module()
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    common.reproject(
        image=input_path,
        output=output_path,
        dst_crs=dst_crs,
        resampling=resampling,
        to_cog=to_cog,
    )

    result = {
        "input": input_path,
        "output": output_path,
        "dst_crs": dst_crs,
        "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
    }

    return result


def split_raster_tiles(input_path, out_dir, tile_size=256, overlap=0):
    """Split a raster into tiles.

    Args:
        input_path: Input raster path.
        out_dir: Output directory for tiles.
        tile_size: Tile size in pixels (int or [width, height]).
        overlap: Overlap in pixels.

    Returns:
        dict: Result with tile count and output directory.
    """
    common = get_common_module()
    input_path = os.path.abspath(input_path)
    out_dir = os.path.abspath(out_dir)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    os.makedirs(out_dir, exist_ok=True)

    common.split_raster(
        filename=input_path,
        out_dir=out_dir,
        tile_size=tile_size,
        overlap=overlap,
    )

    tiles = [f for f in os.listdir(out_dir) if f.endswith((".tif", ".tiff"))]

    result = {
        "input": input_path,
        "output_dir": out_dir,
        "tile_count": len(tiles),
        "tile_size": tile_size,
        "overlap": overlap,
    }

    return result


def image_to_cog(source, output=None, profile="deflate"):
    """Convert a raster to Cloud-Optimized GeoTIFF.

    Args:
        source: Input raster path.
        output: Output path. If None, writes to a new *_cog.tif file
            alongside the source.
        profile: COG profile.

    Returns:
        dict: Result with output path.
    """
    common = get_common_module()
    source = os.path.abspath(source)

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source raster not found: {source}")

    common.image_to_cog(source=source, dst_path=output, profile=profile)

    out = output or source
    result = {
        "output": os.path.abspath(out),
        "file_size": os.path.getsize(out) if os.path.exists(out) else 0,
        "profile": profile,
    }

    return result


def list_basemaps(free_only=True):
    """List available TMS basemap sources.

    Args:
        free_only: Only list free tile services.

    Returns:
        list: List of basemap names.
    """
    common = get_common_module()
    basemaps = common.get_basemaps(free_only=free_only)
    return sorted(basemaps.keys()) if isinstance(basemaps, dict) else sorted(basemaps)
