"""Vector operations: convert raster masks to vectors, inspect, filter."""

import os

from cli_anything.samgeo.utils.samgeo_backend import (
    get_common_module,
    get_geopandas,
)


def raster_to_vector(source, output, simplify_tolerance=None, dst_crs=None):
    """Convert a raster mask to a vector file.

    The output format is determined by the file extension:
    - .gpkg -> GeoPackage
    - .shp -> Shapefile
    - .geojson -> GeoJSON

    Args:
        source: Path to the raster mask file.
        output: Path to the output vector file.
        simplify_tolerance: Geometry simplification tolerance.
        dst_crs: Target CRS for the output vectors.

    Returns:
        dict: Result with output path and feature count.
    """
    common = get_common_module()
    source = os.path.abspath(source)
    output = os.path.abspath(output)

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source raster not found: {source}")

    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ext = os.path.splitext(output)[1].lower()

    kwargs = {"simplify_tolerance": simplify_tolerance}
    if dst_crs is not None:
        kwargs["dst_crs"] = dst_crs

    if ext == ".gpkg":
        common.raster_to_gpkg(source, output, **kwargs)
    elif ext == ".shp":
        common.raster_to_shp(source, output, **kwargs)
    elif ext == ".geojson":
        common.raster_to_geojson(source, output, **kwargs)
    else:
        common.raster_to_vector(source, output, **kwargs)

    gpd = get_geopandas()
    gdf = gpd.read_file(output)
    feature_count = len(gdf)

    result = {
        "output": output,
        "format": ext.lstrip("."),
        "feature_count": feature_count,
        "file_size": os.path.getsize(output) if os.path.exists(output) else 0,
    }

    return result


def vector_info(path):
    """Get information about a vector file.

    Args:
        path: Path to the vector file.

    Returns:
        dict: Vector metadata including feature count, CRS, bounds, columns.
    """
    gpd = get_geopandas()
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector file not found: {path}")

    gdf = gpd.read_file(path)

    info = {
        "path": path,
        "feature_count": len(gdf),
        "crs": str(gdf.crs) if gdf.crs else None,
        "bounds": {
            "minx": gdf.total_bounds[0],
            "miny": gdf.total_bounds[1],
            "maxx": gdf.total_bounds[2],
            "maxy": gdf.total_bounds[3],
        },
        "columns": list(gdf.columns),
        "geometry_types": list(gdf.geometry.geom_type.unique()),
        "file_size": os.path.getsize(path),
    }

    return info


def filter_vectors(
    input_path, output_path, min_area=None, max_area=None, column=None, value=None
):
    """Filter vector features by area or attribute.

    Args:
        input_path: Path to the input vector file.
        output_path: Path to the output vector file.
        min_area: Minimum geometry area.
        max_area: Maximum geometry area.
        column: Column name to filter by.
        value: Value to match in the column.

    Returns:
        dict: Result with feature counts before and after filtering.
    """
    gpd = get_geopandas()
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input vector not found: {input_path}")

    gdf = gpd.read_file(input_path)
    original_count = len(gdf)

    if min_area is not None:
        gdf = gdf[gdf.geometry.area >= min_area]

    if max_area is not None:
        gdf = gdf[gdf.geometry.area <= max_area]

    if column is not None and value is not None:
        if column not in gdf.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(gdf.columns)}"
            )
        gdf = gdf[gdf[column] == value]

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    gdf.to_file(output_path)

    result = {
        "input": input_path,
        "output": output_path,
        "original_count": original_count,
        "filtered_count": len(gdf),
        "removed_count": original_count - len(gdf),
    }

    return result
