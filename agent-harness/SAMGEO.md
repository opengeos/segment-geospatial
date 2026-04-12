# SAMGEO: Agent Harness SOP

## Software Overview

**segment-geospatial** (samgeo) is a Python package for segmenting geospatial data
using Meta AI's Segment Anything Model (SAM) family. It wraps SAM v1, SAM 2, SAM 3,
FastSAM, HQ-SAM, and LangSAM with geospatial-aware I/O (GeoTIFF, GeoPackage, etc.).

## Backend

The backend is the `samgeo` Python library itself. Unlike GUI applications that need
a headless CLI invocation, samgeo is already a Python library — the CLI harness imports
and calls its classes/functions directly.

**Key classes:**
- `SamGeo` (v1) — `samgeo.samgeo.SamGeo`
- `SamGeo2` (v2) — `samgeo.samgeo2.SamGeo2`
- `SamGeo3` (v3) — `samgeo.samgeo3.SamGeo3`
- `LangSAM` — `samgeo.text_sam.LangSAM`

**Key utility functions** (from `samgeo.common`):
- `tms_to_geotiff()` — Download TMS tiles as GeoTIFF
- `raster_to_vector()` / `raster_to_gpkg()` / `raster_to_shp()` / `raster_to_geojson()`
- `reproject()`, `split_raster()`, `image_to_cog()`
- `get_profile()`, `get_basemaps()`

## Data Model

**Project state** is a JSON file tracking:
- Source image path, CRS, bounds
- Active model type and parameters
- Generated mask paths
- Vector output paths
- Operation history (for undo/redo)

**File formats:**
- Input: GeoTIFF, PNG, JPG, NumPy arrays, URLs
- Mask output: GeoTIFF (raster masks)
- Vector output: GeoPackage (.gpkg), Shapefile (.shp), GeoJSON (.geojson)
- Project: JSON (.json)

## CLI Command Groups

| Group | Purpose |
|-------|---------|
| `project` | Create, open, save, inspect projects |
| `model` | List, download, inspect SAM models |
| `segment` | Automatic, point, box, and text segmentation |
| `data` | Download tiles, inspect rasters, reproject, split |
| `vector` | Convert masks to vectors, inspect, filter |
| `export` | Export masks and vectors to various formats |
| `session` | Undo/redo, history, session state |

## Dependencies

- `segment-geospatial` (the package itself — hard dependency)
- `click` (CLI framework)
- `prompt_toolkit` (REPL)
- PyTorch + SAM model weights (downloaded on first use)
