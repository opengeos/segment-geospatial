---
name: "cli-anything-samgeo"
description: "CLI harness for segment-geospatial — segment satellite/aerial imagery with SAM models from the command line"
---

# cli-anything-samgeo

Segment geospatial imagery (satellite, aerial, drone) using Meta AI's Segment
Anything Model (SAM) family from the command line. Supports SAM v1, SAM 2,
SAM 3, FastSAM, HQ-SAM, and LangSAM (text-based).

## Installation

Requires Python >= 3.10.

```bash
pip install segment-geospatial[all]
cd agent-harness && pip install -e .
```

## Syntax

```
cli-anything-samgeo [--json] [--project PATH] COMMAND [ARGS...]
```

Global flags:
- `--json` — Machine-readable JSON output (use this for agent consumption)
- `--project PATH` — Path to project JSON file (required for project-scoped commands)

## Command Groups

### project — Project management
```bash
cli-anything-samgeo project new -n NAME -o PATH [-s SOURCE] [-t MODEL_TYPE] [-m MODEL_ID]
cli-anything-samgeo --project PATH project info
cli-anything-samgeo --project PATH project set-source IMAGE_PATH
cli-anything-samgeo --project PATH project history [-l LIMIT]
```

### model — Model management
```bash
cli-anything-samgeo model list [-t TYPE]
cli-anything-samgeo model info MODEL_TYPE [-m MODEL_ID]
cli-anything-samgeo model check MODEL_TYPE
```

Model types: `sam`, `sam2`, `sam3`, `fast_sam`, `hq_sam`, `text_sam`

### segment — Segmentation operations
```bash
cli-anything-samgeo --project PATH segment automatic -o OUTPUT [--min-size N] [--max-size N]
cli-anything-samgeo --project PATH segment points -o OUTPUT -c "x1,y1;x2,y2" -l "1,0"
cli-anything-samgeo --project PATH segment boxes -o OUTPUT -b "x1,y1,x2,y2"
cli-anything-samgeo --project PATH segment text -o OUTPUT -t "buildings"
```

### data — Data operations
```bash
cli-anything-samgeo data info RASTER_PATH
cli-anything-samgeo data download-tiles -o OUTPUT -b "west,south,east,north" [-z ZOOM] [-s SOURCE]
cli-anything-samgeo data reproject INPUT OUTPUT [--crs EPSG:CODE]
cli-anything-samgeo data split INPUT OUT_DIR [--tile-size N] [--overlap N]
cli-anything-samgeo data basemaps [--all]
```

### vector — Vector operations
```bash
cli-anything-samgeo vector convert RASTER_MASK OUTPUT [--simplify TOLERANCE]
cli-anything-samgeo vector info VECTOR_PATH
cli-anything-samgeo vector filter INPUT OUTPUT [--min-area N] [--max-area N]
```

### export — Export operations
```bash
cli-anything-samgeo --project PATH export render OUTPUT [-f FORMAT] [--overwrite]
cli-anything-samgeo export formats
```

Formats: `geotiff`, `cog`, `gpkg`, `shp`, `geojson`, `png`

### session — Session management
```bash
cli-anything-samgeo --project PATH session status
cli-anything-samgeo --project PATH session history [-l LIMIT]
```

## Example Workflows

### Satellite image segmentation pipeline
```bash
# Download tiles
cli-anything-samgeo --json data download-tiles -o image.tif -b "-122.42,37.77,-122.40,37.79" -z 17

# Create project
cli-anything-samgeo --json project new -n urban-seg -o project.json -s image.tif -t sam2

# Run automatic segmentation
cli-anything-samgeo --json --project project.json segment automatic -o masks.tif

# Convert to GeoPackage
cli-anything-samgeo --json vector convert masks.tif buildings.gpkg --simplify 1.0

# Export as GeoJSON
cli-anything-samgeo --json --project project.json export render output.geojson -f geojson --overwrite
```

### Text-based segmentation
```bash
cli-anything-samgeo --json project new -n text-seg -o proj.json -s image.tif -t text_sam
cli-anything-samgeo --json --project proj.json segment text -o masks.tif -t "buildings"
cli-anything-samgeo --json vector convert masks.tif buildings.gpkg
```

## Agent Guidance

- Always use `--json` for machine-readable output
- Create a project first before running segmentation commands
- The `--project` flag must come before the command group
- Check model availability with `model check TYPE` before segmenting
- Use `data info` to inspect rasters before processing
- Export formats are auto-detected from file extension in `vector convert`
- For large images, use `--batch` with `segment automatic`
