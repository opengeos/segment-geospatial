# cli-anything-samgeo

CLI harness for [segment-geospatial](https://github.com/opengeos/segment-geospatial) — segment geospatial imagery using SAM models from the command line.

## Prerequisites

- Python 3.10+
- segment-geospatial: `pip install segment-geospatial[all]`
- PyTorch with CUDA (recommended) or CPU

## Installation

```bash
cd agent-harness
pip install -e .
```

This installs the `cli-anything-samgeo` command in your PATH.

## Quick Start

```bash
# Create a project
cli-anything-samgeo project new -n my-seg -o project.json -s image.tif

# Run automatic segmentation
cli-anything-samgeo --project project.json segment automatic -o masks.tif

# Convert masks to vectors
cli-anything-samgeo --project project.json vector convert masks.tif output.gpkg

# Export as GeoJSON
cli-anything-samgeo --project project.json export render output.geojson -f geojson

# All commands support --json for machine-readable output
cli-anything-samgeo --json model list
```

## Command Groups

| Command | Description |
|---------|-------------|
| `project` | Create, open, inspect projects |
| `model` | List, inspect, check SAM models |
| `segment` | Automatic, point, box, text segmentation |
| `data` | Download tiles, raster info, reproject, split |
| `vector` | Convert masks to vectors, inspect, filter |
| `export` | Export masks to various formats |
| `session` | Session status and history |

## Interactive REPL

Run without arguments to enter the interactive REPL:

```bash
cli-anything-samgeo
```

## JSON Output

Add `--json` before any command for machine-readable output:

```bash
cli-anything-samgeo --json data info image.tif
cli-anything-samgeo --json model list
```

## Using with Claude Code

This CLI ships with a `SKILL.md` file that lets Claude Code discover and use all
commands automatically. There are two ways to enable it.

### Option 1: Add SKILL.md to your CLAUDE.md

Append a reference to the skill file in your project or user `CLAUDE.md`:

```markdown
# In your CLAUDE.md
Read the skill file at /path/to/agent-harness/cli_anything/samgeo/skills/SKILL.md
for the full cli-anything-samgeo command reference. Use `--json` for all
cli-anything-samgeo commands so output is machine-readable.
```

Replace `/path/to/` with the actual absolute path. Claude Code reads `CLAUDE.md`
at the start of every conversation, so it will know the CLI exists and how to
call it.

### Option 2: Point Claude Code at the skill on the fly

In any Claude Code conversation, paste:

```
Read agent-harness/cli_anything/samgeo/skills/SKILL.md and use that CLI
to segment this satellite image.
```

Claude Code will read the skill file, learn the command syntax, and start
using `cli-anything-samgeo` with `--json` output.

### Example Claude Code session

Once Claude Code knows about the skill, you can give it natural-language tasks:

```
> Segment all buildings in satellite.tif and export the results as a GeoPackage.

# Claude Code will run:
cli-anything-samgeo --json project new -n buildings -o project.json -s satellite.tif -t sam2
cli-anything-samgeo --json --project project.json segment automatic -o masks.tif
cli-anything-samgeo --json vector convert masks.tif buildings.gpkg
```

```
> Download OpenStreetMap tiles for downtown Portland and tell me about the image.

# Claude Code will run:
cli-anything-samgeo --json data download-tiles -o portland.tif -b "-122.68,45.51,-122.66,45.53" -z 17
cli-anything-samgeo --json data info portland.tif
```

### Tips for Claude Code usage

- The `--json` flag is essential — it gives Claude Code structured output it can
  parse and reason about, rather than human-formatted tables.
- The `--project` flag must appear *before* the command group (e.g.,
  `--project proj.json segment automatic`, not `segment automatic --project proj.json`).
- Claude Code can chain multiple commands in sequence to build full pipelines
  (download → segment → vectorize → export).
- Use `model check sam2` to let Claude Code verify a model backend is installed
  before attempting segmentation.

## Running Tests

```bash
cd agent-harness
python -m pytest cli_anything/samgeo/tests/ -v -s
```

## Supported Models

| Type | Models | Install |
|------|--------|---------|
| SAM v1 | vit_h, vit_l, vit_b | `pip install segment-geospatial` |
| SAM 2 | hiera-tiny/small/base-plus/large | `pip install segment-geospatial[samgeo2]` |
| SAM 3 | facebook/sam3 | `pip install segment-geospatial[samgeo3]` |
| FastSAM | FastSAM-x, FastSAM-s | `pip install segment-geospatial[fast]` |
| HQ-SAM | vit_h, vit_l, vit_b, vit_tiny | `pip install segment-geospatial[hq]` |
| LangSAM | text-based (SAM2 backend) | `pip install segment-geospatial[text]` |
