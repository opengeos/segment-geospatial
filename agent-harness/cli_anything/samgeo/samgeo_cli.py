"""Main CLI entry point for cli-anything-samgeo.

Click-based CLI with subcommand groups and REPL mode.
Every command supports --json for machine-readable output.
"""

import json
import os
import shlex
import sys

import click

from cli_anything.samgeo import __version__
from cli_anything.samgeo.core import project as proj_mod
from cli_anything.samgeo.core import model as model_mod
from cli_anything.samgeo.core import session as session_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output(data, as_json=False):
    """Print output as JSON or human-readable text.

    Args:
        data: Dict or list to output.
        as_json: Whether to output as JSON.
    """
    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    click.echo(f"{k}:")
                    for kk, vv in v.items():
                        click.echo(f"  {kk}: {vv}")
                elif isinstance(v, list):
                    click.echo(f"{k}: [{len(v)} items]")
                else:
                    click.echo(f"{k}: {v}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    line = "  ".join(f"{k}={v}" for k, v in item.items())
                    click.echo(f"  {line}")
                else:
                    click.echo(f"  {item}")


def _load_project(ctx):
    """Load the project from the context.

    Args:
        ctx: Click context.

    Returns:
        dict: Project state dict.
    """
    project_path = ctx.obj.get("project_path")
    if not project_path:
        raise click.UsageError(
            "No project specified. Use --project <path> or create one with 'project new'."
        )
    return proj_mod.open_project(project_path)


def _save_project(ctx, project):
    """Save the project back to disk.

    Args:
        ctx: Click context.
        project: Project state dict.
    """
    project_path = ctx.obj.get("project_path")
    if project_path:
        proj_mod.save_project(project, project_path)


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option("--json", "as_json", is_flag=True, help="Output in JSON format.")
@click.option(
    "--project", "project_path", type=click.Path(), help="Path to project JSON file."
)
@click.version_option(version=__version__, prog_name="cli-anything-samgeo")
@click.pass_context
def cli(ctx, as_json, project_path):
    """cli-anything-samgeo: CLI harness for segment-geospatial.

    Segment geospatial imagery using SAM models from the command line.
    Run without a subcommand to enter the interactive REPL.
    """
    ctx.ensure_object(dict)
    ctx.obj["as_json"] = as_json
    if project_path is not None:
        ctx.obj["project_path"] = project_path
    elif "project_path" not in ctx.obj:
        ctx.obj["project_path"] = None

    if ctx.invoked_subcommand is None:
        ctx.invoke(repl, project_path=project_path)


# ---------------------------------------------------------------------------
# Project commands
# ---------------------------------------------------------------------------


@cli.group()
@click.pass_context
def project(ctx):
    """Project management commands."""
    pass


@project.command("new")
@click.option("-n", "--name", required=True, help="Project name.")
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output project JSON path.",
)
@click.option("-s", "--source", type=click.Path(exists=True), help="Source image path.")
@click.option(
    "-t", "--model-type", default="sam2", help="Model type (sam, sam2, sam3, etc.)."
)
@click.option("-m", "--model-id", default=None, help="Model ID.")
@click.option("-d", "--device", default=None, help="Compute device (cpu, cuda, mps).")
@click.pass_context
def project_new(ctx, name, output_path, source, model_type, model_id, device):
    """Create a new segmentation project."""
    p = proj_mod.create_project(
        name=name,
        source_path=source,
        model_type=model_type,
        model_id=model_id,
        device=device,
        output_path=output_path,
    )
    # Store the project path so subsequent REPL commands can find it
    ctx.obj["project_path"] = os.path.abspath(output_path)
    result = {
        "status": "created",
        "path": os.path.abspath(output_path),
        **proj_mod.get_project_info(p),
    }
    _output(result, ctx.obj.get("as_json"))


@project.command("info")
@click.pass_context
def project_info(ctx):
    """Show project information."""
    p = _load_project(ctx)
    _output(proj_mod.get_project_info(p), ctx.obj.get("as_json"))


@project.command("set-source")
@click.argument("source_path", type=click.Path(exists=True))
@click.pass_context
def project_set_source(ctx, source_path):
    """Set or update the source image for the project."""
    p = _load_project(ctx)
    src = proj_mod.set_source(p, source_path)
    _save_project(ctx, p)
    _output({"status": "source_updated", **src}, ctx.obj.get("as_json"))


@project.command("history")
@click.option("-l", "--limit", default=10, type=int, help="Number of entries to show.")
@click.pass_context
def project_history(ctx, limit):
    """Show project operation history."""
    p = _load_project(ctx)
    history = list(reversed(p.get("history", [])))[:limit]
    if ctx.obj.get("as_json"):
        _output(history, True)
    else:
        if not history:
            click.echo("No history entries.")
        else:
            for i, entry in enumerate(history):
                click.echo(f"  [{i+1}] {entry['action']} at {entry['timestamp']}")
                if entry.get("params"):
                    for k, v in entry["params"].items():
                        click.echo(f"       {k}: {v}")


# ---------------------------------------------------------------------------
# Model commands
# ---------------------------------------------------------------------------


@cli.group()
@click.pass_context
def model(ctx):
    """Model management commands."""
    pass


@model.command("list")
@click.option("-t", "--type", "model_type", default=None, help="Filter by model type.")
@click.pass_context
def model_list(ctx, model_type):
    """List available SAM models."""
    models = model_mod.list_models()
    if model_type:
        models = [m for m in models if m["type"] == model_type]

    if ctx.obj.get("as_json"):
        _output(models, True)
    else:
        headers = ["Type", "Model ID", "Name", "Params"]
        rows = [[m["type"], m["model_id"], m["name"], m["params"]] for m in models]
        from cli_anything.samgeo.utils.repl_skin import ReplSkin

        skin = ReplSkin("samgeo")
        skin.table(headers, rows)


@model.command("info")
@click.argument("model_type")
@click.option("-m", "--model-id", default=None, help="Specific model ID.")
@click.pass_context
def model_info_cmd(ctx, model_type, model_id):
    """Show model details."""
    info = model_mod.get_model_info(model_type, model_id)
    _output(info, ctx.obj.get("as_json"))


@model.command("check")
@click.argument("model_type")
@click.pass_context
def model_check(ctx, model_type):
    """Check if a model type is installed."""
    result = model_mod.check_model_available(model_type)
    _output(result, ctx.obj.get("as_json"))


# ---------------------------------------------------------------------------
# Segment commands
# ---------------------------------------------------------------------------


@cli.group()
@click.pass_context
def segment(ctx):
    """Segmentation commands."""
    pass


@segment.command("automatic")
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output mask path."
)
@click.option(
    "--foreground/--no-foreground", default=True, help="Generate foreground masks."
)
@click.option(
    "--unique/--no-unique", default=True, help="Assign unique values to masks."
)
@click.option("--min-size", default=0, type=int, help="Minimum mask area in pixels.")
@click.option("--max-size", default=None, type=int, help="Maximum mask area in pixels.")
@click.option(
    "--batch/--no-batch", default=False, help="Use batch mode for large images."
)
@click.pass_context
def segment_automatic(ctx, output, foreground, unique, min_size, max_size, batch):
    """Run automatic mask generation."""
    p = _load_project(ctx)
    from cli_anything.samgeo.core.segment import automatic_segment

    result = automatic_segment(
        p,
        output,
        foreground=foreground,
        unique=unique,
        min_size=min_size,
        max_size=max_size,
        batch=batch,
    )
    _save_project(ctx, p)
    _output(result, ctx.obj.get("as_json"))


@segment.command("points")
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output mask path."
)
@click.option(
    "-c", "--coords", required=True, help="Point coords as 'x1,y1;x2,y2;...'."
)
@click.option("-l", "--labels", required=True, help="Point labels as '1,0,1,...'.")
@click.option(
    "--multimask/--no-multimask", default=False, help="Output multiple masks."
)
@click.pass_context
def segment_points(ctx, output, coords, labels, multimask):
    """Run prediction with point prompts."""
    p = _load_project(ctx)
    point_coords = [list(map(float, c.split(","))) for c in coords.split(";")]
    point_labels = list(map(int, labels.split(",")))

    from cli_anything.samgeo.core.segment import predict_points

    result = predict_points(
        p,
        point_coords,
        point_labels,
        output,
        multimask_output=multimask,
    )
    _save_project(ctx, p)
    _output(result, ctx.obj.get("as_json"))


@segment.command("boxes")
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output mask path."
)
@click.option("-b", "--box", required=True, help="Bounding box as 'x1,y1,x2,y2'.")
@click.option(
    "--multimask/--no-multimask", default=False, help="Output multiple masks."
)
@click.pass_context
def segment_boxes(ctx, output, box, multimask):
    """Run prediction with bounding box prompts."""
    p = _load_project(ctx)
    boxes = [list(map(float, box.split(",")))]

    from cli_anything.samgeo.core.segment import predict_boxes

    result = predict_boxes(p, boxes, output, multimask_output=multimask)
    _save_project(ctx, p)
    _output(result, ctx.obj.get("as_json"))


@segment.command("text")
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output mask path."
)
@click.option(
    "-t", "--text", "text_prompt", required=True, help="Text description of objects."
)
@click.option(
    "--box-threshold", default=0.24, type=float, help="Box confidence threshold."
)
@click.option(
    "--text-threshold", default=0.24, type=float, help="Text confidence threshold."
)
@click.pass_context
def segment_text(ctx, output, text_prompt, box_threshold, text_threshold):
    """Run text-based segmentation (LangSAM)."""
    p = _load_project(ctx)
    from cli_anything.samgeo.core.segment import text_segment

    result = text_segment(
        p,
        text_prompt,
        output,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    _save_project(ctx, p)
    _output(result, ctx.obj.get("as_json"))


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------


@cli.group()
@click.pass_context
def data(ctx):
    """Data operations."""
    pass


@data.command("info")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def data_info(ctx, path):
    """Show raster file information."""
    from cli_anything.samgeo.core.data import raster_info

    info = raster_info(path)
    _output(info, ctx.obj.get("as_json"))


@data.command("download-tiles")
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output GeoTIFF path."
)
@click.option(
    "-b", "--bbox", required=True, help="Bounding box as 'west,south,east,north'."
)
@click.option("-z", "--zoom", default=None, type=int, help="Zoom level.")
@click.option(
    "-s", "--source", default="OpenStreetMap", help="Tile source name or URL."
)
@click.option("--crs", default="EPSG:3857", help="Output CRS.")
@click.option(
    "--cog/--no-cog", default=False, help="Convert to Cloud-Optimized GeoTIFF."
)
@click.pass_context
def data_download_tiles(ctx, output, bbox, zoom, source, crs, cog):
    """Download TMS tiles as GeoTIFF."""
    from cli_anything.samgeo.core.data import download_tiles

    bbox_list = list(map(float, bbox.split(",")))
    result = download_tiles(
        output, bbox_list, zoom=zoom, source=source, crs=crs, to_cog=cog
    )
    _output(result, ctx.obj.get("as_json"))


@data.command("reproject")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--crs", default="EPSG:4326", help="Target CRS.")
@click.option("--resampling", default="nearest", help="Resampling method.")
@click.pass_context
def data_reproject(ctx, input_path, output_path, crs, resampling):
    """Reproject a raster to a new CRS."""
    from cli_anything.samgeo.core.data import reproject_raster

    result = reproject_raster(
        input_path, output_path, dst_crs=crs, resampling=resampling
    )
    _output(result, ctx.obj.get("as_json"))


@data.command("split")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path())
@click.option("--tile-size", default=256, type=int, help="Tile size in pixels.")
@click.option("--overlap", default=0, type=int, help="Overlap in pixels.")
@click.pass_context
def data_split(ctx, input_path, out_dir, tile_size, overlap):
    """Split a raster into tiles."""
    from cli_anything.samgeo.core.data import split_raster_tiles

    result = split_raster_tiles(
        input_path, out_dir, tile_size=tile_size, overlap=overlap
    )
    _output(result, ctx.obj.get("as_json"))


@data.command("basemaps")
@click.option(
    "--all", "show_all", is_flag=True, help="Show all basemaps including paid ones."
)
@click.pass_context
def data_basemaps(ctx, show_all):
    """List available TMS basemap sources."""
    from cli_anything.samgeo.core.data import list_basemaps

    basemaps = list_basemaps(free_only=not show_all)
    if ctx.obj.get("as_json"):
        _output(basemaps, True)
    else:
        for name in basemaps:
            click.echo(f"  {name}")
        click.echo(f"\n  Total: {len(basemaps)} basemaps")


# ---------------------------------------------------------------------------
# Vector commands
# ---------------------------------------------------------------------------


@cli.group()
@click.pass_context
def vector(ctx):
    """Vector operations."""
    pass


@vector.command("convert")
@click.argument("source", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--simplify", default=None, type=float, help="Simplification tolerance.")
@click.option("--crs", default=None, help="Target CRS.")
@click.pass_context
def vector_convert(ctx, source, output, simplify, crs):
    """Convert a raster mask to vector format."""
    from cli_anything.samgeo.core.vector import raster_to_vector

    result = raster_to_vector(source, output, simplify_tolerance=simplify, dst_crs=crs)
    _output(result, ctx.obj.get("as_json"))


@vector.command("info")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def vector_info_cmd(ctx, path):
    """Show vector file information."""
    from cli_anything.samgeo.core.vector import vector_info

    info = vector_info(path)
    _output(info, ctx.obj.get("as_json"))


@vector.command("filter")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--min-area", default=None, type=float, help="Minimum geometry area.")
@click.option("--max-area", default=None, type=float, help="Maximum geometry area.")
@click.option("--column", default=None, help="Column to filter by.")
@click.option("--value", default=None, help="Value to match.")
@click.pass_context
def vector_filter(ctx, input_path, output_path, min_area, max_area, column, value):
    """Filter vector features by area or attribute."""
    from cli_anything.samgeo.core.vector import filter_vectors

    result = filter_vectors(
        input_path,
        output_path,
        min_area=min_area,
        max_area=max_area,
        column=column,
        value=value,
    )
    _output(result, ctx.obj.get("as_json"))


# ---------------------------------------------------------------------------
# Export commands
# ---------------------------------------------------------------------------


@cli.group()
@click.pass_context
def export(ctx):
    """Export operations."""
    pass


@export.command("render")
@click.argument("output", type=click.Path())
@click.option("-f", "--format", "fmt", default="geotiff", help="Export format.")
@click.option(
    "--simplify",
    default=None,
    type=float,
    help="Simplification tolerance (vector only).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing file.")
@click.pass_context
def export_render(ctx, output, fmt, simplify, overwrite):
    """Export masks to a file format."""
    p = _load_project(ctx)
    from cli_anything.samgeo.core.export import export_masks

    result = export_masks(
        p, output, fmt=fmt, simplify_tolerance=simplify, overwrite=overwrite
    )
    _save_project(ctx, p)
    _output(result, ctx.obj.get("as_json"))


@export.command("formats")
@click.pass_context
def export_formats(ctx):
    """List available export formats."""
    from cli_anything.samgeo.core.export import list_formats

    formats = list_formats()
    if ctx.obj.get("as_json"):
        _output(formats, True)
    else:
        headers = ["Format", "Extension", "Type", "Description"]
        rows = [
            [f["format"], f["extension"], f["type"], f["description"]] for f in formats
        ]
        from cli_anything.samgeo.utils.repl_skin import ReplSkin

        skin = ReplSkin("samgeo")
        skin.table(headers, rows)


# ---------------------------------------------------------------------------
# Session commands
# ---------------------------------------------------------------------------


@cli.group()
@click.pass_context
def session(ctx):
    """Session management commands."""
    pass


@session.command("status")
@click.pass_context
def session_status(ctx):
    """Show current session status."""
    p = _load_project(ctx)
    sess = session_mod.Session(p)
    _output(sess.get_status(), ctx.obj.get("as_json"))


@session.command("history")
@click.option("-l", "--limit", default=10, type=int, help="Max entries to show.")
@click.pass_context
def session_history(ctx, limit):
    """Show operation history."""
    p = _load_project(ctx)
    sess = session_mod.Session(p)
    history = sess.get_history(limit=limit)
    if ctx.obj.get("as_json"):
        _output(history, True)
    else:
        if not history:
            click.echo("No history entries.")
        else:
            for i, entry in enumerate(history):
                click.echo(f"  [{i+1}] {entry['action']} at {entry['timestamp']}")


# ---------------------------------------------------------------------------
# REPL command
# ---------------------------------------------------------------------------


@cli.command(hidden=True)
@click.option("--project-path", type=click.Path(), default=None)
@click.pass_context
def repl(ctx, project_path):
    """Start the interactive REPL."""
    from cli_anything.samgeo.utils.repl_skin import ReplSkin

    skin = ReplSkin("samgeo", version=__version__)
    skin.print_banner()

    pt_session = skin.create_prompt_session()
    sess = session_mod.Session()

    if project_path and os.path.exists(project_path):
        try:
            p = proj_mod.open_project(project_path)
            sess = session_mod.Session(p)
            skin.success(f"Loaded project: {p.get('name', project_path)}")
        except Exception as e:
            skin.error(f"Failed to load project: {e}")

    commands = {
        "help": "Show this help message",
        "project new": "Create a new project",
        "project info": "Show project info",
        "project set-source": "Set source image",
        "project history": "Show operation history",
        "model list": "List available models",
        "model info": "Show model details",
        "model check": "Check model availability",
        "segment automatic": "Run automatic segmentation",
        "segment points": "Segment with point prompts",
        "segment boxes": "Segment with box prompts",
        "segment text": "Segment with text prompt",
        "data info": "Show raster info",
        "data download-tiles": "Download TMS tiles",
        "data reproject": "Reproject a raster",
        "data split": "Split raster into tiles",
        "data basemaps": "List basemap sources",
        "vector convert": "Convert mask to vectors",
        "vector info": "Show vector info",
        "vector filter": "Filter vector features",
        "export render": "Export masks to format",
        "export formats": "List export formats",
        "session status": "Show session status",
        "session history": "Show operation history",
        "quit": "Exit the REPL",
    }

    while True:
        try:
            proj_name = sess.project.get("name") if sess.project else None
            line = skin.get_input(pt_session, project_name=proj_name)

            if not line or not line.strip():
                continue

            line = line.strip()

            if line in ("quit", "exit", "q"):
                skin.print_goodbye()
                break

            if line == "help":
                skin.help(commands)
                continue

            # Parse the line into args and invoke through Click
            try:
                args = shlex.split(line)
            except ValueError as e:
                skin.error(f"Parse error: {e}")
                continue

            # Inject project path if session has one
            if sess.project and sess.project.get("_path"):
                proj_arg = sess.project["_path"]
                if "--project" not in args:
                    args = ["--project", proj_arg] + args

            try:
                cli.main(args=args, standalone_mode=False, **{"obj": ctx.obj})
            except SystemExit:
                pass
            except click.UsageError as e:
                skin.error(str(e))
            except click.Abort:
                skin.warning("Command aborted.")
            except Exception as e:
                skin.error(f"Error: {e}")

        except KeyboardInterrupt:
            click.echo()
            continue
        except EOFError:
            skin.print_goodbye()
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
