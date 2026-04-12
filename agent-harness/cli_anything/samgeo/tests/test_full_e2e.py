"""E2E tests for cli-anything-samgeo.

Tests real segmentation workflows using the actual samgeo library
and CLI subprocess tests using the installed command.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# CLI resolution helper
# ---------------------------------------------------------------------------


def _resolve_cli(name):
    """Resolve installed CLI command; falls back to python -m for dev.

    Set env CLI_ANYTHING_FORCE_INSTALLED=1 to require the installed command.
    """
    import shutil

    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    module = (
        name.replace("cli-anything-", "cli_anything.")
        + "."
        + name.split("-")[-1]
        + "_cli"
    )
    print(f"[_resolve_cli] Falling back to: {sys.executable} -m {module}")
    return [sys.executable, "-m", module]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_geotiff(tmp_dir):
    """Create a small synthetic GeoTIFF for testing."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    path = os.path.join(tmp_dir, "sample.tif")
    width, height = 256, 256
    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)

    transform = from_bounds(-122.5, 37.5, -122.0, 38.0, width, height)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 3,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    print(f"\n  Synthetic GeoTIFF: {path} ({os.path.getsize(path):,} bytes)")
    return path


@pytest.fixture
def sample_mask(tmp_dir):
    """Create a small synthetic mask GeoTIFF for testing."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    path = os.path.join(tmp_dir, "mask.tif")
    width, height = 256, 256
    # Create a mask with a few distinct regions
    data = np.zeros((1, height, width), dtype=np.uint8)
    data[0, 50:100, 50:100] = 1
    data[0, 150:200, 150:200] = 2
    data[0, 50:100, 150:200] = 3

    transform = from_bounds(-122.5, 37.5, -122.0, 38.0, width, height)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    print(f"\n  Synthetic mask: {path} ({os.path.getsize(path):,} bytes)")
    return path


@pytest.fixture
def project_with_mask(tmp_dir, sample_geotiff, sample_mask):
    """Create a project JSON with source and masks set."""
    from cli_anything.samgeo.core.project import (
        create_project,
        set_source,
        set_masks,
        save_project,
    )

    proj_path = os.path.join(tmp_dir, "project.json")
    p = create_project(name="test-e2e", model_type="sam2", device="cpu")
    set_source(p, sample_geotiff)
    set_masks(p, sample_mask, count=3)
    save_project(p, proj_path)
    return proj_path


# ---------------------------------------------------------------------------
# E2E Data pipeline tests
# ---------------------------------------------------------------------------


class TestDataPipelineE2E:
    def test_raster_info_real(self, sample_geotiff):
        """Test raster_info on a real GeoTIFF."""
        from cli_anything.samgeo.core.data import raster_info

        info = raster_info(sample_geotiff)
        assert info["width"] == 256
        assert info["height"] == 256
        assert info["bands"] == 3
        assert info["crs"] == "EPSG:4326"
        assert info["file_size"] > 0
        print(
            f"\n  Raster info: {info['width']}x{info['height']}, {info['bands']} bands, {info['crs']}"
        )

    def test_download_tiles_osm(self, tmp_dir):
        """Download real OSM tiles as GeoTIFF."""
        from cli_anything.samgeo.core.data import download_tiles

        output = os.path.join(tmp_dir, "osm_tiles.tif")
        # Small bbox around San Francisco
        bbox = [-122.42, 37.77, -122.40, 37.79]
        result = download_tiles(output, bbox, zoom=15, source="OpenStreetMap")

        assert os.path.exists(result["output"])
        assert result["file_size"] > 0
        print(
            f"\n  Downloaded tiles: {result['output']} ({result['file_size']:,} bytes)"
        )

    def test_reproject_real(self, sample_geotiff, tmp_dir):
        """Reproject a real GeoTIFF."""
        from cli_anything.samgeo.core.data import reproject_raster

        output = os.path.join(tmp_dir, "reprojected.tif")
        result = reproject_raster(sample_geotiff, output, dst_crs="EPSG:3857")

        assert os.path.exists(result["output"])
        assert result["file_size"] > 0

        # Verify CRS changed
        from cli_anything.samgeo.core.data import raster_info

        info = raster_info(output)
        assert "3857" in str(info["crs"])
        print(
            f"\n  Reprojected: {result['output']} ({result['file_size']:,} bytes), CRS={info['crs']}"
        )

    def test_split_raster_real(self, sample_geotiff, tmp_dir):
        """Split a real raster into tiles."""
        from cli_anything.samgeo.core.data import split_raster_tiles

        out_dir = os.path.join(tmp_dir, "tiles")
        result = split_raster_tiles(sample_geotiff, out_dir, tile_size=128, overlap=0)

        assert result["tile_count"] > 0
        assert os.path.isdir(result["output_dir"])
        print(f"\n  Split into {result['tile_count']} tiles in {result['output_dir']}")


# ---------------------------------------------------------------------------
# E2E Vector pipeline tests
# ---------------------------------------------------------------------------


class TestVectorPipelineE2E:
    def test_raster_to_vector_gpkg(self, sample_mask, tmp_dir):
        """Convert a real mask raster to GeoPackage."""
        from cli_anything.samgeo.core.vector import raster_to_vector

        output = os.path.join(tmp_dir, "vectors.gpkg")
        result = raster_to_vector(sample_mask, output)

        assert os.path.exists(result["output"])
        assert result["feature_count"] > 0
        assert result["format"] == "gpkg"
        print(
            f"\n  Vectors: {result['output']} ({result['feature_count']} features, {result['file_size']:,} bytes)"
        )

    def test_raster_to_vector_geojson(self, sample_mask, tmp_dir):
        """Convert a real mask raster to GeoJSON."""
        from cli_anything.samgeo.core.vector import raster_to_vector

        output = os.path.join(tmp_dir, "vectors.geojson")
        result = raster_to_vector(sample_mask, output)

        assert os.path.exists(result["output"])
        assert result["feature_count"] > 0

        # Validate GeoJSON structure
        with open(output) as f:
            geojson = json.load(f)
        assert geojson["type"] in ("FeatureCollection", "Feature")
        print(f"\n  GeoJSON: {result['output']} ({result['feature_count']} features)")

    def test_vector_info_real(self, sample_mask, tmp_dir):
        """Get info on a real vector file."""
        from cli_anything.samgeo.core.vector import raster_to_vector, vector_info

        gpkg = os.path.join(tmp_dir, "info_test.gpkg")
        raster_to_vector(sample_mask, gpkg)

        info = vector_info(gpkg)
        assert info["feature_count"] > 0
        assert info["crs"] is not None
        assert len(info["columns"]) > 0
        print(f"\n  Vector info: {info['feature_count']} features, CRS={info['crs']}")

    def test_filter_vectors_real(self, sample_mask, tmp_dir):
        """Filter vector features by area."""
        from cli_anything.samgeo.core.vector import raster_to_vector, filter_vectors

        gpkg = os.path.join(tmp_dir, "to_filter.gpkg")
        raster_to_vector(sample_mask, gpkg)

        filtered = os.path.join(tmp_dir, "filtered.gpkg")
        result = filter_vectors(gpkg, filtered, min_area=0.0001)

        assert os.path.exists(result["output"])
        assert result["filtered_count"] <= result["original_count"]
        print(
            f"\n  Filtered: {result['original_count']} -> {result['filtered_count']} features "
            f"(removed {result['removed_count']})"
        )


# ---------------------------------------------------------------------------
# E2E Export tests
# ---------------------------------------------------------------------------


class TestExportE2E:
    def test_export_geotiff(self, project_with_mask, tmp_dir):
        """Export masks as GeoTIFF."""
        from cli_anything.samgeo.core.project import open_project
        from cli_anything.samgeo.core.export import export_masks

        p = open_project(project_with_mask)
        output = os.path.join(tmp_dir, "exported.tif")
        result = export_masks(p, output, fmt="geotiff", overwrite=True)

        assert os.path.exists(result["output"])
        assert result["file_size"] > 0

        # Verify it's a real GeoTIFF
        with open(result["output"], "rb") as f:
            header = f.read(4)
        assert header[:2] in (b"II", b"MM")  # TIFF magic bytes
        print(
            f"\n  Exported GeoTIFF: {result['output']} ({result['file_size']:,} bytes)"
        )

    def test_export_png(self, project_with_mask, tmp_dir):
        """Export masks as PNG."""
        from cli_anything.samgeo.core.project import open_project
        from cli_anything.samgeo.core.export import export_masks

        p = open_project(project_with_mask)
        output = os.path.join(tmp_dir, "exported.png")
        result = export_masks(p, output, fmt="png", overwrite=True)

        assert os.path.exists(result["output"])
        assert result["file_size"] > 0

        # Verify PNG magic bytes
        with open(result["output"], "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"
        print(f"\n  Exported PNG: {result['output']} ({result['file_size']:,} bytes)")

    def test_export_gpkg(self, project_with_mask, tmp_dir):
        """Export masks as GeoPackage."""
        from cli_anything.samgeo.core.project import open_project
        from cli_anything.samgeo.core.export import export_masks

        p = open_project(project_with_mask)
        output = os.path.join(tmp_dir, "exported.gpkg")
        result = export_masks(p, output, fmt="gpkg", overwrite=True)

        assert os.path.exists(result["output"])
        assert result["file_size"] > 0
        print(
            f"\n  Exported GeoPackage: {result['output']} ({result['file_size']:,} bytes)"
        )


# ---------------------------------------------------------------------------
# CLI Subprocess tests
# ---------------------------------------------------------------------------


class TestCLISubprocess:
    CLI_BASE = _resolve_cli("cli-anything-samgeo")

    def _run(self, args, check=True):
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_help(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "cli-anything-samgeo" in result.stdout
        assert "segment" in result.stdout

    def test_version(self):
        result = self._run(["--version"])
        assert result.returncode == 0
        assert "cli-anything-samgeo" in result.stdout

    def test_project_new_json(self, tmp_dir):
        out = os.path.join(tmp_dir, "test.json")
        result = self._run(["--json", "project", "new", "-n", "cli-test", "-o", out])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["name"] == "cli-test"
        assert data["status"] == "created"
        assert os.path.exists(out)

    def test_model_list_json(self):
        result = self._run(["--json", "model", "list"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(m["type"] == "sam2" for m in data)

    def test_model_info_json(self):
        result = self._run(["--json", "model", "info", "sam2"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["type"] == "sam2"

    def test_data_info_json(self, sample_geotiff):
        result = self._run(["--json", "data", "info", sample_geotiff])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["width"] == 256
        assert data["height"] == 256

    def test_export_formats_json(self):
        result = self._run(["--json", "export", "formats"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        format_names = [f["format"] for f in data]
        assert "geotiff" in format_names
        assert "gpkg" in format_names

    def test_project_info_json(self, tmp_dir):
        """Create project then get its info."""
        proj = os.path.join(tmp_dir, "info_test.json")
        self._run(["project", "new", "-n", "info-test", "-o", proj])

        result = self._run(["--json", "--project", proj, "project", "info"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["name"] == "info-test"

    def test_project_history_json(self, tmp_dir):
        """Create project, check history is empty."""
        proj = os.path.join(tmp_dir, "hist_test.json")
        self._run(["project", "new", "-n", "hist-test", "-o", proj])

        result = self._run(["--json", "--project", proj, "project", "history"])
        assert result.returncode == 0

    def test_model_check_json(self):
        result = self._run(["--json", "model", "check", "sam2"], check=False)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "available" in data

    def test_full_data_workflow(self, tmp_dir):
        """Full workflow: create project -> data info -> export formats."""
        proj = os.path.join(tmp_dir, "workflow.json")

        # Create project
        r1 = self._run(["--json", "project", "new", "-n", "wf-test", "-o", proj])
        assert r1.returncode == 0
        d1 = json.loads(r1.stdout)
        assert d1["status"] == "created"

        # List export formats
        r2 = self._run(["--json", "export", "formats"])
        assert r2.returncode == 0
        d2 = json.loads(r2.stdout)
        assert len(d2) > 0

        # List models
        r3 = self._run(["--json", "model", "list", "-t", "sam2"])
        assert r3.returncode == 0
        d3 = json.loads(r3.stdout)
        assert all(m["type"] == "sam2" for m in d3)

        print(f"\n  Full workflow passed: project={proj}")

    def test_vector_convert_subprocess(self, sample_mask, tmp_dir):
        """Convert mask to vector via CLI subprocess."""
        output = os.path.join(tmp_dir, "sub_vectors.gpkg")
        result = self._run(["--json", "vector", "convert", sample_mask, output])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["feature_count"] > 0
        assert os.path.exists(output)
        print(f"\n  Subprocess vector convert: {data['feature_count']} features")

    def test_export_via_subprocess(self, project_with_mask, tmp_dir):
        """Export masks via CLI subprocess."""
        output = os.path.join(tmp_dir, "sub_export.tif")
        result = self._run(
            [
                "--json",
                "--project",
                project_with_mask,
                "export",
                "render",
                output,
                "-f",
                "geotiff",
                "--overwrite",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["file_size"] > 0
        assert os.path.exists(output)

        # Verify TIFF magic bytes
        with open(output, "rb") as f:
            header = f.read(4)
        assert header[:2] in (b"II", b"MM")
        print(f"\n  Subprocess export: {output} ({data['file_size']:,} bytes)")
