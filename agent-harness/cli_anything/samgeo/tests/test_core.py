"""Unit tests for cli-anything-samgeo core modules.

All tests use synthetic data — no external dependencies beyond samgeo.
"""

import json
import os
import tempfile

import pytest

from cli_anything.samgeo.core.project import (
    add_history_entry,
    create_project,
    get_project_info,
    open_project,
    save_project,
    set_masks,
    set_source,
    set_vectors,
)
from cli_anything.samgeo.core.model import (
    get_default_model_id,
    get_model_info,
    get_model_types,
    list_models,
)
from cli_anything.samgeo.core.session import Session
from cli_anything.samgeo.core.export import list_formats, EXPORT_FORMATS


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_project(tmp_dir):
    """Create a sample project dict."""
    return create_project(
        name="test-project",
        model_type="sam2",
        model_id="sam2-hiera-large",
        device="cpu",
    )


@pytest.fixture
def saved_project(tmp_dir, sample_project):
    """Create and save a sample project, return (project, path)."""
    path = os.path.join(tmp_dir, "test.json")
    save_project(sample_project, path)
    return sample_project, path


# -------------------------------------------------------------------------
# project.py tests
# -------------------------------------------------------------------------


class TestProject:
    def test_create_project_basic(self):
        p = create_project(name="basic")
        assert p["name"] == "basic"
        assert p["version"] == "1.0"
        assert p["model"]["type"] == "sam2"
        assert p["source"] is None
        assert p["masks"] is None
        assert p["history"] == []

    def test_create_project_with_model(self):
        p = create_project(
            name="custom", model_type="sam3", model_id="facebook/sam3", device="cuda"
        )
        assert p["model"]["type"] == "sam3"
        assert p["model"]["id"] == "facebook/sam3"
        assert p["model"]["device"] == "cuda"

    def test_create_project_saves_to_disk(self, tmp_dir):
        path = os.path.join(tmp_dir, "proj.json")
        p = create_project(name="saved", output_path=path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "saved"

    def test_open_project(self, saved_project):
        _, path = saved_project
        p = open_project(path)
        assert p["name"] == "test-project"
        assert p["_path"] == os.path.abspath(path)

    def test_open_project_not_found(self):
        with pytest.raises(FileNotFoundError):
            open_project("/nonexistent/path/project.json")

    def test_open_project_invalid(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            json.dump({"random": "data"}, f)
        with pytest.raises(ValueError, match="Invalid project"):
            open_project(path)

    def test_save_project(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "saved.json")
        save_project(sample_project, path)
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "test-project"
        assert "modified" in data

    def test_save_project_no_path(self, sample_project):
        with pytest.raises(ValueError, match="No path"):
            save_project(sample_project)

    def test_get_project_info(self, sample_project):
        info = get_project_info(sample_project)
        assert info["name"] == "test-project"
        assert info["model_type"] == "sam2"
        assert info["has_source"] is False
        assert info["has_masks"] is False

    def test_add_history_entry(self, sample_project):
        add_history_entry(sample_project, "test_action", params={"key": "val"})
        assert len(sample_project["history"]) == 1
        assert sample_project["history"][0]["action"] == "test_action"
        assert "timestamp" in sample_project["history"][0]

    def test_set_masks(self, sample_project, tmp_dir):
        mask_path = os.path.join(tmp_dir, "masks.tif")
        with open(mask_path, "w") as f:
            f.write("fake")  # placeholder
        set_masks(sample_project, mask_path, count=42)
        assert sample_project["masks"]["count"] == 42
        assert sample_project["masks"]["path"] == os.path.abspath(mask_path)

    def test_set_vectors(self, sample_project, tmp_dir):
        vec_path = os.path.join(tmp_dir, "output.gpkg")
        set_vectors(sample_project, vec_path, feature_count=10)
        assert sample_project["vectors"]["feature_count"] == 10

    def test_set_source(self, sample_project, tmp_dir):
        # Create a dummy file (not a real GeoTIFF, so CRS/bounds won't be read)
        src_path = os.path.join(tmp_dir, "image.tif")
        with open(src_path, "w") as f:
            f.write("fake")
        set_source(sample_project, src_path)
        assert sample_project["source"]["path"] == os.path.abspath(src_path)

    def test_project_round_trip(self, tmp_dir):
        """Create, save, open, modify, save, open — full round-trip."""
        path = os.path.join(tmp_dir, "rt.json")
        p = create_project(name="round-trip", output_path=path)
        add_history_entry(p, "step1")

        save_project(p, path)
        p2 = open_project(path)
        assert p2["name"] == "round-trip"
        assert len(p2["history"]) == 1

        add_history_entry(p2, "step2")
        save_project(p2, path)
        p3 = open_project(path)
        assert len(p3["history"]) == 2


# -------------------------------------------------------------------------
# model.py tests
# -------------------------------------------------------------------------


class TestModel:
    def test_list_models(self):
        models = list_models()
        assert len(models) > 0
        assert all("type" in m for m in models)
        assert all("model_id" in m for m in models)

    def test_list_models_has_sam2(self):
        models = list_models()
        sam2_models = [m for m in models if m["type"] == "sam2"]
        assert len(sam2_models) == 4  # tiny, small, base-plus, large

    def test_get_model_info_type(self):
        info = get_model_info("sam2")
        assert info["type"] == "sam2"
        assert info["display_name"] == "SAM 2"
        assert "models" in info

    def test_get_model_info_specific(self):
        info = get_model_info("sam2", "sam2-hiera-large")
        assert info["model_id"] == "sam2-hiera-large"
        assert info["type"] == "sam2"
        assert "params" in info

    def test_get_model_info_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_info("nonexistent")

    def test_get_model_info_invalid_id(self):
        with pytest.raises(ValueError, match="Unknown model ID"):
            get_model_info("sam2", "nonexistent-model")

    def test_get_model_types(self):
        types = get_model_types()
        assert "sam" in types
        assert "sam2" in types
        assert "sam3" in types
        assert "text_sam" in types

    def test_get_default_model_id(self):
        assert get_default_model_id("sam") == "vit_h"
        assert get_default_model_id("sam2") == "sam2-hiera-large"
        assert get_default_model_id("sam3") == "facebook/sam3"


# -------------------------------------------------------------------------
# session.py tests
# -------------------------------------------------------------------------


class TestSession:
    def test_session_init_empty(self):
        s = Session()
        assert s.project == {}
        assert not s.can_undo
        assert not s.can_redo

    def test_session_init_with_project(self, sample_project):
        s = Session(sample_project)
        assert s.project["name"] == "test-project"

    def test_session_mutate_and_undo(self, sample_project):
        s = Session(sample_project)
        original_name = s.project["name"]
        s.mutate("name", "modified")
        assert s.project["name"] == "modified"
        assert s.can_undo

        s.undo()
        assert s.project["name"] == original_name

    def test_session_redo(self, sample_project):
        s = Session(sample_project)
        s.mutate("name", "modified")
        s.undo()
        assert s.can_redo

        s.redo()
        assert s.project["name"] == "modified"

    def test_session_multiple_undos(self, sample_project):
        s = Session(sample_project)
        s.mutate("name", "step1")
        s.mutate("name", "step2")
        s.mutate("name", "step3")
        assert s.project["name"] == "step3"

        s.undo()
        assert s.project["name"] == "step2"
        s.undo()
        assert s.project["name"] == "step1"
        s.undo()
        assert s.project["name"] == "test-project"

    def test_session_undo_empty(self):
        s = Session()
        assert not s.undo()

    def test_session_redo_empty(self):
        s = Session()
        assert not s.redo()

    def test_session_redo_cleared_on_mutate(self, sample_project):
        s = Session(sample_project)
        s.mutate("name", "step1")
        s.undo()
        assert s.can_redo
        s.mutate("name", "step2")
        assert not s.can_redo

    def test_session_save_load(self, tmp_dir, sample_project):
        s = Session(sample_project)
        path = os.path.join(tmp_dir, "session.json")
        s.save_session(path)
        assert os.path.exists(path)

        s2 = Session()
        s2.load_session(path)
        assert s2.project["name"] == "test-project"

    def test_session_load_not_found(self):
        s = Session()
        with pytest.raises(FileNotFoundError):
            s.load_session("/nonexistent/session.json")

    def test_session_get_status(self, sample_project):
        s = Session(sample_project)
        status = s.get_status()
        assert status["has_project"] is True
        assert status["project_name"] == "test-project"
        assert status["undo_depth"] == 0

    def test_session_get_history(self, sample_project):
        add_history_entry(sample_project, "action1")
        add_history_entry(sample_project, "action2")
        s = Session(sample_project)
        history = s.get_history(limit=1)
        assert len(history) == 1
        assert history[0]["action"] == "action2"  # most recent first

    def test_session_update_project(self, sample_project):
        s = Session(sample_project)
        s.update_project({"name": "updated", "extra_key": "value"})
        assert s.project["name"] == "updated"
        assert s.project["extra_key"] == "value"
        s.undo()
        assert s.project["name"] == "test-project"


# -------------------------------------------------------------------------
# export.py tests
# -------------------------------------------------------------------------


class TestExport:
    def test_list_formats(self):
        formats = list_formats()
        assert len(formats) > 0
        format_names = [f["format"] for f in formats]
        assert "geotiff" in format_names
        assert "gpkg" in format_names
        assert "geojson" in format_names

    def test_export_no_masks_error(self, sample_project, tmp_dir):
        from cli_anything.samgeo.core.export import export_masks

        with pytest.raises(ValueError, match="No masks generated"):
            export_masks(sample_project, os.path.join(tmp_dir, "out.tif"))

    def test_export_invalid_format(self, sample_project, tmp_dir):
        sample_project["masks"] = {"path": "/fake/masks.tif"}
        from cli_anything.samgeo.core.export import export_masks

        with pytest.raises(ValueError, match="Unknown export format"):
            export_masks(
                sample_project, os.path.join(tmp_dir, "out.tif"), fmt="invalid"
            )

    def test_export_file_exists_no_overwrite(self, sample_project, tmp_dir):
        sample_project["masks"] = {"path": os.path.join(tmp_dir, "masks.tif")}
        # Create both the mask file and the output file
        for name in ("masks.tif", "out.tif"):
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("fake")
        from cli_anything.samgeo.core.export import export_masks

        with pytest.raises(FileExistsError, match="Output file exists"):
            export_masks(
                sample_project, os.path.join(tmp_dir, "out.tif"), overwrite=False
            )


# -------------------------------------------------------------------------
# data.py & vector.py error handling tests
# -------------------------------------------------------------------------


class TestDataErrors:
    def test_raster_info_not_found(self):
        from cli_anything.samgeo.core.data import raster_info

        with pytest.raises(FileNotFoundError):
            raster_info("/nonexistent/file.tif")

    def test_reproject_not_found(self, tmp_dir):
        from cli_anything.samgeo.core.data import reproject_raster

        with pytest.raises(FileNotFoundError):
            reproject_raster("/nonexistent.tif", os.path.join(tmp_dir, "out.tif"))

    def test_split_not_found(self, tmp_dir):
        from cli_anything.samgeo.core.data import split_raster_tiles

        with pytest.raises(FileNotFoundError):
            split_raster_tiles("/nonexistent.tif", tmp_dir)


class TestVectorErrors:
    def test_vector_info_not_found(self):
        from cli_anything.samgeo.core.vector import vector_info

        with pytest.raises(FileNotFoundError):
            vector_info("/nonexistent/file.gpkg")

    def test_raster_to_vector_not_found(self, tmp_dir):
        from cli_anything.samgeo.core.vector import raster_to_vector

        with pytest.raises(FileNotFoundError):
            raster_to_vector("/nonexistent.tif", os.path.join(tmp_dir, "out.gpkg"))

    def test_filter_not_found(self, tmp_dir):
        from cli_anything.samgeo.core.vector import filter_vectors

        with pytest.raises(FileNotFoundError):
            filter_vectors("/nonexistent.gpkg", os.path.join(tmp_dir, "out.gpkg"))
