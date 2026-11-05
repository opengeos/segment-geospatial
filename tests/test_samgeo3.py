"""Fast constructor tests for SAM3 model selection logic."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


def _load_samgeo3(monkeypatch):
    """Load samgeo3.py with minimal stubs for heavy optional dependencies."""
    repo_root = Path(__file__).resolve().parents[1]
    samgeo_dir = repo_root / "samgeo"

    package = types.ModuleType("samgeo")
    package.__path__ = [str(samgeo_dir)]

    common = types.ModuleType("samgeo.common")
    common.get_device = lambda: "cpu"
    common.download_file = lambda url, output, quiet=True: output
    common.show_image = lambda *args, **kwargs: None
    package.common = common

    registry_path = samgeo_dir / "model_registry.py"
    registry_spec = importlib.util.spec_from_file_location(
        "samgeo.model_registry", registry_path
    )
    registry = importlib.util.module_from_spec(registry_spec)
    registry_spec.loader.exec_module(registry)

    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "samgeo", package)
    monkeypatch.setitem(sys.modules, "samgeo.common", common)
    monkeypatch.setitem(sys.modules, "samgeo.model_registry", registry)

    module_path = samgeo_dir / "samgeo3.py"
    spec = importlib.util.spec_from_file_location("samgeo.samgeo3", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "samgeo.samgeo3", module)
    spec.loader.exec_module(module)
    return module


class _DummyModel:
    def to(self, device):
        self.device = device
        return self


def _patch_meta_build(monkeypatch, samgeo3):
    builder = Mock(return_value=_DummyModel())
    processor = Mock()
    monkeypatch.setattr(samgeo3, "build_sam3_image_model", builder, raising=False)
    monkeypatch.setattr(samgeo3, "MetaSam3Processor", processor, raising=False)
    return builder, processor


def _patch_hf_hub_download(monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = Mock(
        side_effect=lambda repo_id, filename: f"/tmp/{repo_id.replace('/', '-')}/{filename}"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub.hf_hub_download


def _init_meta_backend(samgeo3, **overrides):
    params = {
        "model_id": "facebook/sam3.1",
        "bpe_path": None,
        "device": "cpu",
        "eval_mode": True,
        "checkpoint_path": None,
        "load_from_HF": True,
        "enable_segmentation": True,
        "enable_inst_interactivity": False,
        "compile_mode": False,
        "resolution": 1008,
        "confidence_threshold": 0.5,
    }
    params.update(overrides)
    model = samgeo3.SamGeo3.__new__(samgeo3.SamGeo3)
    model._init_meta_backend(**params)


@pytest.fixture
def samgeo3(monkeypatch):
    return _load_samgeo3(monkeypatch)


def test_samgeo3_rejects_invalid_backend_before_dependency_checks(samgeo3):
    """Backend typos should fail as user errors, even without SAM3 installed."""
    with pytest.raises(ValueError, match="Invalid backend"):
        samgeo3.SamGeo3(backend="unknown")


def test_sam31_rejects_transformers_backend_before_dependency_checks(samgeo3):
    """SAM 3.1 is a Meta checkpoint path, not a Transformers model id."""
    with pytest.raises(ValueError, match="backend='meta'"):
        samgeo3.SamGeo3(backend="transformers", model_id="facebook/sam3.1")


def test_samgeo3_constructor_preserves_custom_meta_model_id(monkeypatch, samgeo3):
    """Custom Meta model ids remain accepted for direct SamGeo3 users."""
    monkeypatch.setattr(samgeo3, "SAM3_META_AVAILABLE", True)
    init_meta = Mock()
    monkeypatch.setattr(samgeo3.SamGeo3, "_init_meta_backend", init_meta)

    model = samgeo3.SamGeo3(
        backend="meta",
        model_id="custom/sam3-checkpoint",
        device="cpu",
    )

    assert model.model_id == "custom/sam3-checkpoint"
    init_meta.assert_called_once()
    assert init_meta.call_args.kwargs["model_id"] == "custom/sam3-checkpoint"


def test_sam31_load_from_hf_uses_versioned_checkpoint_download(
    monkeypatch, tmp_path, samgeo3
):
    """SAM 3.1 selects the sam3.1 checkpoint helper without touching the network."""
    bpe_path = tmp_path / "bpe.txt.gz"
    bpe_path.write_text("stub")
    checkpoint_path = tmp_path / "sam3.1_multiplex.pt"

    builder, _ = _patch_meta_build(monkeypatch, samgeo3)

    def download_ckpt_from_hf(*, version):
        downloader(version=version)
        return str(checkpoint_path)

    downloader = Mock()
    monkeypatch.setattr(samgeo3, "download_ckpt_from_hf", download_ckpt_from_hf)

    _init_meta_backend(
        samgeo3,
        bpe_path=str(bpe_path),
    )

    downloader.assert_called_once_with(version="sam3.1")
    assert builder.call_args.kwargs["checkpoint_path"] == str(checkpoint_path)
    assert builder.call_args.kwargs["load_from_HF"] is False


def test_sam31_old_downloader_signature_uses_hf_hub_fallback(
    monkeypatch, tmp_path, samgeo3
):
    """PyPI sam3 0.1.4 has an old no-arg helper, so fallback to HF directly."""
    bpe_path = tmp_path / "bpe.txt.gz"
    bpe_path.write_text("stub")

    builder, _ = _patch_meta_build(monkeypatch, samgeo3)
    old_downloader = Mock(return_value="/tmp/facebook-sam3/sam3.pt")

    def download_ckpt_from_hf():
        return old_downloader()

    monkeypatch.setattr(samgeo3, "download_ckpt_from_hf", download_ckpt_from_hf)
    hf_hub_download = _patch_hf_hub_download(monkeypatch)

    _init_meta_backend(
        samgeo3,
        bpe_path=str(bpe_path),
    )

    old_downloader.assert_not_called()
    hf_hub_download.assert_any_call(
        repo_id="facebook/sam3.1",
        filename="config.json",
    )
    hf_hub_download.assert_any_call(
        repo_id="facebook/sam3.1",
        filename="sam3.1_multiplex.pt",
    )
    assert (
        builder.call_args.kwargs["checkpoint_path"]
        == "/tmp/facebook-sam3.1/sam3.1_multiplex.pt"
    )
    assert builder.call_args.kwargs["load_from_HF"] is False


def test_sam31_explicit_checkpoint_skips_hf_download(monkeypatch, tmp_path, samgeo3):
    """Explicit checkpoints remain authoritative and avoid HF downloads."""
    bpe_path = tmp_path / "bpe.txt.gz"
    bpe_path.write_text("stub")
    checkpoint_path = tmp_path / "local-sam31.pt"
    checkpoint_path.write_text("stub")

    builder, _ = _patch_meta_build(monkeypatch, samgeo3)
    downloader = Mock()
    monkeypatch.setattr(samgeo3, "download_ckpt_from_hf", downloader)

    _init_meta_backend(
        samgeo3,
        bpe_path=str(bpe_path),
        checkpoint_path=str(checkpoint_path),
    )

    downloader.assert_not_called()
    assert builder.call_args.kwargs["checkpoint_path"] == str(checkpoint_path)
    assert builder.call_args.kwargs["load_from_HF"] is False


def test_default_sam3_keeps_standard_hf_loading(monkeypatch, tmp_path, samgeo3):
    """The existing SAM3 default path should not call the SAM 3.1 downloader."""
    bpe_path = tmp_path / "bpe.txt.gz"
    bpe_path.write_text("stub")

    builder, _ = _patch_meta_build(monkeypatch, samgeo3)
    downloader = Mock()
    monkeypatch.setattr(samgeo3, "download_ckpt_from_hf", downloader)

    _init_meta_backend(
        samgeo3,
        model_id="facebook/sam3",
        bpe_path=str(bpe_path),
    )

    downloader.assert_not_called()
    assert builder.call_args.kwargs["checkpoint_path"] is None
    assert builder.call_args.kwargs["load_from_HF"] is True


def test_checkpoint_env_overrides_sam31_hf_download(monkeypatch, tmp_path, samgeo3):
    """SAM3_CHECKPOINT_PATH keeps deployment overrides backward compatible."""
    bpe_path = tmp_path / "bpe.txt.gz"
    bpe_path.write_text("stub")
    env_checkpoint = tmp_path / "env-sam31.pt"
    env_checkpoint.write_text("stub")
    monkeypatch.setenv("SAM3_CHECKPOINT_PATH", str(env_checkpoint))

    builder, _ = _patch_meta_build(monkeypatch, samgeo3)
    downloader = Mock()
    monkeypatch.setattr(samgeo3, "download_ckpt_from_hf", downloader)

    _init_meta_backend(
        samgeo3,
        bpe_path=str(bpe_path),
    )

    downloader.assert_not_called()
    assert builder.call_args.kwargs["checkpoint_path"] == str(env_checkpoint)
    assert builder.call_args.kwargs["load_from_HF"] is False


def test_sam31_missing_checkpoint_helper_has_clear_error(monkeypatch, tmp_path, samgeo3):
    """SAM 3.1 reports a clear error when no downloader path is available."""
    bpe_path = tmp_path / "bpe.txt.gz"
    bpe_path.write_text("stub")
    builder, _ = _patch_meta_build(monkeypatch, samgeo3)
    monkeypatch.setattr(samgeo3, "download_ckpt_from_hf", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ImportError, match="facebook/sam3.1"):
        _init_meta_backend(
            samgeo3,
            bpe_path=str(bpe_path),
        )

    builder.assert_not_called()


def test_save_masks_records_mask_scores(monkeypatch, samgeo3, tmp_path):
    """``save_masks`` maps each unique raster value to its confidence score,
    skipping masks filtered by size so the values stay aligned."""
    import numpy as np

    model = samgeo3.SamGeo3.__new__(samgeo3.SamGeo3)
    model.image_height = 4
    model.image_width = 4
    model.source = None
    model.scores = [0.9, 0.2, 0.6]
    small = np.zeros((4, 4), dtype=bool)
    small[0, 0] = True
    big_a = np.zeros((4, 4), dtype=bool)
    big_a[:2, :] = True
    big_b = np.zeros((4, 4), dtype=bool)
    big_b[2:, :] = True
    model.masks = [big_a, small, big_b]
    monkeypatch.setattr(
        samgeo3.common, "array_to_image", lambda *a, **k: None, raising=False
    )

    model.save_masks(output=None, min_size=2)

    assert model.mask_scores == {1: 0.9, 2: 0.6}
    assert model.objects.max() == 2

    model.save_masks(output=None, unique=False)
    assert model.mask_scores is None


def test_save_masks_promotes_dtype_so_scores_stay_aligned(monkeypatch, samgeo3):
    """More masks than uint8 can hold promote the dtype instead of wrapping
    ids, so every raster value still matches its ``mask_scores`` key."""
    import numpy as np

    n = 300
    model = samgeo3.SamGeo3.__new__(samgeo3.SamGeo3)
    model.image_height = 2
    model.image_width = n
    model.source = None
    model.scores = [i / n for i in range(n)]
    masks = []
    for i in range(n):
        m = np.zeros((2, n), dtype=bool)
        m[:, i] = True
        masks.append(m)
    model.masks = masks
    monkeypatch.setattr(
        samgeo3.common, "array_to_image", lambda *a, **k: None, raising=False
    )

    model.save_masks(output=None, dtype="uint8")

    assert model.objects.dtype == np.uint16
    np.testing.assert_array_equal(
        model.objects[0], np.arange(1, n + 1, dtype=np.uint16)
    )
    assert model.mask_scores == {i + 1: i / n for i in range(n)}

    # Signed dtypes are promoted the same way once ids would wrap.
    model.save_masks(output=None, dtype="int8")
    assert model.objects.dtype == np.uint16
    np.testing.assert_array_equal(
        model.objects[0], np.arange(1, n + 1, dtype=np.uint16)
    )

    with pytest.raises(ValueError, match="integer dtype"):
        model.save_masks(output=None, dtype="float32")


def test_samgeo3_tracks_prompt_type_and_vector_attributes(monkeypatch, samgeo3):
    """SamGeo3 tracks prompt_type across prompts and attaches Name/Type to vectors."""
    import numpy as np

    model = samgeo3.SamGeo3.__new__(samgeo3.SamGeo3)
    model.backend = "meta"
    model.device = "cpu"
    model.source = "/data/imagery/aerial_scene.tif"
    model.image_height = 8
    model.image_width = 8
    model.inference_state = {
        "masks": [np.ones((8, 8), dtype=bool)],
        "boxes": [np.array([0, 0, 4, 4])],
        "scores": [0.92],
    }
    model.processor = Mock()
    model.processor.set_text_prompt.return_value = model.inference_state
    model.processor.add_geometric_prompt.return_value = model.inference_state

    # Text prompt sets prompt_type = "text"
    model.generate_masks("building", quiet=True)
    assert model.prompt_type == "text"
    assert model._get_vector_attributes() == {
        "Name": "aerial_scene.tif",
        "Type": "text",
    }

    # Box prompt sets prompt_type = "box"
    model.generate_masks_by_boxes([[0, 0, 4, 4]])
    assert model.prompt_type == "box"
    assert model._get_vector_attributes() == {
        "Name": "aerial_scene.tif",
        "Type": "box",
    }

    # Point prompt via predict_inst sets prompt_type = "point"
    dummy_model = Mock()
    dummy_model.inst_interactive_predictor = True
    dummy_model.predict_inst.return_value = (
        np.ones((1, 8, 8), dtype=bool),
        np.array([0.88]),
        np.zeros((1, 8, 8), dtype=np.float32),
    )
    model.model = dummy_model
    model.predict_inst(point_coords=[[2, 2]], point_labels=[1])
    assert model.prompt_type == "point"

    recorded = {}

    def fake_raster_to_vector(source, output=None, simplify_tolerance=None, **kwargs):
        recorded["source"] = source
        recorded["output"] = output
        recorded["kwargs"] = kwargs
        return output or "/data/imagery/masks.gpkg"

    monkeypatch.setattr(
        samgeo3.common, "raster_to_vector", fake_raster_to_vector, raising=False
    )
    monkeypatch.setattr(
        samgeo3.common, "array_to_image", lambda *a, **kw: None, raising=False
    )
    model.save_masks("/data/imagery/masks.tif")
    out_path = model.raster_to_vector(
        "/data/imagery/masks.tif",
        "/data/imagery/masks.shp",
        attributes={"custom": "val"},
    )
    assert out_path == "/data/imagery/masks.shp"
    assert recorded["kwargs"]["attributes"] == {
        "Name": "aerial_scene.tif",
        "Type": "point",
        "custom": "val",
    }

    # Unrelated raster should not receive stale Name/Type from model state
    model.raster_to_vector(
        "/data/imagery/unrelated.tif",
        "/data/imagery/unrelated.shp",
    )
    assert recorded["kwargs"]["attributes"] is None

    model.raster_to_vector(
        "/data/imagery/unrelated.tif",
        "/data/imagery/unrelated.shp",
        attributes={"custom": "only_custom"},
    )
    assert recorded["kwargs"]["attributes"] == {"custom": "only_custom"}


def test_samgeo1_and_samgeo2_mask_metadata_and_prompt_types(monkeypatch):
    import numpy as np

    repo_root = Path(__file__).resolve().parents[1]
    samgeo_dir = repo_root / "samgeo"

    package = types.ModuleType("samgeo")
    package.__path__ = [str(samgeo_dir)]

    class _StubCommon(types.ModuleType):
        def __getattr__(self, name):
            return lambda *a, **kw: None

    common = _StubCommon("samgeo.common")
    package.common = common

    seg_mod = types.ModuleType("segment_anything")
    seg_mod.SamAutomaticMaskGenerator = Mock()
    seg_mod.SamPredictor = Mock()
    seg_mod.sam_model_registry = {}

    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "segment_anything", seg_mod)
    monkeypatch.setitem(sys.modules, "samgeo", package)
    monkeypatch.setitem(sys.modules, "samgeo.common", common)

    spec1 = importlib.util.spec_from_file_location(
        "samgeo.samgeo", samgeo_dir / "samgeo.py"
    )
    mod1 = importlib.util.module_from_spec(spec1)
    monkeypatch.setitem(sys.modules, "samgeo.samgeo", mod1)
    spec1.loader.exec_module(mod1)
    SamGeo = mod1.SamGeo

    spec2 = importlib.util.spec_from_file_location(
        "samgeo.samgeo2", samgeo_dir / "samgeo2.py"
    )
    mod2 = importlib.util.module_from_spec(spec2)
    monkeypatch.setitem(sys.modules, "samgeo.samgeo2", mod2)
    spec2.loader.exec_module(mod2)
    SamGeo2 = mod2.SamGeo2

    # --- SamGeo (SAM 1) ---
    sam1 = SamGeo.__new__(SamGeo)
    sam1.source = "/data/imagery/scene1.tif"
    sam1.image = np.zeros((8, 8, 3), dtype=np.uint8)
    sam1.masks = [{"segmentation": np.ones((8, 8), dtype=bool), "area": 64}]
    sam1.prompt_type = "automatic"
    sam1._mask_metadata = {}
    sam1.device = "cpu"
    sam1.predictor = Mock()
    sam1.predictor.predict.return_value = (
        np.ones((1, 8, 8), dtype=bool),
        np.array([0.9]),
        np.zeros((1, 8, 8), dtype=np.float32),
    )

    # Mask-only predict sets prompt_type to None rather than "automatic"
    sam1.predict(mask_input=np.zeros((1, 8, 8), dtype=np.float32))
    assert sam1.prompt_type is None

    # Box predict sets prompt_type to "box"
    sam1.predict(boxes=[0.0, 0.0, 4.0, 4.0])
    assert sam1.prompt_type == "box"

    recorded_sam1 = {}

    def fake_r2v_sam1(source, output=None, simplify_tolerance=None, **kwargs):
        recorded_sam1["kwargs"] = kwargs
        return output or "out.gpkg"

    monkeypatch.setattr(mod1, "raster_to_vector", fake_r2v_sam1, raising=False)
    sam1.save_prediction("/data/imagery/pred1.tif")
    sam1.raster_to_vector("/data/imagery/pred1.tif", "/data/imagery/pred1.gpkg")
    assert recorded_sam1["kwargs"]["attributes"] == {
        "Name": "scene1.tif",
        "Type": "box",
    }

    # Unrelated raster on SamGeo does not get stale Name/Type
    sam1.raster_to_vector("/data/imagery/other.tif", "/data/imagery/other.gpkg")
    assert recorded_sam1["kwargs"]["attributes"] is None

    # --- SamGeo2 (SAM 2) ---
    sam2 = SamGeo2.__new__(SamGeo2)
    sam2.source = "/data/imagery/scene2.tif"
    sam2.image = np.zeros((8, 8, 3), dtype=np.uint8)
    sam2.prompt_type = None
    sam2._mask_metadata = {}
    sam2.predictor = Mock()
    sam2.predictor.predict.return_value = (
        np.ones((1, 8, 8), dtype=bool),
        np.array([0.95]),
        np.zeros((1, 8, 8), dtype=np.float32),
    )
    sam2.predictor.predict_batch.return_value = (
        [np.ones((1, 8, 8), dtype=bool)],
        [np.array([0.95])],
        [np.zeros((1, 8, 8), dtype=np.float32)],
    )

    # Mask-only predict sets prompt_type = "mask"
    sam2.predict(mask_input=np.zeros((1, 8, 8), dtype=np.float32))
    assert sam2.prompt_type == "mask"

    # predict_by_points with box_batch sets prompt_type = "point_box"
    sam2.predict_by_points(
        point_coords_batch=[[2, 2]],
        box_batch=[np.array([0, 0, 4, 4])],
    )
    assert sam2.prompt_type == "point_box"

    recorded_sam2 = {}

    def fake_r2v_sam2(source, output=None, simplify_tolerance=None, **kwargs):
        recorded_sam2["kwargs"] = kwargs
        return output or "out2.gpkg"

    monkeypatch.setattr(common, "raster_to_vector", fake_r2v_sam2, raising=False)
    sam2.save_masks("/data/imagery/pred2.tif")
    sam2.raster_to_vector(
        "/data/imagery/pred2.tif",
        "/data/imagery/pred2.gpkg",
        attributes={"extra": 1},
    )
    assert recorded_sam2["kwargs"]["attributes"] == {
        "Name": "scene2.tif",
        "Type": "point_box",
        "extra": 1,
    }

    # Unrelated raster on SamGeo2 does not get stale Name/Type
    sam2.raster_to_vector(
        "/data/imagery/unrelated2.tif", "/data/imagery/unrelated2.gpkg"
    )
    assert recorded_sam2["kwargs"]["attributes"] is None


def test_qgis_samgeo_plugin_selectable_sam1_and_sam2_weights(monkeypatch):
    import numpy as np

    samgeo3 = _load_samgeo3(monkeypatch)
    samgeo3.common.prepare_image_for_sam = lambda img, bands=None: img

    # Verify SamGeo3.set_image resets prompt_type and _mask_metadata
    model = samgeo3.SamGeo3.__new__(samgeo3.SamGeo3)
    model.backend = "transformers"
    model.prompt_type = "text"
    model._mask_metadata = {"/tmp/old.tif": {"Name": "old.tif", "Type": "text"}}
    model.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    assert model.prompt_type is None
    assert model._mask_metadata == {}

    # Load qgis-samgeo-plugin/samgeo_plugin.py with lightweight stubs
    repo_root = Path(__file__).resolve().parents[1]
    plugin_dir = repo_root / "qgis-samgeo-plugin"

    qgis_mod = types.ModuleType("qgis")
    pyqt_mod = types.ModuleType("qgis.PyQt")
    qtcore_mod = types.ModuleType("qgis.PyQt.QtCore")
    qtcore_mod.Qt = Mock()
    qtcore_mod.QCoreApplication = Mock()
    qtcore_mod.QVariant = Mock()
    qtgui_mod = types.ModuleType("qgis.PyQt.QtGui")
    qtgui_mod.QIcon = Mock()

    class _StubWidgets(types.ModuleType):
        def __getattr__(self, name):
            return Mock()

    qtwidgets_mod = _StubWidgets("qgis.PyQt.QtWidgets")
    qgscore_mod = _StubWidgets("qgis.core")
    map_tools_mod = types.ModuleType("qgis_samgeo_plugin.map_tools")
    map_tools_mod.PointPromptTool = Mock()
    map_tools_mod.BoxPromptTool = Mock()

    pkg_mod = types.ModuleType("qgis_samgeo_plugin")
    pkg_mod.__path__ = [str(plugin_dir)]

    monkeypatch.setitem(sys.modules, "qgis", qgis_mod)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", pyqt_mod)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", qtcore_mod)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtGui", qtgui_mod)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtWidgets", qtwidgets_mod)
    monkeypatch.setitem(sys.modules, "qgis.core", qgscore_mod)
    monkeypatch.setitem(sys.modules, "qgis_samgeo_plugin", pkg_mod)
    monkeypatch.setitem(sys.modules, "qgis_samgeo_plugin.map_tools", map_tools_mod)

    spec_p = importlib.util.spec_from_file_location(
        "qgis_samgeo_plugin.samgeo_plugin", plugin_dir / "samgeo_plugin.py"
    )
    mod_p = importlib.util.module_from_spec(spec_p)
    monkeypatch.setitem(sys.modules, "qgis_samgeo_plugin.samgeo_plugin", mod_p)
    spec_p.loader.exec_module(mod_p)
    SamGeoPlugin = mod_p.SamGeoPlugin

    class _ComboStub:
        def __init__(self, text=""):
            self.items = []
            self._text = text
            self.enabled = True

        def clear(self):
            self.items.clear()

        def addItems(self, items):
            self.items.extend(items)
            if self.items and not self._text:
                self._text = self.items[0]

        def currentText(self):
            return self._text

        def setEnabled(self, enabled):
            self.enabled = enabled

    plugin = SamGeoPlugin.__new__(SamGeoPlugin)
    plugin.model_combo = _ComboStub("SamGeo2 (SAM2)")
    plugin.model_id_combo = _ComboStub()
    plugin.backend_combo = _ComboStub("meta")
    plugin.device_combo = _ComboStub("cpu")
    plugin.conf_spin = Mock(value=lambda: 0.5)
    plugin.interactive_check = Mock(isChecked=lambda: True)
    plugin.progress_bar = Mock()
    plugin.model_status = Mock()
    plugin.log_message = Mock()
    plugin.show_error = Mock()

    # Switching between SAM 2 and SAM 1 updates selectable weights
    plugin.on_model_version_changed("SamGeo2 (SAM2)")
    assert tuple(plugin.model_id_combo.items) == SamGeoPlugin.SAM2_MODELS
    assert plugin.backend_combo.enabled is False

    plugin.on_model_version_changed("SamGeo (SAM1)")
    assert tuple(plugin.model_id_combo.items) == SamGeoPlugin.SAM1_MODELS
    assert plugin.backend_combo.enabled is False

    plugin.on_model_version_changed("SamGeo3 (SAM3)")
    assert tuple(plugin.model_id_combo.items) == SamGeoPlugin.SAM3_MODELS
    assert plugin.backend_combo.enabled is True

    # Verify load_model passes selected SAM 2 and SAM 1 weights
    fake_lib = types.ModuleType("qgis_samgeo_plugin._samgeo_lib")
    fake_samgeo_ns = Mock()
    fake_lib.get_samgeo = lambda: fake_samgeo_ns
    monkeypatch.setitem(sys.modules, "qgis_samgeo_plugin._samgeo_lib", fake_lib)

    plugin.model_combo._text = "SamGeo2 (SAM2)"
    plugin.model_id_combo._text = "sam2-hiera-small"
    plugin.load_model()
    fake_samgeo_ns.SamGeo2.assert_called_once_with(
        model_id="sam2-hiera-small",
        device="cpu",
        automatic=False,
    )

    plugin.model_combo._text = "SamGeo (SAM1)"
    plugin.model_id_combo._text = "vit_b"
    plugin.load_model()
    fake_samgeo_ns.SamGeo.assert_called_once_with(
        model_type="vit_b",
        device="cpu",
        automatic=False,
    )
