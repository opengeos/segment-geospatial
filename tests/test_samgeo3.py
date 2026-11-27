"""Fast constructor tests for SAM3 model selection logic."""

import importlib.util
import sys
import types
import warnings
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


@pytest.mark.parametrize(
    "capability, expected",
    [
        ((7, 0), "float16"),
        ((7, 5), "float16"),
        ((8, 0), "bfloat16"),
        ((8, 9), "bfloat16"),
    ],
)
def test_autocast_dtype_uses_fp16_before_ampere(
    monkeypatch, samgeo3, capability, expected
):
    """GPUs below compute capability 8.0 get fp16, newer ones keep bf16."""
    torch = pytest.importorskip("torch")
    get_capability = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    monkeypatch.setattr(torch.cuda, "current_device", Mock(return_value=0))

    assert samgeo3._cuda_autocast_dtype("cuda") == getattr(torch, expected)
    assert samgeo3._cuda_autocast_dtype("cuda:1") == getattr(torch, expected)
    assert [c.args[0] for c in get_capability.call_args_list] == [0, 1]


def test_meta_autocast_uses_fp16_on_turing(monkeypatch, samgeo3):
    """The Meta backend autocasts to fp16 on a T4-class GPU."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "get_device_capability", Mock(return_value=(7, 5)))
    monkeypatch.setattr(torch.cuda, "current_device", Mock(return_value=0))
    model = samgeo3.SamGeo3.__new__(samgeo3.SamGeo3)
    model.backend = "meta"
    model.device = "cuda"

    with warnings.catch_warnings():
        # torch warns that CUDA autocast is disabled when there is no GPU
        warnings.simplefilter("ignore", UserWarning)
        ctx = model._meta_autocast()
    assert ctx.fast_dtype == torch.float16


def _sam3_addmm_act(activation, linear, mat1):
    """Same as sam3.perflib.fused.addmm_act: always casts to bf16."""
    import torch

    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    use_gelu = activation in (torch.nn.functional.gelu, torch.nn.GELU)
    y = torch.ops.aten._addmm_activation(
        linear.bias.detach().bfloat16(),
        mat1.bfloat16().view(-1, mat1.shape[-1]),
        linear.weight.detach().bfloat16().t(),
        use_gelu=use_gelu,
    )
    return y.view(mat1.shape[:-1] + (y.shape[-1],))


@pytest.mark.parametrize("act", ["GELU", "ReLU"])
def test_addmm_act_wrapper_runs_in_fp16(samgeo3, act):
    """Under fp16 autocast the wrapped kernel returns fp16 close to fp32."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    linear = torch.nn.Linear(16, 32)
    x = torch.randn(2, 5, 16)
    activation = getattr(torch.nn, act)
    original = Mock(side_effect=AssertionError("should not call the bf16 kernel"))
    wrapped = samgeo3._make_fp16_aware_addmm_act(original)

    with torch.no_grad():
        expected = activation()(linear(x))
        with torch.autocast("cpu", dtype=torch.float16):
            out = wrapped(activation, linear, x)

    original.assert_not_called()
    assert out.dtype == torch.float16
    assert out.shape == (2, 5, 32)
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)


def test_addmm_act_wrapper_keeps_bf16_behavior(samgeo3):
    """Without fp16 autocast the wrapper behaves exactly like sam3's kernel."""
    torch = pytest.importorskip("torch")
    linear = torch.nn.Linear(8, 8)
    x = torch.randn(3, 8)
    wrapped = samgeo3._make_fp16_aware_addmm_act(_sam3_addmm_act)

    with torch.no_grad():
        assert wrapped(torch.nn.GELU, linear, x).dtype == torch.bfloat16
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert wrapped(torch.nn.GELU, linear, x).dtype == torch.bfloat16
    with pytest.raises(ValueError, match="grad"):
        wrapped(torch.nn.GELU, linear, x)


def test_install_addmm_act_wrapper_only_once(monkeypatch, samgeo3):
    """Installing the wrapper twice does not wrap it a second time."""
    vitdet = types.ModuleType("sam3.model.vitdet")
    vitdet.addmm_act = _sam3_addmm_act
    sam3_model = types.ModuleType("sam3.model")
    sam3_model.vitdet = vitdet
    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", sam3_model)
    monkeypatch.setitem(sys.modules, "sam3.model.vitdet", vitdet)

    samgeo3._install_fp16_aware_addmm_act()
    wrapped = vitdet.addmm_act
    samgeo3._install_fp16_aware_addmm_act()

    assert vitdet.addmm_act is wrapped
    assert wrapped.__wrapped__ is _sam3_addmm_act


def _box_prompt_model(samgeo3, backend):
    """Build a bare SamGeo3 with a mocked processor for box prompt tests."""
    model = samgeo3.SamGeo3.__new__(samgeo3.SamGeo3)
    model.backend = backend
    model.device = "cpu"
    model.source = None
    model.image_width = 100
    model.image_height = 50
    model.confidence_threshold = 0.5
    model.mask_threshold = 0.5
    model._convert_results_to_numpy = Mock()
    result = {"masks": [], "boxes": [], "scores": []}
    model.processor = Mock()
    if backend == "meta":
        model.inference_state = {}
        model.processor.set_text_prompt.return_value = result
        model.processor.add_geometric_prompt.return_value = result
    else:
        model.pil_image = object()
        model.model = Mock()
        model.processor.return_value.to.return_value = {}
        model.processor.post_process_instance_segmentation.return_value = [result]
    return model


def test_boxes_with_text_prompt_meta_sets_text_before_boxes(samgeo3):
    """The Meta backend sets the text prompt, then adds each box on top of it."""
    model = _box_prompt_model(samgeo3, "meta")

    model.generate_masks_by_boxes(
        [[10, 10, 30, 20], [50, 20, 70, 40]],
        box_labels=[True, False],
        prompt="building",
    )

    calls = [c[0] for c in model.processor.method_calls]
    assert calls == [
        "reset_all_prompts",
        "set_text_prompt",
        "add_geometric_prompt",
        "add_geometric_prompt",
    ]
    assert model.processor.set_text_prompt.call_args.kwargs["prompt"] == "building"
    box_calls = model.processor.add_geometric_prompt.call_args_list
    assert [c.kwargs["label"] for c in box_calls] == [True, False]
    assert box_calls[0].kwargs["box"] == [0.2, 0.3, 0.2, 0.2]


def test_boxes_without_text_prompt_meta_skips_text(samgeo3):
    """Without a text prompt the Meta backend uses the boxes alone."""
    model = _box_prompt_model(samgeo3, "meta")

    model.generate_masks_by_boxes([[10, 10, 30, 20]])

    model.processor.set_text_prompt.assert_not_called()
    model.processor.add_geometric_prompt.assert_called_once()


def test_boxes_with_text_prompt_transformers_passes_text_and_labels(samgeo3):
    """The Transformers backend sends the text, boxes and box labels together."""
    pytest.importorskip("torch")
    model = _box_prompt_model(samgeo3, "transformers")

    model.generate_masks_by_boxes(
        [[10, 10, 30, 20], [50, 20, 70, 40]],
        box_labels=[True, False],
        prompt="building",
    )

    kwargs = model.processor.call_args.kwargs
    assert kwargs["text"] == "building"
    assert kwargs["input_boxes"] == [
        [[10.0, 10.0, 30.0, 20.0], [50.0, 20.0, 70.0, 40.0]]
    ]
    assert kwargs["input_boxes_labels"] == [[1, 0]]


def test_boxes_transformers_sends_negative_labels_without_text(samgeo3):
    """Negative boxes are no longer silently treated as positive."""
    pytest.importorskip("torch")
    model = _box_prompt_model(samgeo3, "transformers")

    model.generate_masks_by_boxes([[10, 10, 30, 20]], box_labels=[False])

    kwargs = model.processor.call_args.kwargs
    assert "text" not in kwargs
    assert kwargs["input_boxes_labels"] == [[0]]


@pytest.mark.parametrize("backend", ["meta", "transformers"])
def test_text_prompt_with_no_boxes(samgeo3, backend):
    """An empty box list with a text prompt falls back to text only."""
    pytest.importorskip("torch")
    model = _box_prompt_model(samgeo3, backend)

    model.generate_masks_by_boxes([], prompt="building")

    if backend == "meta":
        model.processor.set_text_prompt.assert_called_once()
        model.processor.add_geometric_prompt.assert_not_called()
    else:
        kwargs = model.processor.call_args.kwargs
        assert kwargs["text"] == "building"
        assert "input_boxes" not in kwargs


def test_no_boxes_and_no_text_prompt_raises(samgeo3):
    """Calling with neither boxes nor a prompt fails with a clear message."""
    model = _box_prompt_model(samgeo3, "meta")

    with pytest.raises(ValueError, match="at least one box or a text prompt"):
        model.generate_masks_by_boxes([])


def test_boxes_accept_geojson_feature_collection(samgeo3):
    """Drawn regions (a GeoJSON FeatureCollection) are reduced to their bounds."""
    pytest.importorskip("geopandas")
    rois = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[1, 2], [3, 2], [3, 5], [1, 5], [1, 2]]],
                },
            }
        ],
    }

    boxes, crs = samgeo3.SamGeo3._boxes_to_list(rois, "EPSG:4326")

    assert boxes == [[1.0, 2.0, 3.0, 5.0]]
    assert crs == "EPSG:4326"


@pytest.mark.parametrize("prompt", [None, "building"])
def test_empty_feature_collection(samgeo3, prompt):
    """Nothing drawn on the map: text-only with a prompt, a clear error without."""
    pytest.importorskip("torch")
    model = _box_prompt_model(samgeo3, "meta")
    empty = {"type": "FeatureCollection", "features": []}

    if prompt is None:
        with pytest.raises(ValueError, match="at least one box or a text prompt"):
            model.generate_masks_by_boxes(empty, box_crs="EPSG:4326")
    else:
        model.generate_masks_by_boxes(empty, box_crs="EPSG:4326", prompt=prompt)
        model.processor.set_text_prompt.assert_called_once()
        model.processor.add_geometric_prompt.assert_not_called()


def test_boxes_to_list_does_not_modify_caller_geodataframe(samgeo3):
    """Setting the CRS on a CRS-less GeoDataFrame works on a copy."""
    gpd = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")
    gdf = gpd.GeoDataFrame(geometry=[shapely_geometry.box(1, 2, 3, 5)])

    boxes, crs = samgeo3.SamGeo3._boxes_to_list(gdf, "EPSG:4326")

    assert boxes == [[1.0, 2.0, 3.0, 5.0]]
    assert crs == "EPSG:4326"
    assert gdf.crs is None


def test_box_crs_without_georeference_warns(samgeo3):
    """A CRS on boxes for a non-georeferenced image is flagged, not silently used."""
    pytest.importorskip("torch")
    model = _box_prompt_model(samgeo3, "meta")

    with pytest.warns(UserWarning, match="no georeference"):
        model.generate_masks_by_boxes([[10, 10, 30, 20]], box_crs="EPSG:4326")
