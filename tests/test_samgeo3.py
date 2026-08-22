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
    assert int(model.objects.max()) == n
    assert set(model.mask_scores) == set(range(1, n + 1))
    assert model.mask_scores[n] == (n - 1) / n
