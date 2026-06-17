"""Tests for the samgeo REST API."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("fastapi")

from samgeo.api import app, _model_cache, _image_hash_cache  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear model and image caches before each test."""
    _model_cache.clear()
    _image_hash_cache.clear()
    yield
    _model_cache.clear()
    _image_hash_cache.clear()


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary sample image file for testing."""
    from PIL import Image

    path = tmp_path / "test.tif"
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(str(path))
    return str(path)


def test_health():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_list_models():
    """Test the model listing endpoint."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "loaded" in data
    assert "sam" in data["models"]
    assert "sam2" in data["models"]
    assert "sam3" in data["models"]
    assert "facebook/sam3.1" in data["models"]["sam3"]
    assert data["loaded"] == []


def test_clear_models():
    """Test clearing the model cache."""
    response = client.delete("/models")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"


def test_automatic_missing_file():
    """Test automatic segmentation without a file."""
    response = client.post("/segment/automatic")
    assert response.status_code == 422  # Validation error


def test_automatic_invalid_model_version(sample_image_path):
    """Test automatic segmentation with invalid model version."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/automatic",
            files={"file": ("test.tif", f, "image/tiff")},
            data={"model_version": "invalid"},
        )
    assert response.status_code == 400
    assert "Invalid model_version" in response.json()["detail"]


def test_automatic_invalid_model_id(sample_image_path):
    """Test automatic segmentation with invalid model_id."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/automatic",
            files={"file": ("test.tif", f, "image/tiff")},
            data={"model_version": "sam2", "model_id": "nonexistent"},
        )
    assert response.status_code == 400
    assert "Invalid model_id" in response.json()["detail"]


def test_predict_missing_prompts(sample_image_path):
    """Test predict endpoint without any prompts."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/predict",
            files={"file": ("test.tif", f, "image/tiff")},
            data={"model_version": "sam2"},
        )
    assert response.status_code == 400
    assert "point_coords or boxes" in response.json()["detail"]


@patch("samgeo.api.get_model")
def test_predict_sam3_points_accepted(mock_get, sample_image_path):
    """SAM3 point prompts are accepted by /segment/predict (added in #508).

    SAM3 support was added to the predict endpoint, so a point prompt should
    be dispatched to ``predict_inst`` and produce a mask rather than being
    rejected. The model is mocked so the test stays fast and offline.
    """
    from threading import Lock

    mock_model = MagicMock()
    mock_model.masks = [np.ones((64, 64), dtype=np.uint8)]

    def fake_save_masks(output, **kwargs):
        from PIL import Image

        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(output)

    mock_model.save_masks = fake_save_masks
    mock_get.return_value = (mock_model, Lock(), ("sam3", "facebook/sam3", True))

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/predict",
            files={"file": ("test.tif", f, "image/tiff")},
            data={
                "model_version": "sam3",
                "point_coords": "[[10, 20]]",
                "point_labels": "[1]",
                "output_format": "png",
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    mock_model.predict_inst.assert_called_once()


def test_invalid_output_format(sample_image_path):
    """Test that invalid output_format returns 400 before segmentation runs."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/automatic",
            files={"file": ("test.tif", f, "image/tiff")},
            data={"model_version": "sam2", "output_format": "invalid"},
        )
    assert response.status_code == 400
    assert "Invalid output_format" in response.json()["detail"]


@patch("samgeo.api.get_model")
def test_automatic_png_response(mock_get, sample_image_path):
    """Test automatic segmentation returning a PNG image."""
    from threading import Lock

    mock_model = MagicMock()

    def fake_generate(source, output, **kwargs):
        from PIL import Image

        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        img.save(output)

    mock_model.generate = fake_generate
    mock_get.return_value = (mock_model, Lock(), ("sam2", "sam2-hiera-large", True))

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/automatic",
            files={"file": ("test.tif", f, "image/tiff")},
            data={"model_version": "sam2", "output_format": "png"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_get_model_caches_automatic_and_predict_separately():
    """Regression: automatic and predict instances must not share a cache slot.

    A model built for automatic mask generation has no image ``predictor``;
    reusing it for a predict request crashed with
    ``'SamGeo2' object has no attribute 'predictor'``. The cache key now
    includes the ``automatic`` flag so the two are distinct instances.
    """
    import samgeo.api as api

    with patch("samgeo.samgeo2.SamGeo2") as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock()
        auto_model, _, auto_key = api.get_model("sam2", "sam2-hiera-tiny")
        pred_model, _, pred_key = api.get_model(
            "sam2", "sam2-hiera-tiny", automatic=False
        )

    assert auto_key != pred_key
    assert auto_model is not pred_model

    # A repeat request for the same key is served from cache (no new instance).
    with patch("samgeo.samgeo2.SamGeo2") as mock_again:
        cached_model, _, _ = api.get_model("sam2", "sam2-hiera-tiny")
        mock_again.assert_not_called()
    assert cached_model is auto_model


def test_get_model_caches_distinct_configs_separately():
    """Requests differing only by model-shaping kwargs get separate instances.

    Regression: the cache key once ignored constructor kwargs, so a second
    SAM3 text request with a different ``confidence_threshold`` (or a SAM2
    request with a different ``points_per_side``) silently reused the first,
    differently-configured instance. The key now folds in the kwargs.
    """
    import samgeo.api as api

    # SAM3: two text requests differing only by confidence_threshold.
    with patch("samgeo.samgeo3.SamGeo3") as mock_sam3:
        mock_sam3.side_effect = lambda **kw: MagicMock()
        m_a, _, key_a = api.get_model("sam3", backend="meta", confidence_threshold=0.4)
        m_b, _, key_b = api.get_model("sam3", backend="meta", confidence_threshold=0.6)
        m_a2, _, _ = api.get_model("sam3", backend="meta", confidence_threshold=0.4)
    assert key_a != key_b
    assert m_a is not m_b
    assert m_a2 is m_a  # identical config is a cache hit

    # SAM2: two automatic requests differing only by points_per_side.
    with patch("samgeo.samgeo2.SamGeo2") as mock_sam2:
        mock_sam2.side_effect = lambda **kw: MagicMock()
        s_a, _, k_a = api.get_model("sam2", "sam2-hiera-tiny", points_per_side=16)
        s_b, _, k_b = api.get_model("sam2", "sam2-hiera-tiny", points_per_side=32)
    assert k_a != k_b
    assert s_a is not s_b


def test_get_model_accepts_sam31_model_id():
    """SAM 3.1 is a supported SAM3 model id and keeps its own cache key."""
    import samgeo.api as api

    with patch("samgeo.samgeo3.SamGeo3") as mock_sam3:
        mock_sam3.side_effect = lambda **kw: MagicMock()
        sam3_model, _, sam3_key = api.get_model("sam3", "facebook/sam3")
        sam31_model, _, sam31_key = api.get_model("sam3", "facebook/sam3.1")
        sam31_again, _, _ = api.get_model("sam3", "facebook/sam3.1")

    assert sam3_key != sam31_key
    assert sam3_model is not sam31_model
    assert sam31_again is sam31_model
    mock_sam3.assert_any_call(
        model_id="facebook/sam3",
        enable_inst_interactivity=True,
    )
    mock_sam3.assert_any_call(
        model_id="facebook/sam3.1",
        enable_inst_interactivity=True,
    )


def test_freeze_kwargs_is_order_independent():
    """_freeze_kwargs yields the same hashable key regardless of arg order."""
    from samgeo.api import _freeze_kwargs

    a = _freeze_kwargs({"backend": "meta", "sam_kwargs": {"x": 1, "y": 2}})
    b = _freeze_kwargs({"sam_kwargs": {"y": 2, "x": 1}, "backend": "meta"})
    assert a == b
    assert hash(a) == hash(b)
    assert _freeze_kwargs({"a": 1}) != _freeze_kwargs({"a": 2})


def test_set_image_cached_skips_for_sam2_but_not_sam3():
    """The image-encoder skip is safe for SAM/SAM2 but must be off for SAM3.

    Regression: SAM3 mutates its encoded-image state during ``generate_*``, so
    reusing it on a second request without re-encoding fails with
    ``expected scalar type BFloat16 but found Float``. ``_set_image_cached``
    therefore always re-encodes for SAM3, while still caching for SAM/SAM2.
    """
    from samgeo.api import _set_image_cached, _image_hash_cache

    # SAM2: second call with the same hash is skipped (set_image not re-run).
    sam2_model = MagicMock()
    sam2_key = ("sam2", "sam2-hiera-tiny", False)
    assert _set_image_cached(sam2_model, sam2_key, "/tmp/x.tif", "hash-a") is True
    assert _set_image_cached(sam2_model, sam2_key, "/tmp/x.tif", "hash-a") is False
    assert sam2_model.set_image.call_count == 1

    # SAM3: every call re-encodes even when the hash matches.
    sam3_model = MagicMock()
    sam3_key = ("sam3", "facebook/sam3", True)
    assert _set_image_cached(sam3_model, sam3_key, "/tmp/y.tif", "hash-b") is True
    assert _set_image_cached(sam3_model, sam3_key, "/tmp/y.tif", "hash-b") is True
    assert sam3_model.set_image.call_count == 2
