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


def test_predict_sam3_rejected(sample_image_path):
    """Test that predict endpoint rejects SAM3."""
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/predict",
            files={"file": ("test.tif", f, "image/tiff")},
            data={
                "model_version": "sam3",
                "point_coords": "[[100, 200]]",
                "point_labels": "[1]",
            },
        )
    assert response.status_code == 400
    assert "SAM3" in response.json()["detail"]


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
    mock_get.return_value = (mock_model, Lock())

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/segment/automatic",
            files={"file": ("test.tif", f, "image/tiff")},
            data={"model_version": "sam2", "output_format": "png"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
