"""Tests for shared model registry constants."""

import importlib.util
from pathlib import Path


def _load_model_registry():
    module_path = Path(__file__).resolve().parents[1] / "samgeo" / "model_registry.py"
    spec = importlib.util.spec_from_file_location("model_registry", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_registry = _load_model_registry()


def test_default_model_ids_are_available_models():
    """Every default model id should be listed as available."""
    for model_version, model_id in model_registry.DEFAULT_MODEL_IDS.items():
        assert model_id in model_registry.AVAILABLE_MODELS[model_version]


def test_registry_keys_stay_in_sync():
    """Shared registry sections should describe the same model versions."""
    assert set(model_registry.DEFAULT_MODEL_IDS) == set(model_registry.AVAILABLE_MODELS)
    assert set(model_registry.DEFAULT_MODEL_IDS) == set(model_registry.EXTRAS_MAP)


def test_sam31_is_registered_as_a_sam3_model():
    """SAM 3.1 is exposed as a SAM3-compatible model id."""
    assert "facebook/sam3" in model_registry.SAM3_MODEL_IDS
    assert "facebook/sam3.1" in model_registry.SAM3_MODEL_IDS
    assert (
        list(model_registry.SAM3_MODEL_IDS)
        == model_registry.AVAILABLE_MODELS["sam3"]
    )
