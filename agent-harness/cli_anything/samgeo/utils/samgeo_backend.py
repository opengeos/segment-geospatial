"""Backend module: wraps the samgeo library for CLI usage.

Checks that segment-geospatial is installed, provides model instantiation,
and device detection.
"""

import importlib
import os


def check_samgeo_installed():
    """Check that segment-geospatial is installed.

    Raises:
        RuntimeError: If samgeo is not installed with install instructions.
    """
    try:
        importlib.import_module("samgeo")
    except ImportError:
        raise RuntimeError(
            "segment-geospatial is not installed. Install it with:\n"
            "  pip install segment-geospatial\n"
            "For SAM2 support: pip install segment-geospatial[samgeo2]\n"
            "For SAM3 support: pip install segment-geospatial[samgeo3]\n"
            "For all models:   pip install segment-geospatial[all]"
        )


def get_device(requested=None):
    """Detect the best available device.

    Args:
        requested: Explicit device string ('cpu', 'cuda', 'mps').
            If None, auto-detects.

    Returns:
        str: Device string.
    """
    if requested and str(requested).lower() != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_sam_model(model_type, model_id=None, device=None, automatic=True, **kwargs):
    """Instantiate a SAM model.

    Args:
        model_type: One of 'sam', 'sam2', 'sam3', 'fast_sam', 'hq_sam', 'text_sam'.
        model_id: Model identifier (varies by type).
        device: Device to use.
        automatic: Whether to use automatic mask generation mode.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        Model instance.
    """
    check_samgeo_installed()
    device = get_device(device)

    if model_type == "sam":
        from samgeo.samgeo import SamGeo

        model_id = model_id or "vit_h"
        return SamGeo(model_type=model_id, device=device, automatic=automatic, **kwargs)

    elif model_type == "sam2":
        from samgeo.samgeo2 import SamGeo2

        model_id = model_id or "sam2-hiera-large"
        return SamGeo2(model_id=model_id, device=device, automatic=automatic, **kwargs)

    elif model_type == "sam3":
        from samgeo.samgeo3 import SamGeo3

        model_id = model_id or "facebook/sam3"
        return SamGeo3(model_id=model_id, device=device, **kwargs)

    elif model_type == "fast_sam":
        from samgeo.fast_sam import SamGeo as SamGeoFast

        model_id = model_id or "FastSAM-x.pt"
        return SamGeoFast(model=model_id, device=device, automatic=automatic, **kwargs)

    elif model_type == "hq_sam":
        from samgeo.hq_sam import SamGeo as SamGeoHQ

        model_id = model_id or "vit_h"
        return SamGeoHQ(
            model_type=model_id, device=device, automatic=automatic, **kwargs
        )

    elif model_type == "text_sam":
        from samgeo.text_sam import LangSAM

        model_id = model_id or "vit_h"
        return LangSAM(model_type=model_id, **kwargs)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: sam, sam2, sam3, fast_sam, hq_sam, text_sam"
        )


def get_common_module():
    """Return the samgeo.common module.

    Returns:
        module: The samgeo.common module.
    """
    check_samgeo_installed()
    from samgeo import common

    return common


def get_rasterio():
    """Return the rasterio module.

    Returns:
        module: The rasterio module.
    """
    try:
        import rasterio

        return rasterio
    except ImportError:
        raise RuntimeError(
            "rasterio is not installed. Install it with:\n" "  pip install rasterio"
        )


def get_geopandas():
    """Return the geopandas module.

    Returns:
        module: The geopandas module.
    """
    try:
        import geopandas as gpd

        return gpd
    except ImportError:
        raise RuntimeError(
            "geopandas is not installed. Install it with:\n" "  pip install geopandas"
        )
