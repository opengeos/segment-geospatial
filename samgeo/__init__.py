"""Top-level package for segment-geospatial."""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "1.2.2"


from .samgeo import *  # noqa: F403
from .samgeo2 import *  # noqa: F403
from .samgeo3 import *  # noqa: F403

# Optional detectree2 support for tree crown delineation
try:
    from .detectree2 import (
        TreeCrownDelineator,
        tile_orthomosaic,
        prepare_training_data,
        stitch_predictions,
        list_pretrained_models,
        download_sample_data,
    )
except ImportError:
    pass  # detectree2 not installed
