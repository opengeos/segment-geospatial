"""Tree crown delineation using detectree2.

This module provides a high-level interface for automatic tree crown delineation
in aerial RGB imagery using the detectree2 library, which is based on Mask R-CNN
(Detectron2 implementation).

Reference:
    Ball, J.G.C., et al. (2023). Accurate delineation of individual tree crowns
    in tropical forests from aerial RGB imagery using Mask R-CNN.
    Remote Sens Ecol Conserv. 9(5):641-655.
    https://doi.org/10.1002/rse2.332

    Repository: https://github.com/PatBall1/detectree2
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Pre-trained model URLs from the detectree2 model garden
PRETRAINED_MODELS = {
    "paracou": "https://zenodo.org/records/15863800/files/250312_flexi.pth",
    "sepilok": "https://zenodo.org/records/10522461/files/230103_randresize_full.pth",
    "danum": "https://zenodo.org/records/10522461/files/230103_randresize_full.pth",
    "default": "https://zenodo.org/records/15863800/files/250312_flexi.pth",
}


def _check_detectree2():
    """Check if detectree2 is installed and raise informative error if not."""
    try:
        import detectree2

        return detectree2
    except ImportError:
        raise ImportError(
            "The detectree2 package is required for tree crown delineation. "
            "Install it with: pip install git+https://github.com/PatBall1/detectree2.git\n"
            "Note: detectree2 requires PyTorch and Detectron2 to be installed first."
        )


def _check_detectron2():
    """Check if detectron2 is installed."""
    try:
        import detectron2

        return detectron2
    except ImportError:
        raise ImportError(
            "The detectron2 package is required for tree crown delineation. "
            "Please follow the installation instructions at: "
            "https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
        )


class TreeCrownDelineator:
    """Class for automatic tree crown delineation using detectree2.

    This class provides methods for detecting and delineating individual tree
    crowns in aerial RGB imagery using pre-trained or custom Mask R-CNN models.

    Attributes:
        model_path (str): Path to the trained model weights.
        device (str): Device to run inference on ('cuda' or 'cpu').
        cfg: Detectron2 configuration object.
        predictor: Detectron2 DefaultPredictor instance.

    Example:
        >>> from samgeo.detectree2 import TreeCrownDelineator
        >>> delineator = TreeCrownDelineator()
        >>> delineator.predict("orthomosaic.tif", "crowns.gpkg")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "default",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> None:
        """Initialize the TreeCrownDelineator.

        Args:
            model_path: Path to a trained model file (.pth). If None, downloads
                a pre-trained model based on model_name.
            model_name: Name of pre-trained model to use if model_path is None.
                Options: 'paracou', 'sepilok', 'danum', 'default'.
            device: Device for inference ('cuda' or 'cpu'). If None, auto-detects.
            confidence_threshold: Minimum confidence score for predictions (0-1).
            nms_threshold: IoU threshold for non-maximum suppression (0-1).
        """
        _check_detectree2()
        _check_detectron2()

        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model_path = model_path
        self.model_name = model_name
        self._predictor = None
        self._cfg = None

        # Download model if not provided
        if self.model_path is None:
            self.model_path = self._download_model(model_name)

        logger.info(f"TreeCrownDelineator initialized with model: {self.model_path}")
        logger.info(f"Using device: {self.device}")

    def _download_model(self, model_name: str) -> str:
        """Download a pre-trained model from the model garden.

        Args:
            model_name: Name of the model to download.

        Returns:
            Path to the downloaded model file.
        """
        from samgeo.common import download_file

        if model_name not in PRETRAINED_MODELS:
            available = ", ".join(PRETRAINED_MODELS.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        url = PRETRAINED_MODELS[model_name]
        filename = os.path.basename(url)

        # Download to cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "detectree2")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, filename)

        if not os.path.exists(model_path):
            logger.info(f"Downloading pre-trained model '{model_name}' from {url}")
            download_file(url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        else:
            logger.info(f"Using cached model: {model_path}")

        return model_path

    def _setup_predictor(self) -> None:
        """Set up the Detectron2 predictor with the model configuration."""
        if self._predictor is not None:
            return

        from detectree2.models.train import setup_cfg
        from detectron2.engine import DefaultPredictor

        self._cfg = setup_cfg(update_model=self.model_path)
        self._cfg.MODEL.DEVICE = self.device
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        self._cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_threshold

        self._predictor = DefaultPredictor(self._cfg)
        logger.info("Predictor initialized successfully")

    def predict(
        self,
        image_path: str,
        output_path: str,
        tile_width: int = 40,
        tile_height: int = 40,
        buffer: int = 30,
        simplify_tolerance: float = 0.3,
        min_confidence: float = 0.5,
        iou_threshold: float = 0.6,
        output_format: str = "gpkg",
        cleanup: bool = True,
        **kwargs: Any,
    ) -> "gpd.GeoDataFrame":
        """Detect and delineate tree crowns in an orthomosaic.

        Args:
            image_path: Path to the input orthomosaic (GeoTIFF).
            output_path: Path for the output crown polygons.
            tile_width: Width of prediction tiles in meters.
            tile_height: Height of prediction tiles in meters.
            buffer: Buffer size around tiles in meters (for edge handling).
            simplify_tolerance: Tolerance for simplifying crown geometries.
            min_confidence: Minimum confidence score to keep predictions.
            iou_threshold: IoU threshold for removing overlapping crowns.
            output_format: Output format ('gpkg', 'shp', 'geojson').
            cleanup: Whether to remove temporary files after prediction.
            **kwargs: Additional arguments passed to tile_data.

        Returns:
            GeoDataFrame containing the detected tree crown polygons.
        """
        import geopandas as gpd

        from detectree2.models.outputs import (
            clean_crowns,
            project_to_geojson,
            stitch_crowns,
        )
        from detectree2.models.predict import predict_on_data
        from detectree2.preprocessing.tiling import tile_data

        # Initialize predictor
        self._setup_predictor()

        # Create temporary directory for tiles
        temp_dir = tempfile.mkdtemp(prefix="detectree2_")
        tiles_dir = os.path.join(temp_dir, "tiles")
        pred_dir = os.path.join(temp_dir, "predictions")
        geo_dir = os.path.join(temp_dir, "predictions_geo")

        try:
            logger.info(f"Tiling orthomosaic: {image_path}")
            tile_data(
                image_path,
                tiles_dir,
                buffer=buffer,
                tile_width=tile_width,
                tile_height=tile_height,
                **kwargs,
            )

            logger.info("Running predictions on tiles...")
            predict_on_data(tiles_dir, self._predictor)

            logger.info("Projecting predictions to geographic coordinates...")
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(geo_dir, exist_ok=True)
            project_to_geojson(tiles_dir, pred_dir, geo_dir)

            logger.info("Stitching and cleaning crown predictions...")
            crowns = stitch_crowns(geo_dir)
            crowns = clean_crowns(crowns, iou_threshold, confidence=min_confidence)

            # Simplify geometries
            if simplify_tolerance > 0:
                crowns = crowns.set_geometry(crowns.simplify(simplify_tolerance))

            # Save to file
            driver_map = {
                "gpkg": "GPKG",
                "shp": "ESRI Shapefile",
                "geojson": "GeoJSON",
            }
            driver = driver_map.get(output_format.lower(), "GPKG")

            crowns.to_file(output_path, driver=driver)
            logger.info(f"Crown polygons saved to: {output_path}")

            return crowns

        finally:
            if cleanup and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")

    def predict_tiles(
        self,
        tiles_dir: str,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """Run predictions on pre-tiled images.

        Args:
            tiles_dir: Directory containing tiled images.
            output_dir: Directory to save predictions. If None, saves in tiles_dir.

        Returns:
            List of paths to prediction files.
        """
        from detectree2.models.predict import predict_on_data

        self._setup_predictor()

        if output_dir is None:
            output_dir = tiles_dir

        logger.info(f"Running predictions on tiles in: {tiles_dir}")
        predict_on_data(tiles_dir, self._predictor)

        # Find prediction files
        pred_dir = os.path.join(tiles_dir, "predictions")
        if os.path.exists(pred_dir):
            pred_files = list(Path(pred_dir).glob("*.json"))
            logger.info(f"Generated {len(pred_files)} prediction files")
            return [str(f) for f in pred_files]

        return []


def tile_orthomosaic(
    image_path: str,
    output_dir: str,
    tile_width: int = 40,
    tile_height: int = 40,
    buffer: int = 30,
    crowns_path: Optional[str] = None,
    threshold: float = 0.6,
    mode: str = "rgb",
    **kwargs: Any,
) -> str:
    """Tile an orthomosaic for training or prediction.

    Args:
        image_path: Path to the input orthomosaic (GeoTIFF).
        output_dir: Directory to save the tiles.
        tile_width: Width of tiles in meters.
        tile_height: Height of tiles in meters.
        buffer: Buffer size around tiles in meters.
        crowns_path: Path to crown polygons (for training data preparation).
        threshold: Minimum crown coverage to keep a tile (when crowns provided).
        mode: Image mode ('rgb' or 'ms' for multispectral).
        **kwargs: Additional arguments passed to tile_data.

    Returns:
        Path to the output directory containing tiles.
    """
    _check_detectree2()

    import geopandas as gpd
    import rasterio

    from detectree2.preprocessing.tiling import tile_data

    crowns = None
    if crowns_path is not None:
        # Read crowns and match CRS to image
        with rasterio.open(image_path) as src:
            img_crs = src.crs
        crowns = gpd.read_file(crowns_path)
        crowns = crowns.to_crs(img_crs)

    logger.info(f"Tiling orthomosaic: {image_path}")
    tile_data(
        image_path,
        output_dir,
        buffer=buffer,
        tile_width=tile_width,
        tile_height=tile_height,
        crowns=crowns,
        threshold=threshold,
        mode=mode,
        **kwargs,
    )

    logger.info(f"Tiles saved to: {output_dir}")
    return output_dir


def prepare_training_data(
    image_path: str,
    crowns_path: str,
    output_dir: str,
    tile_width: int = 40,
    tile_height: int = 40,
    buffer: int = 30,
    threshold: float = 0.6,
    test_fraction: float = 0.15,
    mode: str = "rgb",
) -> Tuple[str, str]:
    """Prepare training and test data for detectree2.

    Args:
        image_path: Path to the input orthomosaic (GeoTIFF).
        crowns_path: Path to manually delineated crown polygons.
        output_dir: Directory to save the training data.
        tile_width: Width of tiles in meters.
        tile_height: Height of tiles in meters.
        buffer: Buffer size around tiles in meters.
        threshold: Minimum crown coverage to keep a tile.
        test_fraction: Fraction of data to use for testing (0-1).
        mode: Image mode ('rgb' or 'ms' for multispectral).

    Returns:
        Tuple of (train_dir, test_dir) paths.
    """
    _check_detectree2()

    from detectree2.preprocessing.tiling import tile_data, to_traintest_folders

    # First tile the data
    tile_orthomosaic(
        image_path,
        output_dir,
        tile_width=tile_width,
        tile_height=tile_height,
        buffer=buffer,
        crowns_path=crowns_path,
        threshold=threshold,
        mode=mode,
    )

    # Split into train/test
    logger.info(f"Splitting data into train/test (test fraction: {test_fraction})")
    to_traintest_folders(output_dir, output_dir, test_frac=test_fraction)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    logger.info(f"Training data: {train_dir}")
    logger.info(f"Test data: {test_dir}")

    return train_dir, test_dir


def stitch_predictions(
    geo_predictions_dir: str,
    output_path: str,
    iou_threshold: float = 0.6,
    min_confidence: float = 0.5,
    simplify_tolerance: float = 0.3,
    output_format: str = "gpkg",
) -> "gpd.GeoDataFrame":
    """Stitch and clean tile predictions into a single crown map.

    Args:
        geo_predictions_dir: Directory containing geo-referenced predictions.
        output_path: Path for the output crown polygons.
        iou_threshold: IoU threshold for removing overlapping crowns.
        min_confidence: Minimum confidence score to keep predictions.
        simplify_tolerance: Tolerance for simplifying crown geometries.
        output_format: Output format ('gpkg', 'shp', 'geojson').

    Returns:
        GeoDataFrame containing the stitched and cleaned crown polygons.
    """
    _check_detectree2()

    import geopandas as gpd

    from detectree2.models.outputs import clean_crowns, stitch_crowns

    logger.info(f"Stitching predictions from: {geo_predictions_dir}")
    crowns = stitch_crowns(geo_predictions_dir)

    logger.info("Cleaning overlapping crowns...")
    crowns = clean_crowns(crowns, iou_threshold, confidence=min_confidence)

    if simplify_tolerance > 0:
        crowns = crowns.set_geometry(crowns.simplify(simplify_tolerance))

    # Save to file
    driver_map = {
        "gpkg": "GPKG",
        "shp": "ESRI Shapefile",
        "geojson": "GeoJSON",
    }
    driver = driver_map.get(output_format.lower(), "GPKG")

    crowns.to_file(output_path, driver=driver)
    logger.info(f"Crown polygons saved to: {output_path}")

    return crowns


def list_pretrained_models() -> Dict[str, str]:
    """List available pre-trained models.

    Returns:
        Dictionary mapping model names to their download URLs.
    """
    return PRETRAINED_MODELS.copy()


def download_sample_data(output_dir: str = "./detectree2_sample") -> str:
    """Download sample data for testing detectree2.

    Args:
        output_dir: Directory to save the sample data.

    Returns:
        Path to the output directory.
    """
    from samgeo.common import download_file

    sample_url = "https://zenodo.org/records/8136161/files/Paracou_sample.zip"

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "sample.zip")

    logger.info(f"Downloading sample data from {sample_url}")
    download_file(sample_url, zip_path)

    # Extract
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)
    logger.info(f"Sample data extracted to: {output_dir}")

    return output_dir
