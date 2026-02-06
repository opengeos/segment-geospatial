"""SAM 3D Objects module for 3D reconstruction from segmented masks.

This module provides tools for reconstructing 3D objects from 2D images using
SAM 3D Objects, a foundation model from Meta that converts masked objects in
images into 3D models with pose, shape, texture, and layout.

Note:
    SAM 3D Objects requires:
    - Linux 64-bit system
    - NVIDIA GPU with at least 32GB VRAM
    - HuggingFace authentication for checkpoint access

Reference:
    SAM 3D Team (2025). SAM 3D: 3Dfy Anything in Images.
    https://arxiv.org/abs/2511.16624

    Repository: https://github.com/facebookresearch/sam-3d-objects
    Website: https://ai.meta.com/sam3d/
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Installation instructions
SAM3D_INSTALL_INSTRUCTIONS = """
SAM 3D Objects Installation Instructions
========================================

Prerequisites:
- Linux 64-bit system
- NVIDIA GPU with at least 32GB VRAM
- Conda or Mamba package manager

Step 1: Clone the repository
----------------------------
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects

Step 2: Create environment
--------------------------
mamba env create -f environments/default.yml
mamba activate sam3d-objects

Step 3: Install dependencies
----------------------------
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra

Step 4: Download checkpoints (requires HuggingFace authentication)
-----------------------------------------------------------------
# First, request access at: https://huggingface.co/facebook/sam-3d-objects
# Then authenticate: huggingface-cli login

pip install 'huggingface-hub[cli]<1.0'
TAG=hf
huggingface-cli download \\
  --repo-type model \\
  --local-dir checkpoints/${TAG}-download \\
  --max-workers 1 \\
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download

For more details, see: https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md
"""


def _check_sam3d():
    """Check if SAM 3D Objects is installed and raise informative error if not."""
    try:
        # Try to import from the sam-3d-objects package
        import sys

        # Check if the sam-3d-objects notebook inference module is available
        # The package doesn't have a standard import, so we check for the inference module
        sam3d_path = os.environ.get("SAM3D_PATH")
        if sam3d_path:
            sys.path.insert(0, os.path.join(sam3d_path, "notebook"))

        from inference import Inference

        return Inference
    except ImportError:
        raise ImportError(
            "SAM 3D Objects is not installed or not configured properly.\n\n"
            f"{SAM3D_INSTALL_INSTRUCTIONS}\n\n"
            "After installation, set the SAM3D_PATH environment variable:\n"
            "  export SAM3D_PATH=/path/to/sam-3d-objects\n"
        )


def print_install_instructions():
    """Print SAM 3D Objects installation instructions."""
    print(SAM3D_INSTALL_INSTRUCTIONS)


class Sam3DReconstructor:
    """Class for 3D object reconstruction using SAM 3D Objects.

    This class provides methods for converting segmented masks into 3D models
    (Gaussian splats or meshes) using the SAM 3D Objects model.

    Attributes:
        inference: The SAM 3D inference object.
        config_path: Path to the pipeline configuration.

    Example:
        >>> from samgeo.sam3d import Sam3DReconstructor
        >>> reconstructor = Sam3DReconstructor()
        >>> output = reconstructor.reconstruct("image.png", "mask.png")
        >>> output["gs"].save_ply("object.ply")
    """

    def __init__(
        self,
        sam3d_path: Optional[str] = None,
        config_tag: str = "hf",
        compile_model: bool = False,
    ) -> None:
        """Initialize the Sam3DReconstructor.

        Args:
            sam3d_path: Path to the sam-3d-objects repository. If None, uses
                the SAM3D_PATH environment variable.
            config_tag: Configuration tag for the model ('hf' for HuggingFace).
            compile_model: Whether to compile the model for faster inference.
        """
        self.sam3d_path = sam3d_path or os.environ.get("SAM3D_PATH")
        if not self.sam3d_path:
            raise ValueError(
                "SAM3D_PATH not set. Please provide sam3d_path or set the "
                "SAM3D_PATH environment variable to the sam-3d-objects directory."
            )

        self.config_tag = config_tag
        self.compile_model = compile_model
        self._inference = None

        # Add the notebook path to sys.path for imports
        import sys

        notebook_path = os.path.join(self.sam3d_path, "notebook")
        if notebook_path not in sys.path:
            sys.path.insert(0, notebook_path)

        logger.info(f"Sam3DReconstructor initialized with path: {self.sam3d_path}")

    def _setup_inference(self) -> None:
        """Set up the SAM 3D inference pipeline."""
        if self._inference is not None:
            return

        from inference import Inference

        config_path = os.path.join(
            self.sam3d_path, "checkpoints", self.config_tag, "pipeline.yaml"
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Please download the checkpoints first. See print_install_instructions()."
            )

        logger.info(f"Loading SAM 3D model from: {config_path}")
        self._inference = Inference(config_path, compile=self.compile_model)
        logger.info("SAM 3D model loaded successfully")

    def reconstruct(
        self,
        image: Union[str, np.ndarray, "PIL.Image.Image"],
        mask: Union[str, np.ndarray, "PIL.Image.Image"],
        seed: int = 42,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Reconstruct a 3D object from an image and mask.

        Args:
            image: Input image (path, numpy array, or PIL Image).
            mask: Binary mask for the object (path, numpy array, or PIL Image).
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to the inference.

        Returns:
            Dictionary containing:
            - 'gs': Gaussian splat object (can be saved with .save_ply())
            - Additional outputs from the model
        """
        self._setup_inference()

        from inference import load_image, load_single_mask

        # Load image if path
        if isinstance(image, str):
            image = load_image(image)

        # Load mask if path
        if isinstance(mask, str):
            mask = load_single_mask(os.path.dirname(mask), index=0)
        elif isinstance(mask, np.ndarray):
            # Convert numpy mask to the expected format
            from PIL import Image

            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            mask = Image.fromarray(mask)

        logger.info("Running 3D reconstruction...")
        output = self._inference(image, mask, seed=seed, **kwargs)
        logger.info("3D reconstruction complete")

        return output

    def reconstruct_multiple(
        self,
        image: Union[str, np.ndarray, "PIL.Image.Image"],
        masks: List[Union[str, np.ndarray, "PIL.Image.Image"]],
        seed: int = 42,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Reconstruct multiple 3D objects from an image and multiple masks.

        Args:
            image: Input image (path, numpy array, or PIL Image).
            masks: List of binary masks for objects.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to the inference.

        Returns:
            List of dictionaries, each containing reconstruction outputs.
        """
        results = []
        for i, mask in enumerate(masks):
            logger.info(f"Reconstructing object {i + 1}/{len(masks)}...")
            output = self.reconstruct(image, mask, seed=seed + i, **kwargs)
            results.append(output)
        return results

    def save_ply(
        self,
        output: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Save the Gaussian splat to a PLY file.

        Args:
            output: Output dictionary from reconstruct().
            output_path: Path to save the PLY file.

        Returns:
            Path to the saved PLY file.
        """
        if "gs" not in output:
            raise ValueError("Output does not contain Gaussian splat ('gs')")

        output["gs"].save_ply(output_path)
        logger.info(f"Gaussian splat saved to: {output_path}")
        return output_path


def reconstruct_from_samgeo(
    samgeo_result: "gpd.GeoDataFrame",
    image_path: str,
    output_dir: str,
    sam3d_path: Optional[str] = None,
    max_objects: int = 10,
    seed: int = 42,
) -> List[str]:
    """Reconstruct 3D objects from SamGeo segmentation results.

    This function takes segmentation results from SamGeo (or similar) and
    reconstructs 3D models for each segmented object.

    Args:
        samgeo_result: GeoDataFrame with segmentation masks.
        image_path: Path to the original image.
        output_dir: Directory to save the 3D models.
        sam3d_path: Path to sam-3d-objects. Uses SAM3D_PATH env var if None.
        max_objects: Maximum number of objects to reconstruct.
        seed: Random seed for reproducibility.

    Returns:
        List of paths to the saved PLY files.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    os.makedirs(output_dir, exist_ok=True)

    # Initialize reconstructor
    reconstructor = Sam3DReconstructor(sam3d_path=sam3d_path)

    # Read image bounds for mask rasterization
    with rasterio.open(image_path) as src:
        transform = src.transform
        shape = (src.height, src.width)

    output_files = []
    num_objects = min(len(samgeo_result), max_objects)

    for i in range(num_objects):
        geom = samgeo_result.iloc[i].geometry
        # Rasterize the geometry to create a mask
        mask = rasterize(
            [(geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        mask = (mask > 0).astype(np.uint8) * 255

        try:
            output = reconstructor.reconstruct(image_path, mask, seed=seed + i)
            output_path = os.path.join(output_dir, f"object_{i:03d}.ply")
            reconstructor.save_ply(output, output_path)
            output_files.append(output_path)
        except Exception as e:
            logger.warning(f"Failed to reconstruct object {i}: {e}")
            continue

    logger.info(f"Reconstructed {len(output_files)}/{num_objects} objects")
    return output_files
