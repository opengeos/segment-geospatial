"""
The source code is adapted from https://github.com/aliaksandr960/segment-anything-eo. Credit to the author Aliaksandr Hancharenka.
"""

import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from .common import *

# Available sam_kwargs:

# points_per_side: Optional[int] = 32,
# points_per_batch: int = 64,
# pred_iou_thresh: float = 0.88,
# stability_score_thresh: float = 0.95,
# stability_score_offset: float = 1.0,
# box_nms_thresh: float = 0.7,
# crop_n_layers: int = 0,
# crop_nms_thresh: float = 0.7,
# crop_overlap_ratio: float = 512 / 1500,
# crop_n_points_downscale_factor: int = 1,
# point_grids: Optional[List[np.ndarray]] = None,
# min_mask_region_area: int = 0,
# output_mode: str = "binary_mask",


class SamGeo:
    """The main class for segmenting geospatial data with the Segment Anything Model (SAM). See
    https://github.com/facebookresearch/segment-anything
    """

    def __init__(
        self,
        model_type="vit_h",
        checkpoint="sam_vit_h_4b8939.pth",
        automatic=True,
        device=None,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
        sam_kwargs=None,
    ):
        """Initialize the class.

        Args:
            model_type (str, optional): The model type. It can be one of the following: vit_h, vit_l, vit_l.
                Defaults to 'vit_h'.
            checkpoint (str, optional): The path to the checkpoint. It can be one of the following:
                sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth, sam_vit_b_01ec64.pth.
                Defaults to "sam_vit_h_4b8939.pth".
            automatic (bool, optional): Whether to use the automatic mask generator. Defaults to True.
            device (str, optional): The device to use. It can be one of the following: cpu, cuda.
                Defaults to None, which will use cuda if available.
            erosion_kernel (tuple, optional): The erosion kernel. Defaults to (3, 3).
            mask_multiplier (int, optional): The mask multiplier. Defaults to 255.
            sam_kwargs (dict, optional): The arguments for the SAM model. Defaults to None.
        """
        if not os.path.exists(checkpoint):
            print(f"Checkpoint {checkpoint} does not exist.")
            download_checkpoint(output=checkpoint)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self.sam_kwargs = sam_kwargs
        self.image = None
        self.masks = None
        self.binary_masks = None

        # First line of the SAM example
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)

        sam_kwargs = self.sam_kwargs if self.sam_kwargs is not None else {}

        if automatic:
            # Second line of the SAM example
            self.mask_generator = SamAutomaticMaskGenerator(self.sam, **sam_kwargs)
        else:
            # Second line of the SAM example
            self.predictor = SamPredictor(self.sam, **sam_kwargs)

        self.erosion_kernel = erosion_kernel
        if self.erosion_kernel is not None:
            self.erosion_kernel = np.ones(erosion_kernel, np.uint8)

        self.mask_multiplier = mask_multiplier

    def __call__(self, image):
        h, w, _ = image.shape

        # Third line of the SAM example
        masks = self.mask_generator.generate(image)

        resulting_mask = np.ones((h, w), dtype=np.uint8)
        resulting_borders = np.zeros((h, w), dtype=np.uint8)

        for m in masks:
            mask = (m["segmentation"] > 0).astype(np.uint8)
            resulting_mask += mask

            # Apply erosion to the mask
            if self.erosion_kernel is not None:
                mask_erode = cv2.erode(mask, self.erosion_kernel, iterations=1)
                mask_erode = (mask_erode > 0).astype(np.uint8)
                edge_mask = mask - mask_erode
                resulting_borders += edge_mask

        resulting_mask = (resulting_mask > 0).astype(np.uint8)
        resulting_borders = (resulting_borders > 0).astype(np.uint8)
        resulting_mask_with_borders = resulting_mask - resulting_borders
        return resulting_mask_with_borders * self.mask_multiplier

    def generate_geo(self, in_path, out_path, **kwargs):
        """Segment the input image and save the result to the output path.

        Args:
            in_path (str): The path to the input image.
            out_path (str): The path to the output image.
        """

        return tiff_to_tiff(in_path, out_path, self, **kwargs)

    def generate(self, in_path, output=None, **kwargs):
        if isinstance(in_path, str):
            if in_path.startswith("http"):
                in_path = download_file(in_path)

            if not os.path.exists(in_path):
                raise ValueError(f"Input path {in_path} does not exist.")
            image = cv2.imread(in_path)
        else:
            image = in_path
        self.image = image
        mask_generator = self.mask_generator
        masks = mask_generator.generate(image, **kwargs)
        self.masks = masks

        if output is not None:
            self.save_masks(output)

    def extract_masks(self):
        if self.masks is None:
            raise ValueError("No masks found. Please run generate() first.")

        h, w, _ = self.image.shape

        resulting_mask = np.ones((h, w), dtype=np.uint8)
        resulting_borders = np.zeros((h, w), dtype=np.uint8)

        masks = self.masks
        for m in masks:
            mask = (m["segmentation"] > 0).astype(np.uint8)
            resulting_mask += mask

            # Apply erosion to the mask
            if self.erosion_kernel is not None:
                mask_erode = cv2.erode(mask, self.erosion_kernel, iterations=1)
                mask_erode = (mask_erode > 0).astype(np.uint8)
                edge_mask = mask - mask_erode
                resulting_borders += edge_mask

        resulting_mask = (resulting_mask > 0).astype(np.uint8)
        resulting_borders = (resulting_borders > 0).astype(np.uint8)
        binary_masks = resulting_mask - resulting_borders
        binary_masks = binary_masks * self.mask_multiplier
        self.binary_masks = binary_masks

    def save_masks(self, output, **kwargs):
        self.extract_masks()
        arr_to_image(self.binary_masks, output, **kwargs)

    def show_masks(self, figsize=(20, 20), cmap="binary_r", axis="off", **kwargs):
        import matplotlib.pyplot as plt

        if self.binary_masks is None:
            self.extract_masks()

        plt.figure(figsize=figsize)
        plt.imshow(self.binary_masks, cmap=cmap)
        plt.axis(axis)
        plt.show()

    def show_anns(
        self, figsize=(20, 20), axis="off", opacity=0.35, output=None, **kwargs
    ):
        import matplotlib.pyplot as plt

        anns = self.masks

        if self.image is None:
            print("Please run generate() first.")
            return

        if anns is None or len(anns) == 0:
            return

        plt.figure(figsize=figsize)
        plt.imshow(self.image)

        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [opacity]])
            img[m] = color_mask
        ax.imshow(img)

        if "dpi" not in kwargs:
            kwargs["dpi"] = 300

        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"

        plt.axis(axis)
        if output is not None:
            plt.savefig(output, **kwargs)

        plt.show()

    def predict(self, in_path, out_path, prompts, image_format="RGB", **kwargs):
        """Segment the input image and save the result to the output path.

        Args:
            in_path (str): The path to the input image.
            out_path (str): The path to the output image.
            prompts (list): The prompts to use.
        """
        predictor = self.predictor
        predictor.set_image(in_path, image_format=image_format)
        masks, _, _ = predictor.predict(prompts, **kwargs)
        return masks

    def image_to_image(self, image, **kwargs):
        return image_to_image(image, self, **kwargs)

    def download_tms_as_tiff(self, source, pt1, pt2, zoom, dist):
        image = draw_tile(source, pt1[0], pt1[1], pt2[0], pt2[1], zoom, dist)
        return image

    def tiff_to_vector(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a gpkg file.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        tiff_to_vector(tiff_path, output, simplify_tolerance=None, **kwargs)

    def tiff_to_gpkg(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a gpkg file.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the gpkg file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        tiff_to_gpkg(tiff_path, output, simplify_tolerance=None, **kwargs)

    def tiff_to_shp(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a shapefile.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the shapefile.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        tiff_to_shp(tiff_path, output, simplify_tolerance=None, **kwargs)

    def tiff_to_geojson(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a GeoJSON file.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the GeoJSON file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        tiff_to_geojson(tiff_path, output, simplify_tolerance=None, **kwargs)
