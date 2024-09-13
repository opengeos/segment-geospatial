import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from . import common


class SamGeo2:
    """The main class for segmenting geospatial data with the Segment Anything Model 2 (SAM2). See
    https://github.com/facebookresearch/segment-anything-2 for details.
    """

    def __init__(
        self,
        model_id: str = "sam2-hiera-large",
        device: Optional[str] = None,
        empty_cache: bool = True,
        automatic: bool = True,
        mode: str = "eval",
        hydra_overrides_extra: Optional[List[str]] = None,
        apply_postprocessing: bool = False,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        multimask_output: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SamGeo2 class.

        Args:
            model_id (str): The model ID to use. Can be one of the following: "sam2-hiera-tiny",
                "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large".
                Defaults to "sam2-hiera-large".
            device (Optional[str]): The device to use (e.g., "cpu", "cuda", "mps"). Defaults to None.
            empty_cache (bool): Whether to empty the cache. Defaults to True.
            automatic (bool): Whether to use automatic mask generation. Defaults to True.
            mode (str): The mode to use. Defaults to "eval".
            hydra_overrides_extra (Optional[List[str]]): Additional Hydra overrides. Defaults to None.
            apply_postprocessing (bool): Whether to apply postprocessing. Defaults to False.
            points_per_side (int or None): The number of points to be sampled
                along one side of the image. The total number of points is
                points_per_side**2. If None, 'point_grids' must provide explicit
                point sampling.
            points_per_batch (int): Sets the number of points run simultaneously
                by the model. Higher numbers may be faster but use more GPU memory.
            pred_iou_thresh (float): A filtering threshold in [0,1], using the
                model's predicted mask quality.
            stability_score_thresh (float): A filtering threshold in [0,1], using
                the stability of the mask under changes to the cutoff used to binarize
                the model's mask predictions.
            stability_score_offset (float): The amount to shift the cutoff when
                calculated the stability score.
            mask_threshold (float): Threshold for binarizing the mask logits
            box_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks.
            crop_n_layers (int): If >0, mask prediction will be run again on
                crops of the image. Sets the number of layers to run, where each
                layer has 2**i_layer number of image crops.
            crop_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks between different crops.
            crop_overlap_ratio (float): Sets the degree to which crops overlap.
                In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            crop_n_points_downscale_factor (int): The number of points-per-side
                sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            point_grids (list(np.ndarray) or None): A list over explicit grids
                of points used for sampling, normalized to [0,1]. The nth grid in the
                list is used in the nth crop layer. Exclusive with points_per_side.
            min_mask_region_area (int): If >0, postprocessing will be applied
                to remove disconnected regions and holes in masks with area smaller
                than min_mask_region_area. Requires opencv.
            output_mode (str): The form masks are returned in. Can be 'binary_mask',
                'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
                For large resolutions, 'binary_mask' may consume large amounts of
                memory.
            use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
            multimask_output (bool): Whether to output multimask at each point of the grid.
            **kwargs (Any): Additional keyword arguments.
        """
        if isinstance(model_id, str):
            if not model_id.startswith("facebook/"):
                model_id = f"facebook/{model_id}"
        else:
            raise ValueError("model_id must be a string")

        allowed_models = [
            "facebook/sam2-hiera-tiny",
            "facebook/sam2-hiera-small",
            "facebook/sam2-hiera-base-plus",
            "facebook/sam2-hiera-large",
        ]

        if model_id not in allowed_models:
            raise ValueError(
                f"model_id must be one of the following: {', '.join(allowed_models)}"
            )

        if device is None:
            device = common.choose_device(empty_cache=empty_cache)

        if hydra_overrides_extra is None:
            hydra_overrides_extra = []

        self.model_id = model_id
        self.device = device

        if automatic:
            self.mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
                model_id,
                device=device,
                mode=mode,
                hydra_overrides_extra=hydra_overrides_extra,
                apply_postprocessing=apply_postprocessing,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                mask_threshold=mask_threshold,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crop_n_layers,
                crop_nms_thresh=crop_nms_thresh,
                crop_overlap_ratio=crop_overlap_ratio,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                point_grids=point_grids,
                min_mask_region_area=min_mask_region_area,
                output_mode=output_mode,
                use_m2m=use_m2m,
                multimask_output=multimask_output,
                **kwargs,
            )

    def generate(
        self,
        source: Union[str, np.ndarray],
        output: Optional[str] = None,
        foreground: bool = True,
        batch: bool = False,
        batch_sample_size: Tuple[int, int] = (512, 512),
        batch_nodata_threshold: float = 1.0,
        nodata_value: Optional[int] = None,
        erosion_kernel: Optional[Tuple[int, int]] = None,
        mask_multiplier: int = 255,
        unique: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate masks for the input image.

        Args:
            source (Union[str, np.ndarray]): The path to the input image or the input image as a numpy array.
            output (Optional[str]): The path to the output image. Defaults to None.
            foreground (bool): Whether to generate the foreground mask. Defaults to True.
            batch (bool): Whether to generate masks for a batch of image tiles. Defaults to False.
            batch_sample_size (Tuple[int, int]): When batch=True, the size of the sample window when iterating over rasters.
            batch_nodata_threshold (float): Batch samples with a fraction of nodata pixels above this threshold will
                not be used to generate a mask. The default, 1.0, will skip samples with 100% nodata values. This is useful
                when rasters have large areas of nodata values which can be skipped.
            nodata_value (Optional[int]): Nodata value to use in checking batch_nodata_threshold. The default, None,
                will use the nodata value in the raster metadata if present.
            erosion_kernel (Optional[Tuple[int, int]]): The erosion kernel for filtering object masks and extract borders.
                Such as (3, 3) or (5, 5). Set to None to disable it. Defaults to None.
            mask_multiplier (int): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range, for example [0, 255]. Defaults to 255.
                The parameter is ignored if unique is True.
            unique (bool): Whether to assign a unique value to each object. Defaults to True.
                The unique value increases from 1 to the number of objects. The larger the number, the larger the object area.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the generated masks.
        """

        if isinstance(source, str):
            if source.startswith("http"):
                source = common.download_file(source)

            if not os.path.exists(source):
                raise ValueError(f"Input path {source} does not exist.")

            image = Image.open(source)
            image = np.array(image.convert("RGB"))
        elif isinstance(source, np.ndarray):
            image = source
            source = None
        else:
            raise ValueError("Input source must be either a path or a numpy array.")

        self.source = source  # Store the input image path
        self.image = image  # Store the input image as a numpy array
        mask_generator = self.mask_generator  # The automatic mask generator
        masks = mask_generator.generate(image)  # Segment the input image
        self.masks = masks  # Store the masks as a list of dictionaries
        self.batch = False

        # if output is not None:
        #     # Save the masks to the output path. The output is either a binary mask or a mask of objects with unique values.
        #     self.save_masks(
        #         output, foreground, unique, erosion_kernel, mask_multiplier, **kwargs
        #     )

        return masks
