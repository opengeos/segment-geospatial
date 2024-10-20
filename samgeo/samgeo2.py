import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

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
        video: bool = False,
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
        multimask_output: bool = False,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
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
            video (bool): Whether to use video prediction. Defaults to False.
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
                Defaults to False.
            max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
                the maximum area of max_hole_area in low_res_masks.
            max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
                the maximum area of max_sprinkle_area in low_res_masks.
            **kwargs (Any): Additional keyword arguments to pass to
                SAM2AutomaticMaskGenerator.from_pretrained() or SAM2ImagePredictor.from_pretrained().
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
        self.model_version = "sam2"
        self.device = device

        if video:
            automatic = False

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
        elif video:
            self.predictor = SAM2VideoPredictor.from_pretrained(
                model_id,
                device=device,
                mode=mode,
                hydra_overrides_extra=hydra_overrides_extra,
                apply_postprocessing=apply_postprocessing,
                **kwargs,
            )
        else:
            self.predictor = SAM2ImagePredictor.from_pretrained(
                model_id,
                device=device,
                mode=mode,
                hydra_overrides_extra=hydra_overrides_extra,
                apply_postprocessing=apply_postprocessing,
                mask_threshold=mask_threshold,
                max_hole_area=max_hole_area,
                max_sprinkle_area=max_sprinkle_area,
                **kwargs,
            )

    def generate(
        self,
        source: Union[str, np.ndarray],
        output: Optional[str] = None,
        foreground: bool = True,
        erosion_kernel: Optional[Tuple[int, int]] = None,
        mask_multiplier: int = 255,
        unique: bool = True,
        min_size: int = 0,
        max_size: int = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate masks for the input image.

        Args:
            source (Union[str, np.ndarray]): The path to the input image or the
                input image as a numpy array.
            output (Optional[str]): The path to the output image. Defaults to None.
            foreground (bool): Whether to generate the foreground mask. Defaults
                to True.
            erosion_kernel (Optional[Tuple[int, int]]): The erosion kernel for
                filtering object masks and extract borders.
                Such as (3, 3) or (5, 5). Set to None to disable it. Defaults to None.
            mask_multiplier (int): The mask multiplier for the output mask,
                which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range,
                for example [0, 255]. Defaults to 255.
                The parameter is ignored if unique is True.
            unique (bool): Whether to assign a unique value to each object.
                Defaults to True.
                The unique value increases from 1 to the number of objects. The
                larger the number, the larger the object area.
            min_size (int): The minimum size of the object. Defaults to 0.
            max_size (int): The maximum size of the object. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the generated masks.
        """

        if isinstance(source, str):
            if source.startswith("http"):
                source = common.download_file(source)

            if not os.path.exists(source):
                raise ValueError(f"Input path {source} does not exist.")

            image = cv2.imread(source)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        self._min_size = min_size
        self._max_size = max_size

        if output is not None:
            # Save the masks to the output path. The output is either a binary mask or a mask of objects with unique values.
            self.save_masks(
                output,
                foreground,
                unique,
                erosion_kernel,
                mask_multiplier,
                min_size,
                max_size,
                **kwargs,
            )

    def save_masks(
        self,
        output: Optional[str] = None,
        foreground: bool = True,
        unique: bool = True,
        erosion_kernel: Optional[Tuple[int, int]] = None,
        mask_multiplier: int = 255,
        min_size: int = 0,
        max_size: int = None,
        **kwargs: Any,
    ) -> None:
        """Save the masks to the output path. The output is either a binary mask
        or a mask of objects with unique values.

        Args:
            output (str, optional): The path to the output image. Defaults to
                None, saving the masks to SamGeo.objects.
            foreground (bool, optional): Whether to generate the foreground mask.
                Defaults to True.
            unique (bool, optional): Whether to assign a unique value to each
                object. Defaults to True.
            erosion_kernel (tuple, optional): The erosion kernel for filtering
                object masks and extract borders.
                Such as (3, 3) or (5, 5). Set to None to disable it. Defaults to
                None.
            mask_multiplier (int, optional): The mask multiplier for the output
                mask, which is usually a binary mask [0, 1]. You can use this
                parameter to scale the mask to a larger range, for example
                [0, 255]. Defaults to 255.
            min_size (int, optional): The minimum size of the object. Defaults to 0.
            max_size (int, optional): The maximum size of the object. Defaults to None.
            **kwargs: Additional keyword arguments for common.array_to_image().
        """

        if self.masks is None:
            raise ValueError("No masks found. Please run generate() first.")

        h, w, _ = self.image.shape
        masks = self.masks

        # Set output image data type based on the number of objects
        if len(masks) < 255:
            dtype = np.uint8
        elif len(masks) < 65535:
            dtype = np.uint16
        else:
            dtype = np.uint32

        # Generate a mask of objects with unique values
        if unique:
            # Sort the masks by area in descending order
            sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

            # Create an output image with the same size as the input image
            objects = np.zeros(
                (
                    sorted_masks[0]["segmentation"].shape[0],
                    sorted_masks[0]["segmentation"].shape[1],
                )
            )
            # Assign a unique value to each object
            count = len(sorted_masks)
            for index, ann in enumerate(sorted_masks):
                m = ann["segmentation"]
                if min_size > 0 and ann["area"] < min_size:
                    continue
                if max_size is not None and ann["area"] > max_size:
                    continue
                objects[m] = count - index

        # Generate a binary mask
        else:
            if foreground:  # Extract foreground objects only
                resulting_mask = np.zeros((h, w), dtype=dtype)
            else:
                resulting_mask = np.ones((h, w), dtype=dtype)
            resulting_borders = np.zeros((h, w), dtype=dtype)

            for m in masks:
                if min_size > 0 and m["area"] < min_size:
                    continue
                if max_size is not None and m["area"] > max_size:
                    continue
                mask = (m["segmentation"] > 0).astype(dtype)
                resulting_mask += mask

                # Apply erosion to the mask
                if erosion_kernel is not None:
                    mask_erode = cv2.erode(mask, erosion_kernel, iterations=1)
                    mask_erode = (mask_erode > 0).astype(dtype)
                    edge_mask = mask - mask_erode
                    resulting_borders += edge_mask

            resulting_mask = (resulting_mask > 0).astype(dtype)
            resulting_borders = (resulting_borders > 0).astype(dtype)
            objects = resulting_mask - resulting_borders
            objects = objects * mask_multiplier

        objects = objects.astype(dtype)
        self.objects = objects

        if output is not None:  # Save the output image
            common.array_to_image(self.objects, output, self.source, **kwargs)

    def show_masks(
        self,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "binary_r",
        axis: str = "off",
        foreground: bool = True,
        **kwargs: Any,
    ) -> None:
        """Show the binary mask or the mask of objects with unique values.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            cmap (str, optional): The colormap. Defaults to "binary_r".
            axis (str, optional): Whether to show the axis. Defaults to "off".
            foreground (bool, optional): Whether to show the foreground mask only.
                Defaults to True.
            **kwargs: Other arguments for save_masks().
        """

        import matplotlib.pyplot as plt

        if self.objects is None:
            self.save_masks(foreground=foreground, **kwargs)

        plt.figure(figsize=figsize)
        plt.imshow(self.objects, cmap=cmap)
        plt.axis(axis)
        plt.show()

    def show_anns(
        self,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        alpha: float = 0.35,
        output: Optional[str] = None,
        blend: bool = True,
        **kwargs: Any,
    ) -> None:
        """Show the annotations (objects with random color) on the input image.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            axis (str, optional): Whether to show the axis. Defaults to "off".
            alpha (float, optional): The alpha value for the annotations. Defaults to 0.35.
            output (str, optional): The path to the output image. Defaults to None.
            blend (bool, optional): Whether to show the input image. Defaults to True.
        """

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
            if hasattr(self, "_min_size") and (ann["area"] < self._min_size):
                continue
            if (
                hasattr(self, "_max_size")
                and isinstance(self._max_size, int)
                and ann["area"] > self._max_size
            ):
                continue
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [alpha]])
            img[m] = color_mask
        ax.imshow(img)

        if "dpi" not in kwargs:
            kwargs["dpi"] = 100

        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"

        plt.axis(axis)

        self.annotations = (img[:, :, 0:3] * 255).astype(np.uint8)

        if output is not None:
            if blend:
                array = common.blend_images(
                    self.annotations, self.image, alpha=alpha, show=False
                )
            else:
                array = self.annotations
            common.array_to_image(array, output, self.source)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[str, np.ndarray, Image],
    ) -> None:
        """Set the input image as a numpy array.

        Args:
            image (Union[str, np.ndarray, Image]): The input image as a path,
                a numpy array, or an Image.
        """
        if isinstance(image, str):
            if image.startswith("http"):
                image = common.download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image

            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = image
        elif isinstance(image, np.ndarray) or isinstance(image, Image):
            pass
        else:
            raise ValueError("Input image must be either a path or a numpy array.")

        self.predictor.set_image(image)

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray, str, Image]],
    ) -> None:
        """Set a batch of images for prediction.

        Args:
            image_list (List[Union[np.ndarray, str, Image]]): A list of images,
            which can be numpy arrays, file paths, or PIL images.

        Raises:
            ValueError: If an input image path does not exist or if the input
                image type is not supported.
        """
        images = []
        for image in image_list:
            if isinstance(image, str):
                if image.startswith("http"):
                    image = common.download_file(image)

                if not os.path.exists(image):
                    raise ValueError(f"Input path {image} does not exist.")

                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image):
                image = np.array(image)
            elif isinstance(image, np.ndarray):
                pass
            else:
                raise ValueError("Input image must be either a path or a numpy array.")

            images.append(image)

        self.predictor.set_image_batch(images)

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
        normalize_coords: bool = True,
        point_crs: Optional[str] = None,
        output: Optional[str] = None,
        index: Optional[int] = None,
        mask_multiplier: int = 255,
        dtype: str = "float32",
        return_results: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the mask for the input image.

        Args:
            point_coords (np.ndarray, optional): The point coordinates. Defaults to None.
            point_labels (np.ndarray, optional): The point labels. Defaults to None.
            boxes (list | np.ndarray, optional): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray, optional): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form 1xHxW, where for SAM, H=W=256.
                multimask_output (bool, optional): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            multimask_output (bool, optional): Whether to output multimask at each
                point of the grid. Defaults to False.
            return_logits (bool, optional): If true, returns un-thresholded masks logits
                instead of a binary mask.
            normalize_coords (bool, optional): Whether to normalize the coordinates.
                Defaults to True.
            point_crs (str, optional): The coordinate reference system (CRS) of the point prompts.
            output (str, optional): The path to the output image. Defaults to None.
            index (index, optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            mask_multiplier (int, optional): The mask multiplier for the output mask,
                which is usually a binary mask [0, 1].
            dtype (np.dtype, optional): The data type of the output image. Defaults to np.float32.
            return_results (bool, optional): Whether to return the predicted masks,
                scores, and logits. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The mask, the multimask,
                and the logits.
        """
        import geopandas as gpd

        out_of_bounds = []

        if isinstance(boxes, str):
            gdf = gpd.read_file(boxes)
            if gdf.crs is not None:
                gdf = gdf.to_crs("epsg:4326")
            boxes = gdf.geometry.bounds.values.tolist()
        elif isinstance(boxes, dict):
            import json

            geojson = json.dumps(boxes)
            gdf = gpd.read_file(geojson, driver="GeoJSON")
            boxes = gdf.geometry.bounds.values.tolist()

        if isinstance(point_coords, str):
            point_coords = common.vector_to_geojson(point_coords)

        if isinstance(point_coords, dict):
            point_coords = common.geojson_to_coords(point_coords)

        if hasattr(self, "point_coords"):
            point_coords = self.point_coords

        if hasattr(self, "point_labels"):
            point_labels = self.point_labels

        if (point_crs is not None) and (point_coords is not None):
            point_coords, out_of_bounds = common.coords_to_xy(
                self.source, point_coords, point_crs, return_out_of_bounds=True
            )

        if isinstance(point_coords, list):
            point_coords = np.array(point_coords)

        if point_coords is not None:
            if point_labels is None:
                point_labels = [1] * len(point_coords)
            elif isinstance(point_labels, int):
                point_labels = [point_labels] * len(point_coords)

        if isinstance(point_labels, list):
            if len(point_labels) != len(point_coords):
                if len(point_labels) == 1:
                    point_labels = point_labels * len(point_coords)
                elif len(out_of_bounds) > 0:
                    print(f"Removing {len(out_of_bounds)} out-of-bound points.")
                    point_labels_new = []
                    for i, p in enumerate(point_labels):
                        if i not in out_of_bounds:
                            point_labels_new.append(p)
                    point_labels = point_labels_new
                else:
                    raise ValueError(
                        "The length of point_labels must be equal to the length of point_coords."
                    )
            point_labels = np.array(point_labels)

        predictor = self.predictor

        input_boxes = None
        if isinstance(boxes, list) and (point_crs is not None):
            coords = common.bbox_to_xy(self.source, boxes, point_crs)
            input_boxes = np.array(coords)

        elif isinstance(boxes, list) and (point_crs is None):
            input_boxes = np.array(boxes)

        self.boxes = input_boxes

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )

        self.masks = masks
        self.scores = scores
        self.logits = logits

        if output is not None:
            if boxes is None or (not isinstance(boxes[0], list)):
                self.save_prediction(output, index, mask_multiplier, dtype, **kwargs)
            else:
                self.tensor_to_numpy(
                    index, output, mask_multiplier, dtype, save_args=kwargs
                )

        if return_results:
            return masks, scores, logits

    def predict_by_points(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
        normalize_coords=True,
        point_crs: Optional[str] = None,
        output: Optional[str] = None,
        index: Optional[int] = None,
        unique: bool = True,
        mask_multiplier: int = 255,
        dtype: str = "int32",
        return_results: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the mask for the input image.

        Args:
            point_coords (np.ndarray, optional): The point coordinates. Defaults to None.
            point_labels (np.ndarray, optional): The point labels. Defaults to None.
            boxes (list | np.ndarray, optional): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray, optional): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form 1xHxW, where for SAM, H=W=256.
                multimask_output (bool, optional): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            multimask_output (bool, optional): Whether to output multimask at each
                point of the grid. Defaults to True.
            return_logits (bool, optional): If true, returns un-thresholded masks logits
                instead of a binary mask.
            normalize_coords (bool, optional): Whether to normalize the coordinates.
                Defaults to True.
            point_crs (str, optional): The coordinate reference system (CRS) of the point prompts.
            output (str, optional): The path to the output image. Defaults to None.
            index (index, optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            mask_multiplier (int, optional): The mask multiplier for the output mask,
                which is usually a binary mask [0, 1].
            dtype (np.dtype, optional): The data type of the output image. Defaults to np.int32.
            return_results (bool, optional): Whether to return the predicted masks,
                scores, and logits. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The mask, the multimask,
                and the logits.
        """
        import geopandas as gpd

        if hasattr(self, "image_batch") and self.image_batch is not None:
            pass
        elif self.image is not None:
            self.predictor.set_image_batch([self.image])
            setattr(self, "image_batch", [self.image])
        else:
            raise ValueError("Please set the input image first using set_image().")

        if isinstance(point_coords_batch, dict):
            point_coords_batch = gpd.GeoDataFrame.from_features(point_coords_batch)

        if isinstance(point_coords_batch, str) or isinstance(
            point_coords_batch, gpd.GeoDataFrame
        ):
            if isinstance(point_coords_batch, str):
                gdf = gpd.read_file(point_coords_batch)
            else:
                gdf = point_coords_batch
            if gdf.crs is None and (point_crs is not None):
                gdf.crs = point_crs

            points = gdf.geometry.apply(lambda geom: [geom.x, geom.y])
            coordinates_array = np.array([[point] for point in points])
            points = common.coords_to_xy(self.source, coordinates_array, point_crs)
            num_points = points.shape[0]
            if point_labels_batch is None:
                labels = np.array([[1] for i in range(num_points)])
            else:
                labels = point_labels_batch

        elif isinstance(point_coords_batch, list):
            if point_crs is not None:
                point_coords_batch_crs = common.coords_to_xy(
                    self.source, point_coords_batch, point_crs
                )
            else:
                point_coords_batch_crs = point_coords_batch
            num_points = len(point_coords_batch)

            points = []
            points.append([[point] for point in point_coords_batch_crs])

            if point_labels_batch is None:
                labels = np.array([[1] for i in range(num_points)])
            elif isinstance(point_labels_batch, list):
                labels = []
                labels.append([[label] for label in point_labels_batch])
                labels = labels[0]
            else:
                labels = point_labels_batch

            points = np.array(points[0])
            labels = np.array(labels)

        elif isinstance(point_coords_batch, np.ndarray):
            points = point_coords_batch
            labels = point_labels_batch
        else:
            raise ValueError("point_coords must be a list, a GeoDataFrame, or a path.")

        predictor = self.predictor

        masks_batch, scores_batch, logits_batch = predictor.predict_batch(
            point_coords_batch=[points],
            point_labels_batch=[labels],
            box_batch=box_batch,
            mask_input_batch=mask_input_batch,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )

        masks = masks_batch[0]
        scores = scores_batch[0]
        logits = logits_batch[0]

        if multimask_output and (index is not None):
            masks = masks[:, index, :, :]

        if masks.ndim > 3:
            masks = masks.squeeze()

        output_masks = []
        sums = np.sum(masks, axis=(1, 2))
        for index, mask in enumerate(masks):
            item = {"segmentation": mask.astype("bool"), "area": sums[index]}
            output_masks.append(item)

        self.masks = output_masks
        self.scores = scores
        self.logits = logits

        if output is not None:
            self.save_masks(
                output,
                foreground=True,
                unique=unique,
                mask_multiplier=mask_multiplier,
                dtype=dtype,
                **kwargs,
            )

        if return_results:
            return output_masks, scores, logits

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Predict masks for a batch of images.

        Args:
            point_coords_batch (Optional[List[np.ndarray]]): A batch of point
                coordinates. Defaults to None.
            point_labels_batch (Optional[List[np.ndarray]]): A batch of point
                labels. Defaults to None.
            box_batch (Optional[List[np.ndarray]]): A batch of bounding boxes.
                Defaults to None.
            mask_input_batch (Optional[List[np.ndarray]]): A batch of mask inputs.
                Defaults to None.
            multimask_output (bool): Whether to output multimask at each point
                of the grid. Defaults to False.
            return_logits (bool): Whether to return the logits. Defaults to False.
            normalize_coords (bool): Whether to normalize the coordinates.
                Defaults to True.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: Lists
                of masks, multimasks, and logits.
        """

        return self.predictor.predict_batch(
            point_coords_batch=point_coords_batch,
            point_labels_batch=point_labels_batch,
            box_batch=box_batch,
            mask_input_batch=mask_input_batch,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )

    @torch.inference_mode()
    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Any:
        """Initialize an inference state.

        Args:
            video_path (str): The path to the video file.
            offload_video_to_cpu (bool): Whether to offload the video to CPU.
                Defaults to False.
            offload_state_to_cpu (bool): Whether to offload the state to CPU.
                Defaults to False.
            async_loading_frames (bool): Whether to load frames asynchronously.
                Defaults to False.

        Returns:
            Any: The initialized inference state.
        """
        return self.predictor.init_state(
            video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )

    @torch.inference_mode()
    def reset_state(self, inference_state: Any) -> None:
        """Remove all input points or masks in all frames throughout the video.

        Args:
            inference_state (Any): The current inference state.
        """
        self.predictor.reset_state(inference_state)

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state: Any,
        frame_idx: int,
        obj_id: int,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: Optional[np.ndarray] = None,
    ) -> Any:
        """Add new points or a box to the inference state.

        Args:
            inference_state (Any): The current inference state.
            frame_idx (int): The frame index.
            obj_id (int): The object ID.
            points (Optional[np.ndarray]): The points to add. Defaults to None.
            labels (Optional[np.ndarray]): The labels for the points. Defaults to None.
            clear_old_points (bool): Whether to clear old points. Defaults to True.
            normalize_coords (bool): Whether to normalize the coordinates. Defaults to True.
            box (Optional[np.ndarray]): The bounding box to add. Defaults to None.

        Returns:
            Any: The updated inference state.
        """
        return self.predictor.add_new_points_or_box(
            inference_state,
            frame_idx,
            obj_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points,
            normalize_coords=normalize_coords,
            box=box,
        )

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state: Any,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> Any:
        """Add a new mask to the inference state.

        Args:
            inference_state (Any): The current inference state.
            frame_idx (int): The frame index.
            obj_id (int): The object ID.
            mask (np.ndarray): The mask to add.

        Returns:
            Any: The updated inference state.
        """
        return self.predictor.add_new_mask(inference_state, frame_idx, obj_id, mask)

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state: Any) -> Any:
        """Propagate the inference state in video preflight.

        Args:
            inference_state (Any): The current inference state.

        Returns:
            Any: The propagated inference state.
        """
        return self.predictor.propagate_in_video_preflight(inference_state)

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
    ) -> Any:
        """Propagate the inference state in video.

        Args:
            inference_state (Any): The current inference state.
            start_frame_idx (Optional[int]): The starting frame index. Defaults to None.
            max_frame_num_to_track (Optional[int]): The maximum number of frames
                to track. Defaults to None.
            reverse (bool): Whether to propagate in reverse. Defaults to False.

        Returns:
            Any: The propagated inference state.
        """
        return self.predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

    def tensor_to_numpy(
        self,
        index: Optional[int] = None,
        output: Optional[str] = None,
        mask_multiplier: int = 255,
        dtype: str = "uint8",
        save_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """Convert the predicted masks from tensors to numpy arrays.

        Args:
            index (Optional[int], optional): The index of the mask to save.
                Defaults to None, which will save the mask with the highest score.
            output (Optional[str], optional): The path to the output image.
                Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output
                mask, which is usually a binary mask [0, 1].
            dtype (str, optional): The data type of the output image. Defaults
                to "uint8".
            save_args (Optional[Dict[str, Any]], optional): Optional arguments
                for saving the output image. Defaults to None.

        Returns:
            Optional[np.ndarray]: The predicted mask as a numpy array, or None
                if output is specified.
        """
        if save_args is None:
            save_args = {}

        boxes = self.boxes
        masks = self.masks

        image_pil = self.image
        image_np = np.array(image_pil)

        if index is None:
            index = 0

        masks = masks[:, index, :, :]
        if len(masks.shape) == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        if boxes is None or (len(boxes) == 0):  # No "object" instances found
            print("No objects found in the image.")
            return
        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (_, mask) in enumerate(zip(boxes, masks)):
                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()
                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255]
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255]

        if output is not None:
            common.array_to_image(
                mask_overlay, output, self.source, dtype=dtype, **save_args
            )
        else:
            return mask_overlay

    def save_prediction(
        self,
        output: str,
        index: Optional[int] = None,
        mask_multiplier: int = 255,
        dtype: str = "float32",
        vector: Optional[str] = None,
        simplify_tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Save the predicted mask to the output path.

        Args:
            output (str): The path to the output image.
            index (Optional[int], optional): The index of the mask to save.
                Defaults to None, which will save the mask with the highest score.
            mask_multiplier (int, optional): The mask multiplier for the output
                mask, which is usually a binary mask [0, 1].
            dtype (str, optional): The data type of the output image. Defaults
                to "float32".
            vector (Optional[str], optional): The path to the output vector file.
                Defaults to None.
            simplify_tolerance (Optional[float], optional): The maximum allowed
                geometry displacement. The higher this value, the smaller the
                number of vertices in the resulting geometry.
            **kwargs (Any): Additional keyword arguments.
        """
        if self.scores is None:
            raise ValueError("No predictions found. Please run predict() first.")

        if index is None:
            index = self.scores.argmax(axis=0)

        array = self.masks[index] * mask_multiplier
        self.prediction = array
        common.array_to_image(array, output, self.source, dtype=dtype, **kwargs)

        if vector is not None:
            common.raster_to_vector(
                output, vector, simplify_tolerance=simplify_tolerance
            )

    def show_map(
        self,
        basemap: str = "SATELLITE",
        repeat_mode: bool = True,
        out_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Show the interactive map.

        Args:
            basemap (str, optional): The basemap. It can be one of the following:
                SATELLITE, ROADMAP, TERRAIN, HYBRID.
            repeat_mode (bool, optional): Whether to use the repeat mode for
                draw control. Defaults to True.
            out_dir (Optional[str], optional): The path to the output directory.
                Defaults to None.

        Returns:
            Any: The map object.
        """
        return common.sam_map_gui(
            self, basemap=basemap, repeat_mode=repeat_mode, out_dir=out_dir, **kwargs
        )

    def show_canvas(
        self,
        fg_color: Tuple[int, int, int] = (0, 255, 0),
        bg_color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 5,
    ) -> Tuple[list, list]:
        """Show a canvas to collect foreground and background points.

        Args:
            fg_color (Tuple[int, int, int], optional): The color for the foreground points.
                Defaults to (0, 255, 0).
            bg_color (Tuple[int, int, int], optional): The color for the background points.
                Defaults to (0, 0, 255).
            radius (int, optional): The radius of the points. Defaults to 5.

        Returns:
            Tuple[list, list]: A tuple of two lists of foreground and background points.
        """

        if self.image is None:
            raise ValueError("Please run set_image() first.")

        image = self.image
        fg_points, bg_points = common.show_canvas(image, fg_color, bg_color, radius)
        self.fg_points = fg_points
        self.bg_points = bg_points
        point_coords = fg_points + bg_points
        point_labels = [1] * len(fg_points) + [0] * len(bg_points)
        self.point_coords = point_coords
        self.point_labels = point_labels

    def _convert_prompts(self, prompts: Dict[int, Any]) -> Dict[int, Any]:
        """Convert the points and labels in the prompts to numpy arrays with specific data types.

        Args:
            prompts (Dict[str, Any]): A dictionary containing the prompts with points and labels.

        Returns:
            Dict[str, Any]: The updated dictionary with points and labels converted to numpy arrays.
        """
        for _, value in prompts.items():
            # Convert points to np.float32 array
            if "points" in value:
                value["points"] = np.array(value["points"], dtype=np.float32)
            # Convert labels to np.int32 array
            if "labels" in value:
                value["labels"] = np.array(value["labels"], dtype=np.int32)
            # Convert box to np.float32 array
            if "box" in value:
                value["box"] = np.array(value["box"], dtype=np.float32)

        return prompts

    def set_video(
        self,
        video_path: str,
        output_dir: str = None,
        frame_rate: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Set the video path and parameters.

        Args:
            video_path (str): The path to the video file.
            start_frame (int, optional): The starting frame index. Defaults to 0.
            end_frame (Optional[int], optional): The ending frame index. Defaults to None.
            step (int, optional): The step size. Defaults to 1.
            frame_rate (Optional[int], optional): The frame rate. Defaults to None.
        """

        if isinstance(video_path, str):
            if video_path.startswith("http"):
                video_path = common.download_file(video_path)
            if os.path.isfile(video_path):

                if output_dir is None:
                    output_dir = common.make_temp_dir()
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                print(f"Output directory: {output_dir}")
                common.video_to_images(
                    video_path, output_dir, frame_rate=frame_rate, prefix=prefix
                )

            elif os.path.isdir(video_path):
                files = sorted(os.listdir(video_path))
                if len(files) == 0:
                    raise ValueError(f"No files found in {video_path}.")
                elif files[0].endswith(".tif"):
                    self._tif_source = os.path.join(video_path, files[0])
                    self._tif_dir = video_path
                    self._tif_names = files
                    video_path = common.geotiff_to_jpg_batch(video_path)
                output_dir = video_path

            if not os.path.exists(video_path):
                raise ValueError(f"Input path {video_path} does not exist.")
        else:
            raise ValueError("Input video_path must be a string.")

        self.video_path = output_dir
        self._num_images = len(os.listdir(output_dir))
        self._frame_names = sorted(os.listdir(output_dir))
        self.inference_state = self.predictor.init_state(video_path=output_dir)

    def predict_video(
        self,
        prompts: Dict[int, Any] = None,
        point_crs: Optional[str] = None,
        output_dir: Optional[str] = None,
        img_ext: str = "png",
    ) -> None:
        """Predict masks for the video.

        Args:
            prompts (Dict[int, Any]): A dictionary containing the prompts with points and labels.
            point_crs (Optional[str]): The coordinate reference system (CRS) of the point prompts.
            output_dir (Optional[str]): The directory to save the output images. Defaults to None.
            img_ext (str): The file extension for the output images. Defaults to "png".
        """

        from PIL import Image

        def save_image_from_dict(data, output_path="output_image.png"):
            # Find the shape of the first array in the dictionary (assuming all arrays have the same shape)
            array_shape = next(iter(data.values())).shape[1:]

            # Initialize an empty array with the same shape as the arrays in the dictionary, filled with zeros
            output_array = np.zeros(array_shape, dtype=np.uint8)

            # Iterate over each key and array in the dictionary
            for key, array in data.items():
                # Assign the key value wherever the boolean array is True
                output_array[array[0]] = key

            # Convert the output array to a PIL image
            image = Image.fromarray(output_array)

            # Save the image
            image.save(output_path)

        if prompts is None:
            if hasattr(self, "prompts"):
                prompts = self.prompts
            else:
                raise ValueError("Please provide prompts.")

        if point_crs is not None and self._tif_source is not None:
            for prompt in prompts.values():
                points = prompt.get("points", None)
                if points is not None:
                    points = common.coords_to_xy(self._tif_source, points, point_crs)
                    prompt["points"] = points
                box = prompt.get("box", None)
                if box is not None:
                    box = common.bbox_to_xy(self._tif_source, box, point_crs)
                    prompt["box"] = box

        prompts = self._convert_prompts(prompts)
        predictor = self.predictor
        inference_state = self.inference_state
        for obj_id, prompt in prompts.items():

            points = prompt.get("points", None)
            labels = prompt.get("labels", None)
            box = prompt.get("box", None)
            frame_idx = prompt.get("frame_idx", None)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box,
            )

        video_segments = {}
        num_frames = self._num_images
        num_digits = len(str(num_frames))

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            if output_dir is not None:
                output_path = os.path.join(
                    output_dir, f"{str(out_frame_idx).zfill(num_digits)}.{img_ext}"
                )
                save_image_from_dict(video_segments[out_frame_idx], output_path)

        self.video_segments = video_segments

        # if output_dir is not None:
        #     self.save_video_segments(output_dir, img_ext)

    def save_video_segments(self, output_dir: str, img_ext: str = "png") -> None:
        """Save the video segments to the output directory.

        Args:
            output_dir (str): The path to the output directory.
            img_ext (str): The file extension for the output images. Defaults to "png".
        """
        from PIL import Image

        def save_image_from_dict(
            data, output_path="output_image.png", crs_source=None, **kwargs
        ):
            # Find the shape of the first array in the dictionary (assuming all arrays have the same shape)
            array_shape = next(iter(data.values())).shape[1:]

            # Initialize an empty array with the same shape as the arrays in the dictionary, filled with zeros
            output_array = np.zeros(array_shape, dtype=np.uint8)

            # Iterate over each key and array in the dictionary
            for key, array in data.items():
                # Assign the key value wherever the boolean array is True
                output_array[array[0]] = key

            if crs_source is None:
                # Convert the output array to a PIL image
                image = Image.fromarray(output_array)

                # Save the image
                image.save(output_path)
            else:
                output_path = output_path.replace(".png", ".tif")
                common.array_to_image(output_array, output_path, crs_source, **kwargs)

        num_frames = len(self.video_segments)
        num_digits = len(str(num_frames))

        if hasattr(self, "_tif_source") and self._tif_source.endswith(".tif"):
            crs_source = self._tif_source
            filenames = self._tif_names
        else:
            crs_source = None
            filenames = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize the tqdm progress bar
        for frame_idx, video_segment in tqdm(
            self.video_segments.items(), desc="Rendering frames", total=num_frames
        ):
            if filenames is None:
                output_path = os.path.join(
                    output_dir, f"{str(frame_idx).zfill(num_digits)}.{img_ext}"
                )
            else:
                output_path = os.path.join(output_dir, filenames[frame_idx])
            save_image_from_dict(video_segment, output_path, crs_source)

    def save_video_segments_blended(
        self,
        output_dir: str,
        img_ext: str = "png",
        alpha: float = 0.6,
        dpi: int = 200,
        frame_stride: int = 1,
        output_video: Optional[str] = None,
        fps: int = 30,
    ) -> None:
        """Save blended video segments to the output directory and optionally create a video.

        Args:
            output_dir (str): The directory to save the output images.
            img_ext (str): The file extension for the output images. Defaults to "png".
            alpha (float): The alpha value for the blended masks. Defaults to 0.6.

            dpi (int): The DPI (dots per inch) for the output images. Defaults to 200.
            frame_stride (int): The stride for selecting frames to save. Defaults to 1.
            output_video (Optional[str]): The path to the output video file. Defaults to None.
            fps (int): The frames per second for the output video. Defaults to 30.
        """

        from PIL import Image

        def show_mask(mask, ax, obj_id=None, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                color = np.array([*cmap(cmap_idx)[:3], alpha])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.close("all")

        video_segments = self.video_segments
        video_dir = self.video_path
        frame_names = self._frame_names
        num_frames = len(frame_names)
        num_digits = len(str(num_frames))

        # Initialize the tqdm progress bar
        for out_frame_idx in tqdm(
            range(0, len(frame_names), frame_stride), desc="Rendering frames"
        ):
            image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))

            # Get original image dimensions
            w, h = image.size

            # Set DPI and calculate figure size based on the original image dimensions
            figsize = (
                w / dpi,
                h / dpi,
            )
            figsize = (
                figsize[0] * 1.3,
                figsize[1] * 1.3,
            )

            # Create a figure with the exact size and DPI
            fig = plt.figure(figsize=figsize, dpi=dpi)

            # Disable axis to prevent whitespace
            plt.axis("off")

            # Display the original image
            plt.imshow(image)

            # Overlay masks for each object ID
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

            # Save the figure with no borders or extra padding
            filename = f"{str(out_frame_idx).zfill(num_digits)}.{img_ext}"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=dpi, pad_inches=0, bbox_inches="tight")
            plt.close(fig)

        if output_video is not None:
            common.images_to_video(output_dir, output_video, fps=fps)

    def show_images(self, path: str = None) -> None:
        """Show the images in the video.

        Args:
            path (str, optional): The path to the images. Defaults to None.
        """
        if path is None:
            path = self.video_path

        if path is not None:
            common.show_image_gui(path)

    def show_prompts(
        self,
        prompts: Dict[int, Any],
        frame_idx: int = 0,
        mask: Any = None,
        random_color: bool = False,
        point_crs: Optional[str] = None,
        figsize: Tuple[int, int] = (9, 6),
    ) -> None:
        """Show the prompts on the image.

        Args:
            prompts (Dict[int, Any]): A dictionary containing the prompts with
                points and labels.
            frame_idx (int, optional): The frame index. Defaults to 0.
            mask (Any, optional): The mask. Defaults to None.
            random_color (bool, optional): Whether to use random colors for the
                masks. Defaults to False.
            point_crs (Optional[str], optional): The coordinate reference system
            figsize (Tuple[int, int], optional): The figure size. Defaults to (9, 6).

        """

        from PIL import Image

        def show_mask(mask, ax, obj_id=None, random_color=random_color):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                color = np.array([*cmap(cmap_idx)[:3], 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        def show_points(coords, labels, ax, marker_size=200):
            pos_points = coords[labels == 1]
            neg_points = coords[labels == 0]
            ax.scatter(
                pos_points[:, 0],
                pos_points[:, 1],
                color="green",
                marker="*",
                s=marker_size,
                edgecolor="white",
                linewidth=1.25,
            )
            ax.scatter(
                neg_points[:, 0],
                neg_points[:, 1],
                color="red",
                marker="*",
                s=marker_size,
                edgecolor="white",
                linewidth=1.25,
            )

        def show_box(box, ax):
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
                )
            )

        if point_crs is not None and self._tif_source is not None:
            for prompt in prompts.values():
                points = prompt.get("points", None)
                if points is not None:
                    points = common.coords_to_xy(self._tif_source, points, point_crs)
                    prompt["points"] = points
                box = prompt.get("box", None)
                if box is not None:
                    box = common.bbox_to_xy(self._tif_source, box, point_crs)
                    prompt["box"] = box

        prompts = self._convert_prompts(prompts)
        self.prompts = prompts
        video_dir = self.video_path
        frame_names = self._frame_names
        fig = plt.figure(figsize=figsize)
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = True
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

        for obj_id, prompt in prompts.items():
            points = prompt.get("points", None)
            labels = prompt.get("labels", None)
            box = prompt.get("box", None)
            anno_frame_idx = prompt.get("frame_idx", None)
            if anno_frame_idx == frame_idx:
                if points is not None:
                    show_points(points, labels, plt.gca())
                if box is not None:
                    show_box(box, plt.gca())
                if mask is not None:
                    show_mask(mask, plt.gca(), obj_id=obj_id)

        plt.show()

    def raster_to_vector(self, raster, vector, simplify_tolerance=None, **kwargs):
        """Convert a raster image file to a vector dataset.

        Args:
            raster (str): The path to the raster image.
            output (str): The path to the vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        common.raster_to_vector(
            raster, vector, simplify_tolerance=simplify_tolerance, **kwargs
        )

    def region_groups(
        self,
        image: Union[str, "xr.DataArray", np.ndarray],
        connectivity: int = 1,
        min_size: int = 10,
        max_size: Optional[int] = None,
        threshold: Optional[int] = None,
        properties: Optional[List[str]] = None,
        intensity_image: Optional[Union[str, "xr.DataArray", np.ndarray]] = None,
        out_csv: Optional[str] = None,
        out_vector: Optional[str] = None,
        out_image: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[
        Tuple[np.ndarray, "pd.DataFrame"], Tuple["xr.DataArray", "pd.DataFrame"]
    ]:
        """
        Segment regions in an image and filter them based on size.

        Args:
            image (Union[str, xr.DataArray, np.ndarray]): Input image, can be a file
                path, xarray DataArray, or numpy array.
            connectivity (int, optional): Connectivity for labeling. Defaults to 1
                for 4-connectivity. Use 2 for 8-connectivity.
            min_size (int, optional): Minimum size of regions to keep. Defaults to 10.
            max_size (Optional[int], optional): Maximum size of regions to keep.
                Defaults to None.
            threshold (Optional[int], optional): Threshold for filling holes.
                Defaults to None, which is equal to min_size.
            properties (Optional[List[str]], optional): List of properties to measure.
                See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
                Defaults to None.
            intensity_image (Optional[Union[str, xr.DataArray, np.ndarray]], optional):
                Intensity image to use for properties. Defaults to None.
            out_csv (Optional[str], optional): Path to save the properties as a CSV file.
                Defaults to None.
            out_vector (Optional[str], optional): Path to save the vector file.
                Defaults to None.
            out_image (Optional[str], optional): Path to save the output image.
                Defaults to None.

        Returns:
            Union[Tuple[np.ndarray, pd.DataFrame], Tuple[xr.DataArray, pd.DataFrame]]: Labeled image and properties DataFrame.
        """
        return common.region_groups(
            image,
            connectivity=connectivity,
            min_size=min_size,
            max_size=max_size,
            threshold=threshold,
            properties=properties,
            intensity_image=intensity_image,
            out_csv=out_csv,
            out_vector=out_vector,
            out_image=out_image,
            **kwargs,
        )
