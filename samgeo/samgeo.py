"""
The source code is adapted from https://github.com/aliaksandr960/segment-anything-eo. Credit to the author Aliaksandr Hancharenka.
"""

import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from .common import *


class SamGeo:
    """The main class for segmenting geospatial data with the Segment Anything Model (SAM). See
    https://github.com/facebookresearch/segment-anything for details.
    """

    def __init__(
        self,
        model_type="vit_h",
        automatic=True,
        device=None,
        checkpoint_dir=None,
        sam_kwargs=None,
        **kwargs,
    ):
        """Initialize the class.

        Args:
            model_type (str, optional): The model type. It can be one of the following: vit_h, vit_l, vit_b.
                Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
            automatic (bool, optional): Whether to use the automatic mask generator or input prompts. Defaults to True.
                The automatic mask generator will segment the entire image, while the input prompts will segment selected objects.
            device (str, optional): The device to use. It can be one of the following: cpu, cuda.
                Defaults to None, which will use cuda if available.
            checkpoint_dir (str, optional): The path to the model checkpoint. It can be one of the following:
                sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth, sam_vit_b_01ec64.pth.
                Defaults to None. See https://bit.ly/3VrpxUh for more details.
            sam_kwargs (dict, optional): Optional arguments for fine-tuning the SAM model. Defaults to None.
                The available arguments with default values are listed below. See https://bit.ly/410RV0v for more details.

                points_per_side: Optional[int] = 32,
                points_per_batch: int = 64,
                pred_iou_thresh: float = 0.88,
                stability_score_thresh: float = 0.95,
                stability_score_offset: float = 1.0,
                box_nms_thresh: float = 0.7,
                crop_n_layers: int = 0,
                crop_nms_thresh: float = 0.7,
                crop_overlap_ratio: float = 512 / 1500,
                crop_n_points_downscale_factor: int = 1,
                point_grids: Optional[List[np.ndarray]] = None,
                min_mask_region_area: int = 0,
                output_mode: str = "binary_mask",

        """
        hq = False  # Not using HQ-SAM

        if "checkpoint" in kwargs:
            checkpoint = kwargs["checkpoint"]
            if not os.path.exists(checkpoint):
                checkpoint = download_checkpoint(model_type, checkpoint_dir, hq)
            kwargs.pop("checkpoint")
        else:
            checkpoint = download_checkpoint(model_type, checkpoint_dir, hq)

        # Use cuda if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.empty_cache()

        self.checkpoint = checkpoint
        self.model_type = model_type
        self.model_version = "sam"
        self.device = device
        self.sam_kwargs = sam_kwargs  # Optional arguments for fine-tuning the SAM model
        self.source = None  # Store the input image path
        self.image = None  # Store the input image as a numpy array
        # Store the masks as a list of dictionaries. Each mask is a dictionary
        # containing segmentation, area, bbox, predicted_iou, point_coords, stability_score, and crop_box
        self.masks = None
        self.objects = None  # Store the mask objects as a numpy array
        # Store the annotations (objects with random color) as a numpy array.
        self.annotations = None

        # Store the predicted masks, iou_predictions, and low_res_masks
        self.prediction = None
        self.scores = None
        self.logits = None

        # Build the SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)
        # Use optional arguments for fine-tuning the SAM model
        sam_kwargs = self.sam_kwargs if self.sam_kwargs is not None else {}

        if automatic:
            # Segment the entire image using the automatic mask generator
            self.mask_generator = SamAutomaticMaskGenerator(self.sam, **sam_kwargs)
        else:
            # Segment selected objects using input prompts
            self.predictor = SamPredictor(self.sam, **sam_kwargs)

    def __call__(
        self,
        image,
        foreground=True,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
        **kwargs,
    ):
        """Generate masks for the input tile. This function originates from the segment-anything-eo repository.
            See https://bit.ly/41pwiHw

        Args:
            image (np.ndarray): The input image as a numpy array.
            foreground (bool, optional): Whether to generate the foreground mask. Defaults to True.
            erosion_kernel (tuple, optional): The erosion kernel for filtering object masks and extract borders. Defaults to (3, 3).
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range, for example [0, 255]. Defaults to 255.
        """
        h, w, _ = image.shape

        masks = self.mask_generator.generate(image)

        if foreground:  # Extract foreground objects only
            resulting_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            resulting_mask = np.ones((h, w), dtype=np.uint8)
        resulting_borders = np.zeros((h, w), dtype=np.uint8)

        for m in masks:
            mask = (m["segmentation"] > 0).astype(np.uint8)
            resulting_mask += mask

            # Apply erosion to the mask
            if erosion_kernel is not None:
                mask_erode = cv2.erode(mask, erosion_kernel, iterations=1)
                mask_erode = (mask_erode > 0).astype(np.uint8)
                edge_mask = mask - mask_erode
                resulting_borders += edge_mask

        resulting_mask = (resulting_mask > 0).astype(np.uint8)
        resulting_borders = (resulting_borders > 0).astype(np.uint8)
        resulting_mask_with_borders = resulting_mask - resulting_borders
        return resulting_mask_with_borders * mask_multiplier

    def generate(
        self,
        source,
        output=None,
        foreground=True,
        batch=False,
        batch_sample_size=(512, 512),
        batch_nodata_threshold=1.0,
        nodata_value=None,
        erosion_kernel=None,
        mask_multiplier=255,
        unique=True,
        min_size=0,
        max_size=None,
        **kwargs,
    ):
        """Generate masks for the input image.

        Args:
            source (str | np.ndarray): The path to the input image or the input image as a numpy array.
            output (str, optional): The path to the output image. Defaults to None.
            foreground (bool, optional): Whether to generate the foreground mask. Defaults to True.
            batch (bool, optional): Whether to generate masks for a batch of image tiles. Defaults to False.
            batch_sample_size (tuple, optional): When batch=True, the size of the sample window when iterating over rasters.
            batch_nodata_threshold (float,optional): Batch samples with a fraction of nodata pixels above this threshold will
                not be used to generate a mask. The default, 1.0, will skip samples with 100% nodata values. This is useful
                when rasters have large areas of nodata values which can be skipped.
            nodata_value (int, optional): Nodata value to use in checking batch_nodata_threshold. The default, None,
                will use the nodata value in the raster metadata if present.
            erosion_kernel (tuple, optional): The erosion kernel for filtering object masks and extract borders.
                Such as (3, 3) or (5, 5). Set to None to disable it. Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range, for example [0, 255]. Defaults to 255.
                The parameter is ignored if unique is True.
            unique (bool, optional): Whether to assign a unique value to each object. Defaults to True.
                The unique value increases from 1 to the number of objects. The larger the number, the larger the object area.
            min_size (int, optional): The minimum size of the objects. Defaults to 0.
            max_size (int, optional): The maximum size of the objects. Defaults to None.
            **kwargs: Other arguments for save_masks().

        """

        if isinstance(source, str):
            if source.startswith("http"):
                source = download_file(source)

            if not os.path.exists(source):
                raise ValueError(f"Input path {source} does not exist.")

            if batch:  # Subdivide the image into tiles and segment each tile
                self.batch = True
                self.source = source
                self.masks = output
                return tiff_to_tiff(
                    source,
                    output,
                    self,
                    foreground=foreground,
                    sample_size=batch_sample_size,
                    sample_nodata_threshold=batch_nodata_threshold,
                    nodata_value=nodata_value,
                    erosion_kernel=erosion_kernel,
                    mask_multiplier=mask_multiplier,
                    **kwargs,
                )

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
        self.batch = False
        self._min_size = min_size
        self._max_size = max_size

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
        output=None,
        foreground=True,
        unique=True,
        erosion_kernel=None,
        mask_multiplier=255,
        min_size=0,
        max_size=None,
        **kwargs,
    ):
        """Save the masks to the output path. The output is either a binary mask or a mask of objects with unique values.

        Args:
            output (str, optional): The path to the output image. Defaults to None, saving the masks to SamGeo.objects.
            foreground (bool, optional): Whether to generate the foreground mask. Defaults to True.
            unique (bool, optional): Whether to assign a unique value to each object. Defaults to True.
            erosion_kernel (tuple, optional): The erosion kernel for filtering object masks and extract borders.
                Such as (3, 3) or (5, 5). Set to None to disable it. Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range, for example [0, 255]. Defaults to 255.
            min_size (int, optional): The minimum size of the objects. Defaults to 0.
            max_size (int, optional): The maximum size of the objects. Defaults to None.
            **kwargs: Other arguments for array_to_image().

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
            array_to_image(self.objects, output, self.source, **kwargs)

    def show_masks(
        self, figsize=(12, 10), cmap="binary_r", axis="off", foreground=True, **kwargs
    ):
        """Show the binary mask or the mask of objects with unique values.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            cmap (str, optional): The colormap. Defaults to "binary_r".
            axis (str, optional): Whether to show the axis. Defaults to "off".
            foreground (bool, optional): Whether to show the foreground mask only. Defaults to True.
            **kwargs: Other arguments for save_masks().
        """

        import matplotlib.pyplot as plt

        if self.batch:
            self.objects = cv2.imread(self.masks)
        else:
            if self.objects is None:
                self.save_masks(foreground=foreground, **kwargs)

        plt.figure(figsize=figsize)
        plt.imshow(self.objects, cmap=cmap)
        plt.axis(axis)
        plt.show()

    def show_anns(
        self,
        figsize=(12, 10),
        axis="off",
        alpha=0.35,
        output=None,
        blend=True,
        **kwargs,
    ):
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
                array = blend_images(
                    self.annotations, self.image, alpha=alpha, show=False
                )
            else:
                array = self.annotations
            array_to_image(array, output, self.source)

    def set_image(self, image, image_format="RGB"):
        """Set the input image as a numpy array.

        Args:
            image (np.ndarray): The input image as a numpy array.
            image_format (str, optional): The image format, can be RGB or BGR. Defaults to "RGB".
        """
        if isinstance(image, str):
            if image.startswith("http"):
                image = download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image

            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = image
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("Input image must be either a path or a numpy array.")

        self.predictor.set_image(image, image_format=image_format)

    def save_prediction(
        self,
        output,
        index=None,
        mask_multiplier=255,
        dtype=np.float32,
        vector=None,
        simplify_tolerance=None,
        **kwargs,
    ):
        """Save the predicted mask to the output path.

        Args:
            output (str): The path to the output image.
            index (int, optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
            vector (str, optional): The path to the output vector file. Defaults to None.
            dtype (np.dtype, optional): The data type of the output image. Defaults to np.float32.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.

        """
        if self.scores is None:
            raise ValueError("No predictions found. Please run predict() first.")

        if index is None:
            index = self.scores.argmax(axis=0)

        array = self.masks[index] * mask_multiplier
        self.prediction = array
        array_to_image(array, output, self.source, dtype=dtype, **kwargs)

        if vector is not None:
            raster_to_vector(output, vector, simplify_tolerance=simplify_tolerance)

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        boxes=None,
        point_crs=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        output=None,
        index=None,
        mask_multiplier=255,
        dtype="float32",
        return_results=False,
        **kwargs,
    ):
        """Predict masks for the given input prompts, using the currently set image.

        Args:
            point_coords (str | dict | list | np.ndarray, optional): A Nx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels. It can be a path to a vector file, a GeoJSON
                dictionary, a list of coordinates [lon, lat], or a numpy array. Defaults to None.
            point_labels (list | int | np.ndarray, optional): A length N array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a background point.
            point_crs (str, optional): The coordinate reference system (CRS) of the point prompts.
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
            return_logits (bool, optional): If true, returns un-thresholded masks logits
                instead of a binary mask.
            output (str, optional): The path to the output image. Defaults to None.
            index (index, optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
            dtype (np.dtype, optional): The data type of the output image. Defaults to np.float32.
            return_results (bool, optional): Whether to return the predicted masks, scores, and logits. Defaults to False.

        """
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
            point_coords = vector_to_geojson(point_coords)

        if isinstance(point_coords, dict):
            point_coords = geojson_to_coords(point_coords)

        if hasattr(self, "point_coords"):
            point_coords = self.point_coords

        if hasattr(self, "point_labels"):
            point_labels = self.point_labels

        if (point_crs is not None) and (point_coords is not None):
            point_coords, out_of_bounds = coords_to_xy(
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
            coords = bbox_to_xy(self.source, boxes, point_crs)
            input_boxes = np.array(coords)
            if isinstance(coords[0], int):
                input_boxes = input_boxes[None, :]
            else:
                input_boxes = torch.tensor(input_boxes, device=self.device)
                input_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, self.image.shape[:2]
                )
        elif isinstance(boxes, list) and (point_crs is None):
            input_boxes = np.array(boxes)
            if isinstance(boxes[0], int):
                input_boxes = input_boxes[None, :]

        self.boxes = input_boxes

        if (
            boxes is None
            or (len(boxes) == 1)
            or (len(boxes) == 4 and isinstance(boxes[0], float))
        ):
            if isinstance(boxes, list) and isinstance(boxes[0], list):
                boxes = boxes[0]
            masks, scores, logits = predictor.predict(
                point_coords,
                point_labels,
                input_boxes,
                mask_input,
                multimask_output,
                return_logits,
            )
        else:
            masks, scores, logits = predictor.predict_torch(
                point_coords=point_coords,
                point_labels=point_coords,
                boxes=input_boxes,
                multimask_output=True,
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

    def tensor_to_numpy(
        self, index=None, output=None, mask_multiplier=255, dtype="uint8", save_args={}
    ):
        """Convert the predicted masks from tensors to numpy arrays.

        Args:
            index (index, optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            output (str, optional): The path to the output image. Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
            dtype (np.dtype, optional): The data type of the output image. Defaults to np.uint8.
            save_args (dict, optional): Optional arguments for saving the output image. Defaults to {}.

        Returns:
            np.ndarray: The predicted mask as a numpy array.
        """

        boxes = self.boxes
        masks = self.masks

        image_pil = self.image
        image_np = np.array(image_pil)

        if index is None:
            index = 1

        masks = masks[:, index, :, :]
        masks = masks.squeeze(1)

        if boxes is None or (len(boxes) == 0):  # No "object" instances found
            print("No objects found in the image.")
            return
        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (box, mask) in enumerate(zip(boxes, masks)):
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
            array_to_image(mask_overlay, output, self.source, dtype=dtype, **save_args)
        else:
            return mask_overlay

    def show_map(self, basemap="SATELLITE", repeat_mode=True, out_dir=None, **kwargs):
        """Show the interactive map.

        Args:
            basemap (str, optional): The basemap. It can be one of the following: SATELLITE, ROADMAP, TERRAIN, HYBRID.
            repeat_mode (bool, optional): Whether to use the repeat mode for draw control. Defaults to True.
            out_dir (str, optional): The path to the output directory. Defaults to None.

        Returns:
            leafmap.Map: The map object.
        """
        return sam_map_gui(
            self, basemap=basemap, repeat_mode=repeat_mode, out_dir=out_dir, **kwargs
        )

    def show_canvas(self, fg_color=(0, 255, 0), bg_color=(0, 0, 255), radius=5):
        """Show a canvas to collect foreground and background points.

        Args:
            image (str | np.ndarray): The input image.
            fg_color (tuple, optional): The color for the foreground points. Defaults to (0, 255, 0).
            bg_color (tuple, optional): The color for the background points. Defaults to (0, 0, 255).
            radius (int, optional): The radius of the points. Defaults to 5.

        Returns:
            tuple: A tuple of two lists of foreground and background points.
        """

        if self.image is None:
            raise ValueError("Please run set_image() first.")

        image = self.image
        fg_points, bg_points = show_canvas(image, fg_color, bg_color, radius)
        self.fg_points = fg_points
        self.bg_points = bg_points
        point_coords = fg_points + bg_points
        point_labels = [1] * len(fg_points) + [0] * len(bg_points)
        self.point_coords = point_coords
        self.point_labels = point_labels

    def clear_cuda_cache(self):
        """Clear the CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def image_to_image(self, image, **kwargs):
        return image_to_image(image, self, **kwargs)

    def download_tms_as_tiff(self, source, pt1, pt2, zoom, dist):
        image = draw_tile(source, pt1[0], pt1[1], pt2[0], pt2[1], zoom, dist)
        return image

    def raster_to_vector(self, image, output, simplify_tolerance=None, **kwargs):
        """Save the result to a vector file.

        Args:
            image (str): The path to the image file.
            output (str): The path to the vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_vector(image, output, simplify_tolerance=simplify_tolerance, **kwargs)

    def tiff_to_vector(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a gpkg file.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_vector(
            tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs
        )

    def tiff_to_gpkg(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a gpkg file.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the gpkg file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_gpkg(
            tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs
        )

    def tiff_to_shp(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a shapefile.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the shapefile.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_shp(
            tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs
        )

    def tiff_to_geojson(self, tiff_path, output, simplify_tolerance=None, **kwargs):
        """Convert a tiff file to a GeoJSON file.

        Args:
            tiff_path (str): The path to the tiff file.
            output (str): The path to the GeoJSON file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_geojson(
            tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs
        )


class SamGeoPredictor(SamPredictor):
    def __init__(
        self,
        sam_model,
    ):
        from segment_anything.utils.transforms import ResizeLongestSide

        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    def set_image(self, image):
        super(SamGeoPredictor, self).set_image(image)

    def predict(
        self,
        src_fp=None,
        geo_box=None,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        if geo_box and src_fp:
            self.crs = "EPSG:4326"
            dst_crs = get_crs(src_fp)
            sw = transform_coords(geo_box[0], geo_box[1], self.crs, dst_crs)
            ne = transform_coords(geo_box[2], geo_box[3], self.crs, dst_crs)
            xs = np.array([sw[0], ne[0]])
            ys = np.array([sw[1], ne[1]])
            box = get_pixel_coords(src_fp, xs, ys)
            self.geo_box = geo_box
            self.width = box[2] - box[0]
            self.height = box[3] - box[1]
            self.geo_transform = set_transform(geo_box, self.width, self.height)

        masks, iou_predictions, low_res_masks = super(SamGeoPredictor, self).predict(
            point_coords, point_labels, box, mask_input, multimask_output, return_logits
        )

        return masks, iou_predictions, low_res_masks

    def masks_to_geotiff(self, src_fp, dst_fp, masks):
        profile = get_profile(src_fp)
        write_raster(
            dst_fp,
            masks,
            profile,
            self.width,
            self.height,
            self.geo_transform,
            self.crs,
        )

    def geotiff_to_geojson(self, src_fp, dst_fp, bidx=1):
        gdf = get_features(src_fp, bidx)
        write_features(gdf, dst_fp)
        return gdf
