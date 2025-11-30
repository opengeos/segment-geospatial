"""Segmenting remote sensing images with the Segment Anything Model 3 (SAM3).
https://github.com/facebookresearch/sam3
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor as MetaSam3Processor

    SAM3_META_AVAILABLE = True
except ImportError:
    SAM3_META_AVAILABLE = False

try:
    from transformers import Sam3Model, Sam3Processor as TransformersSam3Processor
    import torch

    SAM3_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SAM3_TRANSFORMERS_AVAILABLE = False

try:
    from skimage.color import lab2rgb, rgb2lab
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    import matplotlib.patches as patches
except ImportError as e:
    print(
        f"Please install required dependencies as:\n\tpip install segment-geospatial[samgeo3]"
    )

from samgeo import common


class SamGeo3:
    """The main class for segmenting geospatial data with the Segment Anything Model 3 (SAM3)."""

    def __init__(
        self,
        backend="meta",
        model_id="facebook/sam3",
        bpe_path=None,
        device=None,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile_mode=False,
        resolution=1008,
        confidence_threshold=0.5,
        mask_threshold=0.5,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SamGeo3 class.

        Args:
            backend (str): Backend to use ('meta' or 'transformers'). Default is 'meta'.
            model_id (str): Model ID for Transformers backend (e.g., 'facebook/sam3').
                Only used when backend='transformers'.
            bpe_path (str, optional): Path to the BPE tokenizer vocabulary (Meta backend only).
            device (str, optional): Device to load the model on ('cuda' or 'cpu').
            eval_mode (bool, optional): Whether to set the model to evaluation mode (Meta backend only).
            checkpoint_path (str, optional): Optional path to model checkpoint (Meta backend only).
            load_from_HF (bool, optional): Whether to load the model from HuggingFace (Meta backend only).
            enable_segmentation (bool, optional): Whether to enable segmentation head (Meta backend only).
            enable_inst_interactivity (bool, optional): Whether to enable instance interactivity (SAM 1 task) (Meta backend only).
            compile_mode (bool, optional): To enable compilation, set to "default" (Meta backend only).
            resolution (int, optional): Resolution of the image (Meta backend only).
            confidence_threshold (float, optional): Confidence threshold for the model.
            mask_threshold (float, optional): Mask threshold for post-processing (Transformers backend only).
            **kwargs: Additional keyword arguments.
        """

        if backend not in ["meta", "transformers"]:
            raise ValueError(
                f"Invalid backend '{backend}'. Choose 'meta' or 'transformers'."
            )

        if backend == "meta" and not SAM3_META_AVAILABLE:
            raise ImportError(
                "Meta SAM3 is not available. Please install it as:\n\tpip install segment-geospatial[samgeo3]"
            )

        if backend == "transformers" and not SAM3_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers SAM3 is not available. Please install it as:\n\tpip install transformers torch"
            )

        if device is None:
            device = common.get_device()

        print(f"Using {device} device and {backend} backend")

        self.backend = backend
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self.model_id = model_id
        self.model_version = "sam3"

        # Initialize backend-specific components
        if backend == "meta":
            self._init_meta_backend(
                bpe_path=bpe_path,
                device=device,
                eval_mode=eval_mode,
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_HF,
                enable_segmentation=enable_segmentation,
                enable_inst_interactivity=enable_inst_interactivity,
                compile_mode=compile_mode,
                resolution=resolution,
                confidence_threshold=confidence_threshold,
            )
        else:  # transformers
            self._init_transformers_backend(
                model_id=model_id,
                device=device,
            )

        # Common attributes
        self.predictor = None
        self.masks = None
        self.boxes = None
        self.scores = None
        self.logits = None
        self.objects = None
        self.prediction = None
        self.source = None
        self.image = None
        self.image_height = None
        self.image_width = None
        self.inference_state = None

    def _init_meta_backend(
        self,
        bpe_path,
        device,
        eval_mode,
        checkpoint_path,
        load_from_HF,
        enable_segmentation,
        enable_inst_interactivity,
        compile_mode,
        resolution,
        confidence_threshold,
    ):
        """Initialize Meta SAM3 backend."""
        if bpe_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            bpe_path = os.path.abspath(
                os.path.join(current_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")
            )
            if not os.path.exists(bpe_path):
                bpe_dir = os.path.dirname(bpe_path)
                os.makedirs(bpe_dir, exist_ok=True)
                url = "https://github.com/facebookresearch/sam3/raw/refs/heads/main/assets/bpe_simple_vocab_16e6.txt.gz"
                bpe_path = common.download_file(url, bpe_path, quiet=True)

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=eval_mode,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
            enable_segmentation=enable_segmentation,
            enable_inst_interactivity=enable_inst_interactivity,
            compile=compile_mode,
        )

        # Ensure the model is on the correct device
        model = model.to(device)

        self.model = model
        self.processor = MetaSam3Processor(
            model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def _init_transformers_backend(self, model_id, device):
        """Initialize Transformers SAM3 backend."""
        self.model = Sam3Model.from_pretrained(model_id).to(device)
        self.processor = TransformersSam3Processor.from_pretrained(model_id)

    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks.
        Args:
            threshold (float): The confidence threshold.
            state (optional): An optional state object to pass to the processor's set_confidence_threshold method (Meta backend only).
        """
        self.confidence_threshold = threshold
        if self.backend == "meta":
            self.inference_state = self.processor.set_confidence_threshold(
                threshold, state
            )
        # For transformers backend, the threshold is stored and used during generate_masks

    def set_image(
        self,
        image: Union[str, np.ndarray],
        state=None,
    ) -> None:
        """Set the input image as a numpy array.

        Args:
            image (Union[str, np.ndarray, Image]): The input image as a path,
                a numpy array, or an Image.
            state (optional): An optional state object to pass to the processor's set_image method (Meta backend only).
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
        elif isinstance(image, np.ndarray):
            self.image = image
            self.source = None
        elif isinstance(image, Image.Image):
            self.image = np.array(image)
            self.source = None
        else:
            raise ValueError(
                "Input image must be either a path, numpy array, or PIL Image."
            )

        self.image_height, self.image_width = self.image.shape[:2]

        # Convert to PIL Image for processing
        image_for_processor = Image.fromarray(self.image)

        # Set image based on backend
        if self.backend == "meta":
            # SAM3's processor expects PIL Image or tensor with (C, H, W) format
            # Numpy arrays from cv2 have (H, W, C) format which causes incorrect dimension extraction
            self.inference_state = self.processor.set_image(
                image_for_processor, state=state
            )
        else:  # transformers
            # For Transformers backend, we just store the PIL image
            # Processing will happen during generate_masks
            self.pil_image = image_for_processor

    def generate_masks(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate masks for the input image using SAM3.

        Args:
            prompt (str): The text prompt describing the objects to segment.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the generated masks.
        """
        if self.backend == "meta":
            self.processor.reset_all_prompts(self.inference_state)
            output = self.processor.set_text_prompt(
                state=self.inference_state, prompt=prompt
            )

            self.masks = output["masks"]
            self.boxes = output["boxes"]
            self.scores = output["scores"]
        else:  # transformers
            if not hasattr(self, "pil_image"):
                raise ValueError("No image set. Please call set_image() first.")

            # Prepare inputs
            inputs = self.processor(
                images=self.pil_image, text=prompt, return_tensors="pt"
            ).to(self.device)

            # Get original sizes for post-processing
            original_sizes = inputs.get("original_sizes")
            if original_sizes is not None:
                original_sizes = original_sizes.tolist()
            else:
                original_sizes = [[self.image_height, self.image_width]]

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process results
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.confidence_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=original_sizes,
            )[0]

            # Convert results to match Meta backend format
            self.masks = results["masks"]
            self.boxes = results["boxes"]
            self.scores = results["scores"]

        num_objects = len(self.masks)
        if num_objects == 0:
            print("No objects found. Please try a different prompt.")
        elif num_objects == 1:
            print("Found one object.")
        else:
            print(f"Found {num_objects} objects.")

    def generate_masks_by_boxes(
        self,
        boxes: List[List[float]],
        box_labels: Optional[List[bool]] = None,
        box_crs: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate masks using bounding box prompts.

        Args:
            boxes (List[List[float]]): List of bounding boxes in XYXY format
                [[xmin, ymin, xmax, ymax], ...].
                If box_crs is None: pixel coordinates.
                If box_crs is specified: coordinates in the given CRS (e.g., "EPSG:4326").
            box_labels (List[bool], optional): List of boolean labels for each box.
                True for positive prompt (include), False for negative prompt (exclude).
                If None, all boxes are treated as positive prompts.
            box_crs (str, optional): Coordinate reference system for box coordinates
                (e.g., "EPSG:4326" for lat/lon). Only used if the source image is a GeoTIFF.
                If None, boxes are assumed to be in pixel coordinates.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary containing masks, boxes, and scores.

        Example:
            # For pixel coordinates:
            boxes = [[100, 200, 300, 400]]
            sam.generate_masks_by_boxes(boxes)

            # For geographic coordinates (GeoTIFF):
            boxes = [[-122.5, 37.7, -122.4, 37.8]]  # [lon_min, lat_min, lon_max, lat_max]
            sam.generate_masks_by_boxes(boxes, box_crs="EPSG:4326")
        """
        if self.backend == "meta":
            if self.inference_state is None:
                raise ValueError("No image set. Please call set_image() first.")
        else:  # transformers
            if not hasattr(self, "pil_image"):
                raise ValueError("No image set. Please call set_image() first.")

        if box_labels is None:
            box_labels = [True] * len(boxes)

        if len(boxes) != len(box_labels):
            raise ValueError(
                f"Number of boxes ({len(boxes)}) must match number of labels ({len(box_labels)})"
            )

        # Transform boxes from CRS to pixel coordinates if needed
        if box_crs is not None and self.source is not None:
            pixel_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box

                # Transform min corner
                min_coords = np.array([[xmin, ymin]])
                min_xy, _ = common.coords_to_xy(
                    self.source, min_coords, box_crs, return_out_of_bounds=True
                )

                # Transform max corner
                max_coords = np.array([[xmax, ymax]])
                max_xy, _ = common.coords_to_xy(
                    self.source, max_coords, box_crs, return_out_of_bounds=True
                )

                # Convert to pixel coordinates and ensure correct min/max order
                # (geographic y increases north, pixel y increases down)
                x1_px = min_xy[0][0]
                y1_px = min_xy[0][1]
                x2_px = max_xy[0][0]
                y2_px = max_xy[0][1]

                # Ensure we have correct min/max values
                x_min_px = min(x1_px, x2_px)
                y_min_px = min(y1_px, y2_px)
                x_max_px = max(x1_px, x2_px)
                y_max_px = max(y1_px, y2_px)

                pixel_boxes.append([x_min_px, y_min_px, x_max_px, y_max_px])

            boxes = pixel_boxes

        # Get image dimensions
        width = self.image_width
        height = self.image_height

        if self.backend == "meta":
            # Reset all prompts
            self.processor.reset_all_prompts(self.inference_state)

            # Process each box
            for box, label in zip(boxes, box_labels):
                # Convert XYXY to CxCyWH format
                xmin, ymin, xmax, ymax = box
                w = xmax - xmin
                h = ymax - ymin
                cx = xmin + w / 2
                cy = ymin + h / 2

                # Normalize to [0, 1] range
                norm_box = [cx / width, cy / height, w / width, h / height]

                # Add geometric prompt
                self.inference_state = self.processor.add_geometric_prompt(
                    state=self.inference_state, box=norm_box, label=label
                )

            # Get the masks from the inference state
            output = self.inference_state

            self.masks = output["masks"]
            self.boxes = output["boxes"]
            self.scores = output["scores"]
        else:  # transformers
            # For Transformers backend, process boxes with the processor
            # Convert boxes to the format expected by Transformers
            # Transformers expects boxes in XYXY format with 3 levels of nesting:
            # [image level, box level, box coordinates]
            # Also convert numpy types to Python native types
            input_boxes = [
                [[float(coord) for coord in box] for box in boxes]
            ]  # Wrap in list for image level and convert to float

            # Prepare inputs with boxes
            inputs = self.processor(
                images=self.pil_image, input_boxes=input_boxes, return_tensors="pt"
            ).to(self.device)

            # Get original sizes for post-processing
            original_sizes = inputs.get("original_sizes")
            if original_sizes is not None:
                original_sizes = original_sizes.tolist()
            else:
                original_sizes = [[self.image_height, self.image_width]]

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process results
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.confidence_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=original_sizes,
            )[0]

            # Convert results to match Meta backend format
            self.masks = results["masks"]
            self.boxes = results["boxes"]
            self.scores = results["scores"]

        num_objects = len(self.masks)
        if num_objects == 0:
            print("No objects found. Please check your box prompts.")
        elif num_objects == 1:
            print("Found one object.")
        else:
            print(f"Found {num_objects} objects.")

    def show_boxes(
        self,
        boxes: List[List[float]],
        box_labels: Optional[List[bool]] = None,
        box_crs: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        positive_color: Tuple[int, int, int] = (0, 255, 0),
        negative_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
    ) -> None:
        """
        Visualize bounding boxes on the image.

        Args:
            boxes (List[List[float]]): List of bounding boxes in XYXY format
                [[xmin, ymin, xmax, ymax], ...].
                If box_crs is None: pixel coordinates.
                If box_crs is specified: coordinates in the given CRS.
            box_labels (List[bool], optional): List of boolean labels for each box.
                True (positive) shown in green, False (negative) shown in red.
                If None, all boxes shown in green.
            box_crs (str, optional): Coordinate reference system for box coordinates
                (e.g., "EPSG:4326"). If None, boxes are in pixel coordinates.
            figsize (Tuple[int, int]): Figure size for display.
            axis (str): Whether to show axis ("on" or "off").
            positive_color (Tuple[int, int, int]): RGB color for positive boxes.
            negative_color (Tuple[int, int, int]): RGB color for negative boxes.
            thickness (int): Line thickness for box borders.
        """
        if self.image is None:
            raise ValueError("No image set. Please call set_image() first.")

        if box_labels is None:
            box_labels = [True] * len(boxes)

        # Transform boxes from CRS to pixel coordinates if needed
        if box_crs is not None and self.source is not None:
            pixel_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box

                # Transform min corner
                min_coords = np.array([[xmin, ymin]])
                min_xy, _ = common.coords_to_xy(
                    self.source, min_coords, box_crs, return_out_of_bounds=True
                )

                # Transform max corner
                max_coords = np.array([[xmax, ymax]])
                max_xy, _ = common.coords_to_xy(
                    self.source, max_coords, box_crs, return_out_of_bounds=True
                )

                # Convert to pixel coordinates and ensure correct min/max order
                # (geographic y increases north, pixel y increases down)
                x1_px = min_xy[0][0]
                y1_px = min_xy[0][1]
                x2_px = max_xy[0][0]
                y2_px = max_xy[0][1]

                # Ensure we have correct min/max values
                x_min_px = min(x1_px, x2_px)
                y_min_px = min(y1_px, y2_px)
                x_max_px = max(x1_px, x2_px)
                y_max_px = max(y1_px, y2_px)

                pixel_boxes.append([x_min_px, y_min_px, x_max_px, y_max_px])

            boxes = pixel_boxes

        # Convert image to PIL if needed
        if isinstance(self.image, np.ndarray):
            img = Image.fromarray(self.image)
        else:
            img = self.image

        # Draw each box
        for box, label in zip(boxes, box_labels):
            # Convert XYXY to XYWH for drawing
            xmin, ymin, xmax, ymax = box
            box_xywh = [xmin, ymin, xmax - xmin, ymax - ymin]

            # Choose color based on label
            color = positive_color if label else negative_color

            # Draw box
            img = draw_box_on_image(img, box_xywh, color=color, thickness=thickness)

        # Display
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis(axis)
        plt.show()

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
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Predict the mask for the input image using interactive prompts.

        Note: This method is only available for the Meta backend with inst_interactivity=True.

        Args:
            point_coords (np.ndarray, optional): The point coordinates. Defaults to None.
            point_labels (np.ndarray, optional): The point labels. Defaults to None.
            boxes (np.ndarray, optional): Bounding box prompts in XYXY format.
            mask_input (np.ndarray, optional): Low resolution mask input.
            multimask_output (bool): Whether to output multiple masks.
            return_logits (bool): Whether to return logits instead of binary masks.
            normalize_coords (bool): Whether to normalize coordinates.
            point_crs (str, optional): Coordinate reference system for points.
            output (str, optional): Path to save output image.
            index (int, optional): Index of mask to save.
            mask_multiplier (int): Multiplier for mask values.
            dtype (str): Data type for output.
            return_results (bool): Whether to return results.
            **kwargs: Additional arguments.

        Returns:
            Tuple of masks, scores, and logits.
        """

        if self.backend == "transformers":
            raise NotImplementedError(
                "Interactive prediction is not supported for Transformers backend. "
                "Use generate_masks() or generate_masks_by_boxes() instead."
            )

        if self.predictor is None:
            raise ValueError(
                "Interactive predictor not available. Enable inst_interactivity=True."
            )

        if self.image is None:
            raise ValueError("No image set. Please call set_image() first.")

        # Handle coordinate transformations if needed
        if (
            point_crs is not None
            and point_coords is not None
            and self.source is not None
        ):
            point_coords, _ = common.coords_to_xy(
                self.source, point_coords, point_crs, return_out_of_bounds=True
            )

        if isinstance(point_coords, list):
            point_coords = np.array(point_coords)

        if point_coords is not None and point_labels is None:
            point_labels = [1] * len(point_coords)

        if isinstance(point_labels, list):
            point_labels = np.array(point_labels)

        # Use the interactive predictor
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )

        self.masks = masks
        self.scores = scores
        self.logits = logits

        if output is not None:
            self.save_prediction(output, index, mask_multiplier, dtype, **kwargs)

        if return_results:
            return masks, scores, logits

    def save_masks(
        self,
        output: Optional[str] = None,
        unique: bool = True,
        min_size: int = 0,
        max_size: Optional[int] = None,
        dtype: str = "uint8",
        save_scores: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Save the generated masks to a file or generate mask array for visualization.

        If the input image is a GeoTIFF, the output will be saved as a GeoTIFF
        with the same georeferencing information. Otherwise, it will be saved as PNG.

        Args:
            output (str, optional): The path to the output file. If None, only generates
                the mask array in memory (self.objects) without saving to disk.
            unique (bool): If True, each mask gets a unique value (1, 2, 3, ...).
                If False, all masks are combined into a binary mask (0 or 255).
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out.
            dtype (str): Data type for the output array.
            save_scores (str, optional): If provided, saves a confidence score map
                to this path. Each pixel will have the confidence score of its mask.
                The output format (GeoTIFF or PNG) follows the same logic as the mask output.
            **kwargs: Additional keyword arguments passed to common.array_to_image().
        """
        if self.masks is None or len(self.masks) == 0:
            raise ValueError("No masks found. Please run generate_masks() first.")

        if save_scores is not None and self.scores is None:
            raise ValueError("No scores found. Cannot save scores.")

        # Create empty array for combined masks
        mask_array = np.zeros(
            (self.image_height, self.image_width),
            dtype=np.uint32 if unique else np.uint8,
        )

        # Create empty array for scores if requested
        if save_scores is not None:
            scores_array = np.zeros(
                (self.image_height, self.image_width), dtype=np.float32
            )

        # Process each mask
        valid_mask_count = 0
        mask_index = 0
        for mask in self.masks:
            # Convert mask to numpy array if it's a tensor
            if hasattr(mask, "cpu"):
                mask_np = mask.squeeze().cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = mask.squeeze() if hasattr(mask, "squeeze") else mask

            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np[0]

            # Convert to boolean
            mask_bool = mask_np > 0

            # Calculate mask size
            mask_size = np.sum(mask_bool)

            # Filter by size
            if mask_size < min_size:
                mask_index += 1
                continue
            if max_size is not None and mask_size > max_size:
                mask_index += 1
                continue

            # Get confidence score for this mask
            if save_scores is not None:
                if hasattr(self.scores[mask_index], "item"):
                    score = self.scores[mask_index].item()
                else:
                    score = float(self.scores[mask_index])

            # Add mask to array
            if unique:
                # Assign unique value to each mask (starting from 1)
                mask_value = valid_mask_count + 1
                mask_array[mask_bool] = mask_value
            else:
                # Binary mask: all foreground pixels are 255
                mask_array[mask_bool] = 255

            # Add score to scores array
            if save_scores is not None:
                scores_array[mask_bool] = score

            valid_mask_count += 1
            mask_index += 1

        if valid_mask_count == 0:
            print("No masks met the size criteria.")
            return

        # Convert to requested dtype
        if dtype == "uint8":
            if unique and valid_mask_count > 255:
                print(
                    f"Warning: {valid_mask_count} masks found, but uint8 can only represent 255 unique values. Consider using dtype='uint16'."
                )
            mask_array = mask_array.astype(np.uint8)
        elif dtype == "uint16":
            mask_array = mask_array.astype(np.uint16)
        elif dtype == "int32":
            mask_array = mask_array.astype(np.int32)
        else:
            mask_array = mask_array.astype(dtype)

        # Store the mask array for visualization
        self.objects = mask_array

        # Only save to file if output path is provided
        if output is not None:
            # Save using common utility which handles GeoTIFF georeferencing
            common.array_to_image(
                mask_array, output, self.source, dtype=dtype, **kwargs
            )
            print(f"Saved {valid_mask_count} mask(s) to {output}")

            # Save scores if requested
            if save_scores is not None:
                common.array_to_image(
                    scores_array, save_scores, self.source, dtype="float32", **kwargs
                )
                print(f"Saved confidence scores to {save_scores}")

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
            index (Optional[int]): The index of the mask to save.
            mask_multiplier (int): The mask multiplier for the output mask.
            dtype (str): The data type of the output image.
            vector (Optional[str]): The path to the output vector file.
            simplify_tolerance (Optional[float]): The maximum allowed geometry displacement.
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

    def show_masks(
        self,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "tab20",
        axis: str = "off",
        unique: bool = True,
        **kwargs: Any,
    ) -> None:
        """Show the binary mask or the mask of objects with unique values.

        Args:
            figsize (tuple): The figure size.
            cmap (str): The colormap. Default is "tab20" for showing unique objects.
                Use "binary_r" for binary masks when unique=False.
                Other good options: "viridis", "nipy_spectral", "rainbow".
            axis (str): Whether to show the axis.
            unique (bool): If True, each mask gets a unique color value. If False, binary mask.
            **kwargs: Additional keyword arguments passed to save_masks() for filtering
                (e.g., min_size, max_size, dtype).
        """

        # Always regenerate mask array to ensure it matches the unique parameter
        # This prevents showing stale cached binary masks when unique=True is requested
        self.save_masks(output=None, unique=unique, **kwargs)

        if self.objects is None:
            # save_masks would have printed a message if no masks met criteria
            return

        plt.figure(figsize=figsize)
        plt.imshow(self.objects, cmap=cmap, interpolation="nearest")
        plt.axis(axis)

        plt.show()

    def show_anns(
        self,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        show_bbox: bool = True,
        show_score: bool = True,
        output: Optional[str] = None,
        blend: bool = True,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Show the annotations (objects with random color) on the input image.

        Args:
            figsize (tuple): The figure size.
            axis (str): Whether to show the axis.
            show_bbox (bool): Whether to show the bounding box.
            show_score (bool): Whether to show the score.
            output (str, optional): The path to the output image. If provided, the
                figure will be saved instead of displayed.
            blend (bool): Whether to show the input image as background. If False,
                only annotations will be shown on a white background.
            alpha (float): The alpha value for the annotations.
            **kwargs: Additional keyword arguments passed to plt.savefig() when
                output is provided (e.g., dpi, bbox_inches, pad_inches).
        """

        if self.image is None:
            print("Please run set_image() first.")
            return

        if self.masks is None or len(self.masks) == 0:
            return

        # Create results dict matching SAM3's format
        results = {
            "masks": self.masks,
            "boxes": self.boxes,
            "scores": self.scores,
        }

        # Convert numpy array to PIL Image to match SAM3's plot_results expectations
        img_pil = Image.fromarray(self.image)

        # Create figure
        fig = plt.figure(figsize=figsize)

        if blend:
            # Show image as background
            plt.imshow(img_pil)
        else:
            # Create white background with same dimensions
            white_background = np.ones_like(self.image) * 255
            plt.imshow(white_background)

        nb_objects = len(results["scores"])

        # Use original dimensions from inference_state (boxes are scaled to these)
        if (
            self.backend == "meta"
            and self.inference_state
            and "original_width" in self.inference_state
        ):
            w = self.inference_state["original_width"]
            h = self.inference_state["original_height"]
        else:
            # Fallback to image dimensions
            w, h = img_pil.size

        COLORS = generate_colors(n_colors=128, n_samples=5000)

        for i in range(nb_objects):
            color = COLORS[i % len(COLORS)]

            # Handle both tensor and numpy array formats
            mask = results["masks"][i]
            if hasattr(mask, "cpu"):
                mask = mask.squeeze(0).cpu()
            elif hasattr(mask, "squeeze"):
                mask = mask.squeeze(0)

            plot_mask(mask, color=color, alpha=alpha)

            if show_bbox:
                # Handle score extraction
                score = results["scores"][i]
                if hasattr(score, "item"):
                    prob = score.item()
                else:
                    prob = float(score)

                if show_score:
                    text = f"(id={i}, {prob=:.2f})"
                else:
                    text = f"(id={i})"

                # Handle box extraction
                box = results["boxes"][i]
                if hasattr(box, "cpu"):
                    box = box.cpu()

                plot_bbox(
                    h,
                    w,
                    box,
                    text=text,
                    box_format="XYXY",
                    color=color,
                    relative_coords=False,
                )

        plt.axis(axis)

        if output is not None:
            # Save the figure
            save_kwargs = {
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "dpi": 100,
            }
            save_kwargs.update(kwargs)
            plt.savefig(output, **save_kwargs)
            print(f"Saved annotations to {output}")
            plt.close(fig)
        else:
            # Display the figure
            plt.show()

    def raster_to_vector(
        self,
        raster: str,
        vector: str,
        simplify_tolerance: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Convert a raster image file to a vector dataset.

        Args:
            raster (str): The path to the raster image.
            vector (str): The path to the output vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
        """
        common.raster_to_vector(
            raster, vector, simplify_tolerance=simplify_tolerance, **kwargs
        )

    def show_map(
        self,
        basemap="Esri.WorldImagery",
        out_dir=None,
        **kwargs,
    ):
        """Show the interactive map.

        Args:
            basemap (str, optional): The basemap. Valid options include "Esri.WorldImagery", "OpenStreetMap", "HYBRID", "ROADMAP", "TERRAIN", etc. See the leafmap documentation for a full list of supported basemaps.
            out_dir (str, optional): The path to the output directory. Defaults to None.

        Returns:
            leafmap.Map: The map object.
        """
        return common.text_sam_gui(
            self,
            basemap=basemap,
            out_dir=out_dir,
            box_threshold=self.confidence_threshold,
            text_threshold=self.mask_threshold,
            **kwargs,
        )

    def show_canvas(
        self,
        fg_color: Tuple[int, int, int] = (0, 255, 0),
        bg_color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 5,
    ) -> Tuple[list, list]:
        """Show a canvas to collect foreground and background points.

        Args:
            fg_color (Tuple[int, int, int]): The color for foreground points.
            bg_color (Tuple[int, int, int]): The color for background points.
            radius (int): The radius of the points.

        Returns:
            Tuple of foreground and background points.
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

        return fg_points, bg_points


def generate_colors(n_colors: int = 256, n_samples: int = 5000) -> np.ndarray:
    """Generate colors for the masks.

    Args:
        n_colors (int, optional): The number of colors to generate. Defaults to 256.
        n_samples (int, optional): The number of samples to generate. Defaults to 5000.

    Returns:
        np.ndarray: The generated colors in RGB format.
    """
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    """Plot the bounding box on the image.

    Args:
        img_height (int): The height of the image.
        img_width (int): The width of the image.
        box (np.ndarray): The bounding box.
        box_format (str): The format of the bounding box.
        relative_coords (bool): Whether the coordinates are relative to the image.
        color (str): The color of the bounding box.
        linestyle (str): The line style of the bounding box.
        text (str): The text to display in the bounding box.
        ax (matplotlib.axes.Axes, optional): The axis to plot the bounding box on.
    """
    # Convert box to numpy array if it's a tensor
    if hasattr(box, "numpy"):
        box = box.numpy()
    elif hasattr(box, "cpu"):
        box = box.cpu().numpy()

    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (float(x), float(y)),
        float(w),
        float(h),
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            float(x),
            float(y) - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )


def draw_box_on_image(image, box, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on an image.

    Args:
        image (PIL.Image.Image or np.ndarray): The image to draw on.
        box (List[float]): Bounding box in XYWH format [x, y, width, height].
        color (Tuple[int, int, int]): RGB color for the box. Default is green.
        thickness (int): Line thickness in pixels.

    Returns:
        PIL.Image.Image: Image with box drawn.
    """
    from PIL import ImageDraw

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Make a copy to avoid modifying the original
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    # Extract box coordinates (XYWH format)
    x, y, w, h = box

    # Draw rectangle
    draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)

    return image_copy


def plot_mask(mask, color="r", alpha=0.5, ax=None):
    """Plot the mask on the image.

    Args:
        mask (np.ndarray): The mask to plot.
        color (str): The color of the mask.
        ax (matplotlib.axes.Axes, optional): The axis to plot the mask on.
        alpha (float): The alpha value for the mask.
    """
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * alpha
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)
