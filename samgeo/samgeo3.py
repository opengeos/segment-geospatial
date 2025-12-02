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
    print(f"To use SamGeo 3, install it as:\n\tpip install segment-geospatial[samgeo3]")

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

        # Batch processing attributes
        self.images_batch = None
        self.sources_batch = None
        self.batch_state = None
        self.batch_results = None

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

    def set_image_batch(
        self,
        images: List[Union[str, np.ndarray]],
        state: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Set multiple images for batch processing.

        Note: This method is only available for the Meta backend.

        Args:
            images (List[Union[str, np.ndarray, Image]]): A list of input images.
                Each image can be a file path, a numpy array, or a PIL Image.
            state (dict, optional): An optional state object to pass to the
                processor's set_image_batch method.

        Returns:
            Dict[str, Any]: The batch state containing backbone outputs and
                image dimensions.

        Example:
            >>> sam = SamGeo3(backend="meta")
            >>> sam.set_image_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
            >>> results = sam.generate_masks_batch("tree")
        """
        if self.backend != "meta":
            raise NotImplementedError(
                "Batch image processing is only available for the Meta backend. "
                "Use set_image() for the Transformers backend."
            )

        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("images must be a non-empty list")

        # Process each image to PIL format
        pil_images = []
        sources = []
        numpy_images = []

        for image in images:
            if isinstance(image, str):
                if image.startswith("http"):
                    image = common.download_file(image)

                if not os.path.exists(image):
                    raise ValueError(f"Input path {image} does not exist.")

                sources.append(image)
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                numpy_images.append(img)
                pil_images.append(Image.fromarray(img))

            elif isinstance(image, np.ndarray):
                sources.append(None)
                numpy_images.append(image)
                pil_images.append(Image.fromarray(image))

            elif isinstance(image, Image.Image):
                sources.append(None)
                numpy_images.append(np.array(image))
                pil_images.append(image)

            else:
                raise ValueError(
                    "Each image must be either a path, numpy array, or PIL Image."
                )

        # Store batch information
        self.images_batch = numpy_images
        self.sources_batch = sources

        # Call the processor's set_image_batch method
        self.batch_state = self.processor.set_image_batch(pil_images, state=state)

        print(f"Set {len(pil_images)} images for batch processing.")

    def generate_masks_batch(
        self,
        prompt: str,
        min_size: int = 0,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate masks for all images in the batch using SAM3.

        Note: This method is only available for the Meta backend.

        Args:
            prompt (str): The text prompt describing the objects to segment.
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out. Defaults to 0.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out. Defaults to None (no maximum).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, one per image, containing:
                - "masks": List of mask arrays
                - "boxes": List of bounding boxes
                - "scores": List of confidence scores
                - "image": The original numpy image
                - "source": The source file path (if available)

        Example:
            >>> sam = SamGeo3(backend="meta")
            >>> sam.set_image_batch(["image1.jpg", "image2.jpg"])
            >>> results = sam.generate_masks_batch("building")
            >>> for i, result in enumerate(results):
            ...     print(f"Image {i}: Found {len(result['masks'])} objects")
        """
        if self.backend != "meta":
            raise NotImplementedError(
                "Batch mask generation is only available for the Meta backend."
            )

        if self.batch_state is None:
            raise ValueError(
                "No images set for batch processing. "
                "Please call set_image_batch() first."
            )

        batch_results = []
        num_images = len(self.images_batch)

        # The batch backbone features are computed once, but text prompting
        # needs to be done per-image since set_text_prompt expects singular
        # original_height/original_width keys
        backbone_out = self.batch_state.get("backbone_out", {})

        for i in range(num_images):
            # Create a per-image state with the correct singular keys
            image_state = {
                "original_height": self.batch_state["original_heights"][i],
                "original_width": self.batch_state["original_widths"][i],
            }

            # Extract backbone features for this specific image
            # The backbone_out contains batched features, we need to slice them
            image_backbone_out = self._extract_image_backbone_features(backbone_out, i)
            image_state["backbone_out"] = image_backbone_out

            # Reset prompts and set text prompt for this image
            self.processor.reset_all_prompts(image_state)
            output = self.processor.set_text_prompt(state=image_state, prompt=prompt)

            # Build result for this image
            result = {
                "masks": output.get("masks", []),
                "boxes": output.get("boxes", []),
                "scores": output.get("scores", []),
                "image": self.images_batch[i],
                "source": self.sources_batch[i],
            }

            # Convert tensors to numpy
            result = self._convert_batch_result_to_numpy(result)

            # Filter by size if needed
            if min_size > 0 or max_size is not None:
                result = self._filter_batch_result_by_size(result, min_size, max_size)

            batch_results.append(result)

        self.batch_results = batch_results

        # Print summary
        total_objects = sum(len(r.get("masks", [])) for r in batch_results)
        print(
            f"Processed {num_images} image(s), found {total_objects} total object(s)."
        )

    def _extract_image_backbone_features(
        self, backbone_out: Dict[str, Any], image_index: int
    ) -> Dict[str, Any]:
        """Extract backbone features for a single image from batched features.

        Args:
            backbone_out: Batched backbone output from set_image_batch.
            image_index: Index of the image to extract features for.

        Returns:
            Dictionary containing backbone features for a single image.
        """
        import torch

        image_backbone = {}

        for key, value in backbone_out.items():
            # Skip None values
            if value is None:
                image_backbone[key] = None
                continue

            if key == "sam2_backbone_out":
                # Handle nested sam2 backbone output
                if not isinstance(value, dict):
                    image_backbone[key] = value
                    continue

                sam2_out = {}
                for sam2_key, sam2_value in value.items():
                    if sam2_value is None:
                        sam2_out[sam2_key] = None
                    elif sam2_key == "backbone_fpn":
                        # backbone_fpn is a list of feature tensors
                        sam2_out[sam2_key] = [
                            (
                                feat[image_index : image_index + 1]
                                if isinstance(feat, torch.Tensor)
                                else feat
                            )
                            for feat in sam2_value
                        ]
                    elif isinstance(sam2_value, torch.Tensor):
                        sam2_out[sam2_key] = sam2_value[image_index : image_index + 1]
                    elif isinstance(sam2_value, list):
                        sam2_out[sam2_key] = [
                            (
                                v[image_index : image_index + 1]
                                if isinstance(v, torch.Tensor)
                                else v
                            )
                            for v in sam2_value
                        ]
                    else:
                        sam2_out[sam2_key] = sam2_value
                image_backbone[key] = sam2_out
            elif isinstance(value, torch.Tensor):
                # Slice the batch dimension
                image_backbone[key] = value[image_index : image_index + 1]
            elif isinstance(value, list):
                # Handle list of tensors
                image_backbone[key] = [
                    (
                        v[image_index : image_index + 1]
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for v in value
                ]
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                nested = {}
                for k, v in value.items():
                    if v is None:
                        nested[k] = None
                    elif isinstance(v, torch.Tensor):
                        nested[k] = v[image_index : image_index + 1]
                    elif isinstance(v, list):
                        nested[k] = [
                            (
                                item[image_index : image_index + 1]
                                if isinstance(item, torch.Tensor)
                                else item
                            )
                            for item in v
                        ]
                    else:
                        nested[k] = v
                image_backbone[key] = nested
            else:
                image_backbone[key] = value

        return image_backbone

    def _convert_batch_result_to_numpy(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert masks, boxes, and scores in a batch result to numpy arrays.

        Args:
            result: Dictionary containing masks, boxes, scores for one image.

        Returns:
            Dictionary with numpy arrays instead of tensors.
        """
        import torch

        # Helper to check if a value is non-empty
        def has_items(val):
            if val is None:
                return False
            if isinstance(val, (list, tuple)):
                return len(val) > 0
            if isinstance(val, torch.Tensor):
                return val.numel() > 0
            if isinstance(val, np.ndarray):
                return val.size > 0
            return bool(val)

        # Convert masks
        masks = result.get("masks")
        if has_items(masks):
            converted_masks = []
            # Handle case where masks is a single tensor with batch dimension
            if isinstance(masks, torch.Tensor):
                # If it's a batched tensor, split into list
                if masks.dim() >= 3:
                    for i in range(masks.shape[0]):
                        converted_masks.append(masks[i].cpu().numpy())
                else:
                    converted_masks.append(masks.cpu().numpy())
            else:
                # It's already a list
                for mask in masks:
                    if hasattr(mask, "cpu"):
                        mask_np = mask.cpu().numpy()
                    elif hasattr(mask, "numpy"):
                        mask_np = mask.numpy()
                    else:
                        mask_np = np.asarray(mask)
                    converted_masks.append(mask_np)
            result["masks"] = converted_masks

        # Convert boxes
        boxes = result.get("boxes")
        if has_items(boxes):
            converted_boxes = []
            if isinstance(boxes, torch.Tensor):
                # If it's a batched tensor [N, 4], split into list
                if boxes.dim() == 2:
                    for i in range(boxes.shape[0]):
                        converted_boxes.append(boxes[i].cpu().numpy())
                else:
                    converted_boxes.append(boxes.cpu().numpy())
            else:
                for box in boxes:
                    if hasattr(box, "cpu"):
                        box_np = box.cpu().numpy()
                    elif hasattr(box, "numpy"):
                        box_np = box.numpy()
                    else:
                        box_np = np.asarray(box)
                    converted_boxes.append(box_np)
            result["boxes"] = converted_boxes

        # Convert scores
        scores = result.get("scores")
        if has_items(scores):
            converted_scores = []
            if isinstance(scores, torch.Tensor):
                # If it's a 1D tensor of scores
                scores_np = scores.cpu().numpy()
                if scores_np.ndim == 0:
                    converted_scores.append(float(scores_np))
                else:
                    for s in scores_np:
                        converted_scores.append(float(s))
            else:
                for score in scores:
                    if hasattr(score, "cpu"):
                        score_val = (
                            score.cpu().item()
                            if hasattr(score, "numel") and score.numel() == 1
                            else score.cpu().numpy()
                        )
                    elif hasattr(score, "item"):
                        score_val = score.item()
                    elif hasattr(score, "numpy"):
                        score_val = score.numpy()
                    else:
                        score_val = float(score)
                    converted_scores.append(score_val)
            result["scores"] = converted_scores

        return result

    def _filter_batch_result_by_size(
        self,
        result: Dict[str, Any],
        min_size: int = 0,
        max_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Filter masks in a batch result by size.

        Args:
            result: Dictionary containing masks, boxes, scores for one image.
            min_size: Minimum mask size in pixels.
            max_size: Maximum mask size in pixels.

        Returns:
            Filtered result dictionary.
        """
        if not result.get("masks"):
            return result

        filtered_masks = []
        filtered_boxes = []
        filtered_scores = []

        masks = result["masks"]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])

        for i, mask in enumerate(masks):
            # Convert mask to numpy if needed
            if hasattr(mask, "cpu"):
                mask_np = mask.squeeze().cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = np.squeeze(mask) if hasattr(mask, "squeeze") else mask

            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np[0]

            # Calculate mask size
            mask_bool = mask_np > 0
            mask_size = np.sum(mask_bool)

            # Filter by size
            if mask_size < min_size:
                continue
            if max_size is not None and mask_size > max_size:
                continue

            # Keep this mask
            filtered_masks.append(masks[i])
            if i < len(boxes):
                filtered_boxes.append(boxes[i])
            if i < len(scores):
                filtered_scores.append(scores[i])

        result["masks"] = filtered_masks
        result["boxes"] = filtered_boxes
        result["scores"] = filtered_scores

        return result

    def save_masks_batch(
        self,
        output_dir: str,
        prefix: str = "mask",
        unique: bool = True,
        min_size: int = 0,
        max_size: Optional[int] = None,
        dtype: str = "uint8",
        **kwargs: Any,
    ) -> List[str]:
        """Save masks from batch processing to files.

        Args:
            output_dir (str): Directory to save the mask files.
            prefix (str): Prefix for output filenames. Files will be named
                "{prefix}_{index}.tif" or "{prefix}_{index}.png".
            unique (bool): If True, each mask gets a unique value (1, 2, 3, ...).
                If False, all masks are combined into a binary mask.
            min_size (int): Minimum mask size in pixels.
            max_size (int, optional): Maximum mask size in pixels.
            dtype (str): Data type for the output array.
            **kwargs: Additional arguments passed to common.array_to_image().

        Returns:
            List[str]: List of paths to saved mask files.

        Example:
            >>> sam = SamGeo3(backend="meta")
            >>> sam.set_image_batch(["img1.tif", "img2.tif"])
            >>> sam.generate_masks_batch("building")
            >>> saved_files = sam.save_masks_batch("output/", prefix="building_mask")
        """
        if self.batch_results is None:
            raise ValueError(
                "No batch results found. Please run generate_masks_batch() first."
            )

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        for i, result in enumerate(self.batch_results):
            masks = result.get("masks", [])
            source = result.get("source")
            image = result.get("image")

            if not masks:
                print(f"No masks for image {i + 1}, skipping.")
                continue

            # Get image dimensions
            if image is not None:
                height, width = image.shape[:2]
            else:
                # Try to get from first mask
                mask = masks[0]
                if hasattr(mask, "shape"):
                    if mask.ndim > 2:
                        height, width = mask.shape[-2:]
                    else:
                        height, width = mask.shape
                else:
                    raise ValueError(f"Cannot determine dimensions for image {i}")

            # Create combined mask array
            mask_array = np.zeros(
                (height, width), dtype=np.uint32 if unique else np.uint8
            )

            valid_mask_count = 0
            for j, mask in enumerate(masks):
                # Convert to numpy
                if hasattr(mask, "cpu"):
                    mask_np = mask.squeeze().cpu().numpy()
                elif hasattr(mask, "numpy"):
                    mask_np = mask.squeeze().numpy()
                else:
                    mask_np = np.squeeze(mask) if hasattr(mask, "squeeze") else mask

                if mask_np.ndim > 2:
                    mask_np = mask_np[0]

                mask_bool = mask_np > 0
                mask_size = np.sum(mask_bool)

                if mask_size < min_size:
                    continue
                if max_size is not None and mask_size > max_size:
                    continue

                if unique:
                    mask_array[mask_bool] = valid_mask_count + 1
                else:
                    mask_array[mask_bool] = 255

                valid_mask_count += 1

            if valid_mask_count == 0:
                print(f"No valid masks for image {i} after filtering.")
                continue

            # Convert dtype
            mask_array = mask_array.astype(dtype)

            # Determine output path and extension
            if source is not None and source.lower().endswith((".tif", ".tiff")):
                ext = ".tif"
            else:
                ext = ".png"

            output_path = os.path.join(output_dir, f"{prefix}_{i + 1}{ext}")

            # Save
            common.array_to_image(
                mask_array, output_path, source, dtype=dtype, **kwargs
            )
            saved_files.append(output_path)
            print(
                f"Saved {valid_mask_count} mask(s) for image {i + 1} to {output_path}"
            )

        return saved_files

    def show_anns_batch(
        self,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        show_bbox: bool = True,
        show_score: bool = True,
        output_dir: Optional[str] = None,
        prefix: str = "anns",
        blend: bool = True,
        alpha: float = 0.5,
        ncols: int = 2,
        **kwargs: Any,
    ) -> None:
        """Show annotations for all images in the batch.

        Args:
            figsize (tuple): Figure size for each subplot.
            axis (str): Whether to show axis.
            show_bbox (bool): Whether to show bounding boxes.
            show_score (bool): Whether to show confidence scores.
            output_dir (str, optional): Directory to save annotation images.
                If None, displays the figure.
            prefix (str): Prefix for output filenames.
            blend (bool): Whether to show image as background.
            alpha (float): Alpha value for mask overlay.
            ncols (int): Number of columns in the grid display.
            **kwargs: Additional arguments for saving.
        """
        if self.batch_results is None:
            raise ValueError(
                "No batch results found. Please run generate_masks_batch() first."
            )

        num_images = len(self.batch_results)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            # Save each image separately
            for i, result in enumerate(self.batch_results):
                self._show_single_ann(
                    result,
                    figsize=figsize,
                    axis=axis,
                    show_bbox=show_bbox,
                    show_score=show_score,
                    blend=blend,
                    alpha=alpha,
                    output=os.path.join(output_dir, f"{prefix}_{i + 1}.png"),
                    **kwargs,
                )
        else:
            # Display in grid
            nrows = (num_images + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows)
            )

            if num_images == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for i, result in enumerate(self.batch_results):
                ax = axes[i]
                self._show_single_ann(
                    result,
                    figsize=figsize,
                    axis=axis,
                    show_bbox=show_bbox,
                    show_score=show_score,
                    blend=blend,
                    alpha=alpha,
                    ax=ax,
                )
                ax.set_title(f"Image {i + 1}")

            # Hide unused subplots
            for j in range(num_images, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

    def _show_single_ann(
        self,
        result: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        show_bbox: bool = True,
        show_score: bool = True,
        blend: bool = True,
        alpha: float = 0.5,
        output: Optional[str] = None,
        ax: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Show annotations for a single batch result.

        Args:
            result: Dictionary containing masks, boxes, scores, and image.
            figsize: Figure size.
            axis: Whether to show axis.
            show_bbox: Whether to show bounding boxes.
            show_score: Whether to show scores.
            blend: Whether to blend with original image.
            alpha: Alpha for mask overlay.
            output: Path to save the figure.
            ax: Matplotlib axis to plot on.
            **kwargs: Additional arguments for saving.
        """
        image = result.get("image")
        masks = result.get("masks", [])
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])

        if image is None or len(masks) == 0:
            return

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
            own_figure = True
        else:
            own_figure = False

        img_pil = Image.fromarray(image)

        if blend:
            ax.imshow(img_pil)
        else:
            white_background = np.ones_like(image) * 255
            ax.imshow(white_background)

        h, w = image.shape[:2]
        COLORS = generate_colors(n_colors=128, n_samples=5000)

        for i in range(len(masks)):
            color = COLORS[i % len(COLORS)]

            mask = masks[i]
            if hasattr(mask, "cpu"):
                mask = mask.squeeze().cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask = mask.squeeze().numpy()
            else:
                mask = np.squeeze(mask)

            if mask.ndim > 2:
                mask = mask[0]

            plot_mask(mask, color=color, alpha=alpha, ax=ax)

            if show_bbox and i < len(boxes):
                score = scores[i] if i < len(scores) else 0.0
                if hasattr(score, "item"):
                    prob = score.item()
                else:
                    prob = float(score)

                if show_score:
                    text = f"(id={i}, {prob=:.2f})"
                else:
                    text = f"(id={i})"

                box = boxes[i]
                if hasattr(box, "cpu"):
                    box = box.cpu().numpy()
                elif hasattr(box, "numpy"):
                    box = box.numpy()

                plot_bbox(
                    h,
                    w,
                    box,
                    text=text,
                    box_format="XYXY",
                    color=color,
                    relative_coords=False,
                    ax=ax,
                )

        ax.axis(axis)

        if output is not None and own_figure:
            save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.1, "dpi": 100}
            save_kwargs.update(kwargs)
            plt.savefig(output, **save_kwargs)
            print(f"Saved annotations to {output}")
            plt.close(fig)
        elif own_figure:
            plt.show()

    def generate_masks(
        self,
        prompt: str,
        min_size: int = 0,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate masks for the input image using SAM3.

        Args:
            prompt (str): The text prompt describing the objects to segment.
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out. Defaults to 0.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out. Defaults to None (no maximum).

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

        # Convert tensors to numpy to free GPU memory
        self._convert_results_to_numpy()

        # Filter masks by size if min_size or max_size is specified
        if min_size > 0 or max_size is not None:
            self._filter_masks_by_size(min_size, max_size)

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
        min_size: int = 0,
        max_size: Optional[int] = None,
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
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out. Defaults to 0.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out. Defaults to None (no maximum).
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

        # Convert tensors to numpy to free GPU memory
        self._convert_results_to_numpy()

        # Filter masks by size if min_size or max_size is specified
        if min_size > 0 or max_size is not None:
            self._filter_masks_by_size(min_size, max_size)

        num_objects = len(self.masks)
        if num_objects == 0:
            print("No objects found. Please check your box prompts.")
        elif num_objects == 1:
            print("Found one object.")
        else:
            print(f"Found {num_objects} objects.")

    def _convert_results_to_numpy(self) -> None:
        """Convert masks, boxes, and scores from tensors to numpy arrays.

        This frees GPU memory by moving data to CPU and converting to numpy.
        """
        if self.masks is None:
            return

        # Convert masks to numpy
        converted_masks = []
        for mask in self.masks:
            if hasattr(mask, "cpu"):
                # PyTorch tensor on GPU
                mask_np = mask.cpu().numpy()
            elif hasattr(mask, "numpy"):
                # PyTorch tensor on CPU
                mask_np = mask.numpy()
            else:
                # Already numpy or other array-like
                mask_np = np.asarray(mask)
            converted_masks.append(mask_np)
        self.masks = converted_masks

        # Convert boxes to numpy
        if self.boxes is not None:
            converted_boxes = []
            for box in self.boxes:
                if hasattr(box, "cpu"):
                    box_np = box.cpu().numpy()
                elif hasattr(box, "numpy"):
                    box_np = box.numpy()
                else:
                    box_np = np.asarray(box)
                converted_boxes.append(box_np)
            self.boxes = converted_boxes

        # Convert scores to numpy/float
        if self.scores is not None:
            converted_scores = []
            for score in self.scores:
                if hasattr(score, "cpu"):
                    score_val = (
                        score.cpu().item()
                        if score.numel() == 1
                        else score.cpu().numpy()
                    )
                elif hasattr(score, "item"):
                    score_val = score.item()
                elif hasattr(score, "numpy"):
                    score_val = score.numpy()
                else:
                    score_val = float(score)
                converted_scores.append(score_val)
            self.scores = converted_scores

    def _filter_masks_by_size(
        self, min_size: int = 0, max_size: Optional[int] = None
    ) -> None:
        """Filter masks by size.

        Args:
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out.
        """
        if self.masks is None or len(self.masks) == 0:
            return

        filtered_masks = []
        filtered_boxes = []
        filtered_scores = []

        for i, mask in enumerate(self.masks):
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

            # Convert to boolean and calculate mask size
            mask_bool = mask_np > 0
            mask_size = np.sum(mask_bool)

            # Filter by size
            if mask_size < min_size:
                continue
            if max_size is not None and mask_size > max_size:
                continue

            # Keep this mask
            filtered_masks.append(self.masks[i])
            if self.boxes is not None and len(self.boxes) > i:
                filtered_boxes.append(self.boxes[i])
            if self.scores is not None and len(self.scores) > i:
                filtered_scores.append(self.scores[i])

        # Update the stored masks, boxes, and scores
        self.masks = filtered_masks
        self.boxes = filtered_boxes if filtered_boxes else self.boxes
        self.scores = filtered_scores if filtered_scores else self.scores

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
                mask = mask.squeeze().cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask = mask.squeeze().numpy()
            else:
                # Already numpy array
                mask = np.squeeze(mask)

            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask[0]

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
        min_size=10,
        max_size=None,
        **kwargs,
    ):
        """Show the interactive map.

        Args:
            basemap (str, optional): The basemap. Valid options include "Esri.WorldImagery", "OpenStreetMap", "HYBRID", "ROADMAP", "TERRAIN", etc. See the leafmap documentation for a full list of supported basemaps.
            out_dir (str, optional): The path to the output directory. Defaults to None.
            min_size (int, optional): The minimum size of the object. Defaults to 10.
            max_size (int, optional): The maximum size of the object. Defaults to None.
        Returns:
            leafmap.Map: The map object.
        """
        return common.text_sam_gui(
            self,
            basemap=basemap,
            out_dir=out_dir,
            box_threshold=self.confidence_threshold,
            text_threshold=self.mask_threshold,
            min_size=min_size,
            max_size=max_size,
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
