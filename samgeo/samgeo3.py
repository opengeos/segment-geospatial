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
    from sam3.model.sam3_image_processor import Sam3Processor
    from skimage.color import lab2rgb, rgb2lab
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    import matplotlib.patches as patches
except ImportError as e:
    print(f"Please install sam3 as:\n\tpip install segment-geospatial[samgeo3]")

from samgeo import common


class SamGeo3:
    """The main class for segmenting geospatial data with the Segment Anything Model 3 (SAM3)."""

    def __init__(
        self,
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
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SamGeo3 class.

        Args:
            bpe_path (str, optional): Path to the BPE tokenizer vocabulary.
            device (str, optional): Device to load the model on ('cuda' or 'cpu').
            eval_mode (bool, optional): Whether to set the model to evaluation mode.
            checkpoint_path (str, optional): Optional path to model checkpoint.
            load_from_HF (bool, optional): Whether to load the model from HuggingFace.
            enable_segmentation (bool, optional): Whether to enable segmentation head.
            enable_inst_interactivity (bool, optional): Whether to enable instance interactivity (SAM 1 task).
            compile_mode (bool, optional): To enable compilation, set to "default".
            resolution (int, optional): Resolution of the image.
            confidence_threshold (float, optional): Confidence threshold for the model.
            **kwargs: Additional keyword arguments.
        """

        if device is None:
            device = common.get_device()

        print(f"Using device: {device}")
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

        self.device = device
        self.processor = Sam3Processor(
            model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )
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

    def set_image(
        self,
        image: Union[str, np.ndarray],
        state=None,
    ) -> None:
        """Set the input image as a numpy array.

        Args:
            image (Union[str, np.ndarray, Image]): The input image as a path,
                a numpy array, or an Image.
            state (optional): An optional state object to pass to the processor's set_image method.
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

        # Convert to PIL Image for the processor to ensure correct dimension extraction
        # SAM3's processor expects PIL Image or tensor with (C, H, W) format
        # Numpy arrays from cv2 have (H, W, C) format which causes incorrect dimension extraction
        image_for_processor = Image.fromarray(self.image)
        self.inference_state = self.processor.set_image(
            image_for_processor, state=state
        )

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
        self.processor.reset_all_prompts(self.inference_state)
        output = self.processor.set_text_prompt(
            state=self.inference_state, prompt=prompt
        )

        self.masks = output["masks"]
        self.boxes = output["boxes"]
        self.scores = output["scores"]

        num_objects = len(self.masks)
        if num_objects == 0:
            print("No objects found. Please try a different prompt.")
        elif num_objects == 1:
            print("Found one object.")
        else:
            print(f"Found {num_objects} objects.")

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
        foreground: bool = True,
        unique: bool = True,
        erosion_kernel: Optional[Tuple[int, int]] = None,
        mask_multiplier: int = 255,
        min_size: int = 0,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Save the masks to the output path.

        Args:
            output (str, optional): The path to the output image.
            foreground (bool): Whether to generate the foreground mask.
            unique (bool): Whether to assign a unique value to each object.
            erosion_kernel (tuple, optional): The erosion kernel for filtering.
            mask_multiplier (int): The mask multiplier for the output mask.
            min_size (int): The minimum size of the object.
            max_size (int): The maximum size of the object.
            **kwargs: Additional keyword arguments for common.array_to_image().
        """

        if self.masks is None or len(self.masks) == 0:
            # Create empty mask if no masks found
            h, w, _ = self.image.shape
            objects = np.zeros((h, w), dtype=np.uint8)
            self.objects = objects
            if output is not None:
                common.array_to_image(self.objects, output, self.source, **kwargs)
            return

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
            objects = np.zeros((h, w))
            # Assign a unique value to each object
            count = len(sorted_masks)
            for index, ann in enumerate(sorted_masks):
                m = ann["segmentation"]
                if min_size > 0 and ann["area"] < min_size:
                    continue
                if max_size is not None and ann["area"] > max_size:
                    continue
                # Ensure mask dimensions match image dimensions
                if m.shape[0] != h or m.shape[1] != w:
                    # Resize mask to match image dimensions
                    m_resized = cv2.resize(
                        m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    m = m_resized.astype(bool)
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
                # Ensure mask dimensions match image dimensions
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
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
        cmap: str = "binary_r",
        axis: str = "off",
        foreground: bool = True,
        **kwargs: Any,
    ) -> None:
        """Show the binary mask or the mask of objects with unique values.

        Args:
            figsize (tuple): The figure size.
            cmap (str): The colormap.
            axis (str): Whether to show the axis.
            foreground (bool): Whether to show the foreground mask only.
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
        if self.inference_state and "original_width" in self.inference_state:
            w = self.inference_state["original_width"]
            h = self.inference_state["original_height"]
        else:
            # Fallback to image dimensions
            w, h = img_pil.size

        COLORS = generate_colors(n_colors=128, n_samples=5000)

        for i in range(nb_objects):
            color = COLORS[i % len(COLORS)]
            plot_mask(results["masks"][i].squeeze(0).cpu(), color=color, alpha=alpha)
            if show_bbox:
                if show_score:
                    prob = results["scores"][i].item()
                    text = f"(id={i}, {prob=:.2f})"
                else:
                    text = f"(id={i})"
                plot_bbox(
                    h,
                    w,
                    results["boxes"][i].cpu(),
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
        basemap: str = "SATELLITE",
        repeat_mode: bool = True,
        out_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Show the interactive map.

        Args:
            basemap (str): The basemap type.
            repeat_mode (bool): Whether to use repeat mode for draw control.
            out_dir (Optional[str]): The path to the output directory.

        Returns:
            The map object.
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


def generate_colors(n_colors=256, n_samples=5000):
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
