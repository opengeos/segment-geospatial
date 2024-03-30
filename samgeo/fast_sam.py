"""Segmenting remote sensing images with the Fast Segment Anything Model (FastSAM.
https://github.com/opengeos/FastSAM
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from .common import *

try:
    from fastsam import FastSAM, FastSAMPrompt
except ImportError:
    print("FastSAM not installed. Installing...")
    install_package("segment-anything-fast")
    from fastsam import FastSAM, FastSAMPrompt


class SamGeo(FastSAM):
    """Segmenting remote sensing images with the Fast Segment Anything Model (FastSAM)."""

    def __init__(self, model="FastSAM-x.pt", **kwargs):
        """Initialize the FastSAM algorithm."""

        if "checkpoint_dir" in kwargs:
            checkpoint_dir = kwargs["checkpoint_dir"]
            kwargs.pop("checkpoint_dir")
        else:
            checkpoint_dir = os.environ.get(
                "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
            )

        models = {
            "FastSAM-x.pt": "https://github.com/opengeos/datasets/releases/download/models/FastSAM-x.pt",
            "FastSAM-s.pt": "https://github.com/opengeos/datasets/releases/download/models/FastSAM-s.pt",
        }

        if model not in models:
            raise ValueError(
                f"Model must be one of {list(models.keys())}, but got {model} instead."
            )

        model_path = os.path.join(checkpoint_dir, model)

        if not os.path.exists(model_path):
            print(f"Downloading {model} to {model_path}...")
            download_file(models[model], model_path)

        super().__init__(model, **kwargs)

    def set_image(self, image, device=None, **kwargs):
        """Set the input image.

        Args:
            image (str): The path to the image file or a HTTP URL.
            device (str, optional): The device to use. Defaults to "cuda" if available, otherwise "cpu".
            kwargs: Additional keyword arguments to pass to the FastSAM model.
        """

        if isinstance(image, str):
            if image.startswith("http"):
                image = download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image
        else:
            self.source = None

        # Use cuda if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.empty_cache()

        everything_results = self(image, device=device, **kwargs)

        self.prompt_process = FastSAMPrompt(image, everything_results, device=device)

    def everything_prompt(self, output=None, **kwargs):
        """Segment the image with the everything prompt. Adapted from
        https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py#L451

        Args:
            output (str, optional): The path to save the output image. Defaults to None.
        """

        prompt_process = self.prompt_process
        ann = prompt_process.everything_prompt()
        self.annotations = ann

        if output is not None:
            self.save_masks(output, **kwargs)
        else:
            return ann

    def point_prompt(self, points, pointlabel, output=None, **kwargs):
        """Segment the image with the point prompt. Adapted from
        https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py#L414

        Args:
            points (list): A list of points.
            pointlabel (list): A list of labels for each point.
            output (str, optional): The path to save the output image. Defaults to None.
        """

        prompt_process = self.prompt_process
        ann = prompt_process.point_prompt(points, pointlabel)
        self.annotations = ann

        if output is not None:
            self.save_masks(output, **kwargs)
        else:
            return ann

    def box_prompt(self, bbox=None, bboxes=None, output=None, **kwargs):
        """Segment the image with the box prompt. Adapted from
        https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py#L377

        Args:
            bbox (list, optional): The bounding box. Defaults to None.
            bboxes (list, optional): A list of bounding boxes. Defaults to None.
            output (str, optional): The path to save the output image. Defaults to None.
        """

        prompt_process = self.prompt_process
        ann = prompt_process.box_prompt(bbox, bboxes)
        self.annotations = ann

        if output is not None:
            self.save_masks(output, **kwargs)
        else:
            return ann

    def text_prompt(self, text, output=None, **kwargs):
        """Segment the image with the text prompt. Adapted from
        https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py#L439

        Args:
            text (str): The text to segment.
            output (str, optional): The path to save the output image. Defaults to None.
        """

        prompt_process = self.prompt_process
        ann = prompt_process.text_prompt(text)
        self.annotations = ann

        if output is not None:
            self.save_masks(output, **kwargs)
        else:
            return ann

    def save_masks(
        self,
        output=None,
        better_quality=True,
        dtype=None,
        mask_multiplier=255,
        **kwargs,
    ) -> np.ndarray:
        """Save the mask of the image. Adapted from
        https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py#L222

        Returns:
            np.ndarray: The mask of the image.
        """
        annotations = self.annotations
        if isinstance(annotations[0], dict):
            annotations = [annotation["segmentation"] for annotation in annotations]
        image = self.prompt_process.img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height = image.shape[0]
        width = image.shape[1]

        if better_quality:
            if isinstance(annotations[0], torch.Tensor):
                annotations = np.array(annotations.cpu())
            for i, mask in enumerate(annotations):
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                annotations[i] = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
                )
        if self.device == "cpu":
            annotations = np.array(annotations)

        else:
            if isinstance(annotations[0], np.ndarray):
                annotations = torch.from_numpy(annotations)

        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()

        if dtype is None:
            # Set output image data type based on the number of objects
            if len(annotations) < 255:
                dtype = np.uint8
            elif len(annotations) < 65535:
                dtype = np.uint16
            else:
                dtype = np.uint32

        masks = np.sum(annotations, axis=0)

        masks = cv2.resize(masks, (width, height), interpolation=cv2.INTER_NEAREST)
        masks[masks > 0] = 1
        masks = masks.astype(dtype) * mask_multiplier
        self.objects = masks

        if output is not None:  # Save the output image
            array_to_image(self.objects, output, self.source, **kwargs)
        else:
            return masks

    def fast_show_mask(
        self,
        random_color=False,
    ):
        """Show the mask of the image. Adapted from
        https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py#L222

        Args:
            random_color (bool, optional): Whether to use random colors for each object. Defaults to False.

        Returns:
            np.ndarray: The mask of the image.
        """

        target_height = self.image.shape[0]
        target_width = self.image.shape[1]
        annotations = self.annotations
        annotation = np.array(annotations.cpu())

        mask_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        # Sort annotations based on area.
        areas = np.sum(annotation, axis=(1, 2))
        sorted_indices = np.argsort(areas)
        annotation = annotation[sorted_indices]

        index = (annotation != 0).argmax(axis=0)
        if random_color:
            color = np.random.random((mask_sum, 1, 1, 3))
        else:
            color = np.ones((mask_sum, 1, 1, 3)) * np.array(
                [30 / 255, 144 / 255, 255 / 255]
            )
        transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
        visual = np.concatenate([color, transparency], axis=-1)
        mask_image = np.expand_dims(annotation, -1) * visual

        show = np.zeros((height, weight, 4))
        h_indices, w_indices = np.meshgrid(
            np.arange(height), np.arange(weight), indexing="ij"
        )
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        # Use vectorized indexing to update the values of 'show'.
        show[h_indices, w_indices, :] = mask_image[indices]

        show = cv2.resize(
            show, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )

        return show

    def raster_to_vector(
        self, image, output, simplify_tolerance=None, dst_crs="EPSG:4326", **kwargs
    ):
        """Save the result to a vector file.

        Args:
            image (str): The path to the image file.
            output (str): The path to the vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_vector(
            image,
            output,
            simplify_tolerance=simplify_tolerance,
            dst_crs=dst_crs,
            **kwargs,
        )

    def show_anns(
        self,
        output=None,
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

        annotations = self.annotations
        prompt_process = self.prompt_process

        if output is None:
            output = temp_file_path(".png")

        prompt_process.plot(annotations, output, **kwargs)

        show_image(output)
