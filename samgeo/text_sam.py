"""The LangSAM model for segmenting objects from satellite images using text prompts.
The source code is adapted from the https://github.com/luca-medeiros/lang-segment-anything repository.
Credits to Luca Medeiros for the original implementation.
"""

import argparse
import inspect
import os
import warnings

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from huggingface_hub import hf_hub_download
from .common import *
from .samgeo2 import SamGeo2

try:
    import rasterio
except ImportError:
    print("Installing rasterio...")
    install_package("rasterio")

warnings.filterwarnings("ignore")


try:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.inference import predict
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

except ImportError:
    print("Installing GroundingDINO...")
    install_package("groundingdino-py")
    print("Please restart the kernel and run the notebook again.")

# Mode checkpoints
SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# Cache path
CACHE_PATH = os.environ.get(
    "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
)


def load_model_hf(
    repo_id: str, filename: str, ckpt_config_filename: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Loads a model from HuggingFace Model Hub.

    Args:
        repo_id (str): Repository ID on HuggingFace Model Hub.
        filename (str): Name of the model file in the repository.
        ckpt_config_filename (str): Name of the config file for the model in the repository.
        device (str): Device to load the model onto. Default is 'cpu'.

    Returns:
        torch.nn.Module: The loaded model.
    """

    cache_config_file = hf_hub_download(
        repo_id=repo_id,
        filename=ckpt_config_filename,
        force_filename=ckpt_config_filename,
    )
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)
    cache_file = hf_hub_download(
        repo_id=repo_id, filename=filename, force_filename=filename
    )
    checkpoint = torch.load(cache_file, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def transform_image(image: Image) -> torch.Tensor:
    """
    Transforms an image using standard transformations for image-based models.

    Args:
        image (Image): The PIL Image to be transformed.

    Returns:
        torch.Tensor: The transformed image as a tensor.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image_transformed


# Class definition for LangSAM
class LangSAM:
    """
    A Language-based Segment-Anything Model (LangSAM) class which combines GroundingDINO and SAM.
    """

    def __init__(self, model_type="vit_h", checkpoint=None):
        """Initialize the LangSAM instance.

        Args:
            model_type (str, optional): The model type. It can be one of the SAM 1
                models () vit_h, vit_l, vit_b) or SAM 2 models (sam2-hiera-tiny,
                sam2-hiera-small, sam2-hiera-base-plus, sam2-hiera-large)
                Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
            checkpoint_url (str, optional): The URL to the checkpoint file. Defaults to None
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_sam(model_type, checkpoint)

        self.source = None
        self.image = None
        self.masks = None
        self.boxes = None
        self.phrases = None
        self.logits = None
        self.prediction = None

    def build_sam(self, model_type, checkpoint_url=None):
        """Build the SAM model.

        Args:
            model_type (str, optional): The model type. It can be one of the SAM 1
                models () vit_h, vit_l, vit_b) or SAM 2 models (sam2-hiera-tiny,
                sam2-hiera-small, sam2-hiera-base-plus, sam2-hiera-large)
                Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
            checkpoint_url (str, optional): The URL to the checkpoint file. Defaults to None
        """
        sam1_models = ["vit_h", "vit_l", "vit_b"]
        sam2_models = [
            "sam2-hiera-tiny",
            "sam2-hiera-small",
            "sam2-hiera-base-plus",
            "sam2-hiera-large",
        ]
        if model_type in sam1_models:
            if checkpoint_url is not None:
                sam = sam_model_registry[model_type](checkpoint=checkpoint_url)
            else:
                checkpoint_url = SAM_MODELS[model_type]
                sam = sam_model_registry[model_type]()
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                sam.load_state_dict(state_dict, strict=True)
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
            self._sam_version = 1
        elif model_type in sam2_models:
            self.sam = SamGeo2(model_id=model_type, device=self.device, automatic=False)
            self._sam_version = 2

    def build_groundingdino(self):
        """Build the GroundingDINO model."""
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(
            ckpt_repo_id, ckpt_filename, ckpt_config_filename, self.device
        )

    def predict_dino(self, image, text_prompt, box_threshold, text_threshold):
        """
        Run the GroundingDINO model prediction.

        Args:
            image (Image): Input PIL Image.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.

        Returns:
            tuple: Tuple containing boxes, logits, and phrases.
        """

        image_trans = transform_image(image)
        boxes, logits, phrases = predict(
            model=self.groundingdino,
            image=image_trans,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        W, H = image.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_sam(self, image, boxes):
        """
        Run the SAM model prediction.

        Args:
            image (Image): Input PIL Image.
            boxes (torch.Tensor): Tensor of bounding boxes.

        Returns:
            Masks tensor.
        """
        if self._sam_version == 1:
            image_array = np.asarray(image)
            self.sam.set_image(image_array)
            transformed_boxes = self.sam.transform.apply_boxes_torch(
                boxes, image_array.shape[:2]
            )
            masks, _, _ = self.sam.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.sam.device),
                multimask_output=False,
            )
            return masks.cpu()
        elif self._sam_version == 2:

            if isinstance(self.source, str):
                self.sam.set_image(self.source)
            # If no source is set provide PIL image
            if self.source is None:
                self.sam.set_image(image)
            self.sam.boxes = boxes.numpy().tolist()
            masks, _, _ = self.sam.predict(
                boxes=boxes.numpy().tolist(),
                multimask_output=False,
                return_results=True,
            )
            self.masks = masks
            return masks

    def set_image(self, image):
        """Set the input image.

        Args:
            image (str): The path to the image file or a HTTP URL.
        """

        if isinstance(image, str):
            if image.startswith("http"):
                image = download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image
        else:
            self.source = None

    def predict(
        self,
        image,
        text_prompt,
        box_threshold,
        text_threshold,
        output=None,
        mask_multiplier=255,
        dtype=np.uint8,
        save_args={},
        return_results=False,
        return_coords=False,
        detection_filter=None,
        **kwargs,
    ):
        """
        Run both GroundingDINO and SAM model prediction.

        Parameters:
            image (Image): Input PIL Image.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.
            output (str, optional): Output path for the prediction. Defaults to None.
            mask_multiplier (int, optional): Mask multiplier for the prediction. Defaults to 255.
            dtype (np.dtype, optional): Data type for the prediction. Defaults to np.uint8.
            save_args (dict, optional): Save arguments for the prediction. Defaults to {}.
            return_results (bool, optional): Whether to return the results. Defaults to False.
            detection_filter (callable, optional):
                Callable with box, mask, logit, phrase, and index args returns a boolean.
                If provided, the function will be called for each detected object.
                Defaults to None.

        Returns:
            tuple: Tuple containing masks, boxes, phrases, and logits.
        """

        if isinstance(image, str):
            if image.startswith("http"):
                image = download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image

            # Load the georeferenced image
            with rasterio.open(image) as src:
                image_np = src.read().transpose(
                    (1, 2, 0)
                )  # Convert rasterio image to numpy array
                self.transform = src.transform  # Save georeferencing information
                self.crs = src.crs  # Save the Coordinate Reference System

                if self.crs is None:
                    warnings.warn(
                        "The CRS (Coordinate Reference System) "
                        "of input image is None. "
                        "Please define a projection on the input image "
                        "before running segment-geospatial, "
                        "or manually set CRS on result object "
                        "like `sam.crs = 'EPSG:3857'`.",
                        UserWarning,
                    )

                image_pil = Image.fromarray(
                    image_np[:, :, :3]
                )  # Convert numpy array to PIL image, excluding the alpha channel
        else:
            image_pil = image
            image_np = np.array(image_pil)

        self.image = image_pil

        boxes, logits, phrases = self.predict_dino(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            # If masks have 4 dimensions and the second dimension is 1 (e.g., [boxes, 1, height, width]),
            # squeeze that dimension to reduce it to 3 dimensions ([boxes, height, width]).
            # If boxes = 1, the mask's shape will be [1, height, width] after squeezing.
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

        if boxes.nelement() == 0:  # No "object" instances found
            print("No objects found in the image.")
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Create an empty mask overlay

        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            # Validate the detection_filter argument
            if detection_filter is not None:

                if not callable(detection_filter):
                    raise ValueError("detection_filter must be callable.")

                if not len(inspect.signature(detection_filter).parameters) == 5:
                    raise ValueError(
                        "detection_filter required args: "
                        "box, mask, logit, phrase, and index."
                    )

            for i, (box, mask, logit, phrase) in enumerate(
                zip(boxes, masks, logits, phrases)
            ):

                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()

                # Apply the user-supplied filtering logic if provided
                if detection_filter is not None:
                    if not detection_filter(box, mask, logit, phrase, i):
                        continue

                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255]
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255]

        if output is not None:
            array_to_image(mask_overlay, output, self.source, dtype=dtype, **save_args)

        self.masks = masks
        self.boxes = boxes
        self.phrases = phrases
        self.logits = logits
        self.prediction = mask_overlay

        if return_results:
            return masks, boxes, phrases, logits

        if return_coords:
            boxlist = []
            for box in self.boxes:
                box = box.cpu().numpy()
                boxlist.append((box[0], box[1]))
            return boxlist

    def predict_batch(
        self,
        images,
        out_dir,
        text_prompt,
        box_threshold,
        text_threshold,
        mask_multiplier=255,
        dtype=np.uint8,
        save_args={},
        merge=True,
        verbose=True,
        **kwargs,
    ):
        """
        Run both GroundingDINO and SAM model prediction for a batch of images.

        Parameters:
            images (list): List of input PIL Images.
            out_dir (str): Output directory for the prediction.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.
            mask_multiplier (int, optional): Mask multiplier for the prediction. Defaults to 255.
            dtype (np.dtype, optional): Data type for the prediction. Defaults to np.uint8.
            save_args (dict, optional): Save arguments for the prediction. Defaults to {}.
            merge (bool, optional): Whether to merge the predictions into a single GeoTIFF file. Defaults to True.
        """

        import glob

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if isinstance(images, str):
            images = list(glob.glob(os.path.join(images, "*.tif")))
            images.sort()

        if not isinstance(images, list):
            raise ValueError("images must be a list or a directory to GeoTIFF files.")

        for i, image in enumerate(images):
            basename = os.path.splitext(os.path.basename(image))[0]
            if verbose:
                print(
                    f"Processing image {str(i+1).zfill(len(str(len(images))))} of {len(images)}: {image}..."
                )
            output = os.path.join(out_dir, f"{basename}_mask.tif")
            self.predict(
                image,
                text_prompt,
                box_threshold,
                text_threshold,
                output=output,
                mask_multiplier=mask_multiplier,
                dtype=dtype,
                save_args=save_args,
                **kwargs,
            )

        if merge:
            output = os.path.join(out_dir, "merged.tif")
            merge_rasters(out_dir, output)
            if verbose:
                print(f"Saved the merged prediction to {output}.")

    def save_boxes(self, output=None, dst_crs="EPSG:4326", **kwargs):
        """Save the bounding boxes to a vector file.

        Args:
            output (str): The path to the output vector file.
            dst_crs (str, optional): The destination CRS. Defaults to "EPSG:4326".
            **kwargs: Additional arguments for boxes_to_vector().
        """

        if self.boxes is None:
            print("Please run predict() first.")
            return
        else:
            boxes = self.boxes.tolist()
            coords = rowcol_to_xy(self.source, boxes=boxes, dst_crs=dst_crs, **kwargs)
            if output is None:
                return boxes_to_vector(coords, self.crs, dst_crs, output)
            else:
                boxes_to_vector(coords, self.crs, dst_crs, output)

    def show_anns(
        self,
        figsize=(12, 10),
        axis="off",
        cmap="viridis",
        alpha=0.4,
        add_boxes=True,
        box_color="r",
        box_linewidth=1,
        title=None,
        output=None,
        blend=True,
        **kwargs,
    ):
        """Show the annotations (objects with random color) on the input image.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 10).
            axis (str, optional): Whether to show the axis. Defaults to "off".
            cmap (str, optional): The colormap for the annotations. Defaults to "viridis".
            alpha (float, optional): The alpha value for the annotations. Defaults to 0.4.
            add_boxes (bool, optional): Whether to show the bounding boxes. Defaults to True.
            box_color (str, optional): The color for the bounding boxes. Defaults to "r".
            box_linewidth (int, optional): The line width for the bounding boxes. Defaults to 1.
            title (str, optional): The title for the image. Defaults to None.
            output (str, optional): The path to the output image. Defaults to None.
            blend (bool, optional): Whether to show the input image. Defaults to True.
            kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().
        """

        import warnings
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        warnings.filterwarnings("ignore")

        anns = self.prediction

        if anns is None:
            print("Please run predict() first.")
            return
        elif len(anns) == 0:
            print("No objects found in the image.")
            return

        plt.figure(figsize=figsize)
        plt.imshow(self.image)

        if add_boxes:
            for box in self.boxes:
                # Draw bounding box
                box = box.cpu().numpy()  # Convert the tensor to a numpy array
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=box_linewidth,
                    edgecolor=box_color,
                    facecolor="none",
                )
                plt.gca().add_patch(rect)

        if "dpi" not in kwargs:
            kwargs["dpi"] = 100

        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"

        plt.imshow(anns, cmap=cmap, alpha=alpha)

        if title is not None:
            plt.title(title)
        plt.axis(axis)

        if output is not None:
            if blend:
                plt.savefig(output, **kwargs)
            else:
                array_to_image(self.prediction, output, self.source)

    def raster_to_vector(self, image, output, simplify_tolerance=None, **kwargs):
        """Save the result to a vector file.

        Args:
            image (str): The path to the image file.
            output (str): The path to the vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry.
        """

        raster_to_vector(image, output, simplify_tolerance=simplify_tolerance, **kwargs)

    def show_map(self, basemap="SATELLITE", out_dir=None, **kwargs):
        """Show the interactive map.

        Args:
            basemap (str, optional): The basemap. It can be one of the following: SATELLITE, ROADMAP, TERRAIN, HYBRID.
            out_dir (str, optional): The path to the output directory. Defaults to None.

        Returns:
            leafmap.Map: The map object.
        """
        return text_sam_gui(self, basemap=basemap, out_dir=out_dir, **kwargs)

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
        return self.sam.region_groups(
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


def main():
    parser = argparse.ArgumentParser(description="LangSAM")
    parser.add_argument("--image", required=True, help="path to the image")
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument(
        "--box_threshold", default=0.5, type=float, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", default=0.5, type=float, help="text threshold"
    )
    args = parser.parse_args()

    with rasterio.open(args.image) as src:
        image_np = src.read().transpose(
            (1, 2, 0)
        )  # Convert rasterio image to numpy array
        transform = src.transform  # Save georeferencing information
        crs = src.crs  # Save the Coordinate Reference System

    model = LangSAM()

    image_pil = Image.fromarray(
        image_np[:, :, :3]
    )  # Convert numpy array to PIL image, excluding the alpha channel
    image_np_copy = image_np.copy()  # Create a copy for modifications

    masks, boxes, phrases, logits = model.predict(
        image_pil, args.prompt, args.box_threshold, args.text_threshold
    )

    if boxes.nelement() == 0:  # No "object" instances found
        print("No objects found in the image.")
    else:
        # Create an empty image to store the mask overlays
        mask_overlay = np.zeros_like(
            image_np[..., 0], dtype=np.int64
        )  # Adjusted for single channel

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()  # Convert the tensor to a numpy array
            mask = masks[i].cpu().numpy()  # Convert the tensor to a numpy array

            # Add the mask to the mask_overlay image
            mask_overlay += (mask > 0) * (i + 1)  # Assign a unique value for each mask

    # Normalize mask_overlay to be in [0, 255]
    mask_overlay = ((mask_overlay > 0) * 255).astype(
        rasterio.uint8
    )  # Binary mask in [0, 255]

    with rasterio.open(
        "mask.tif",
        "w",
        driver="GTiff",
        height=mask_overlay.shape[0],
        width=mask_overlay.shape[1],
        count=1,
        dtype=mask_overlay.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask_overlay, 1)


if __name__ == "__main__":
    main()
