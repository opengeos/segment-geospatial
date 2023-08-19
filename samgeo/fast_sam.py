"""Segmenting remote sensing images with the Fast Segment Anything Model (FastSAM.
https://github.com/opengeos/FastSAM
"""

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
            "FastSAM-x.pt": "https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing",
            "FastSAM-s.pt": "https://drive.google.com/file/d/10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV/view?usp=sharing",
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

    def set_image(self, image, device=None):
        """Set the input image.

        Args:
            image (str): The path to the image file or a HTTP URL.
            device (str, optional): The device to use. Defaults to "cuda" if available, otherwise "cpu".
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

        everything_results = self(image)

        self.prompt_process = FastSAMPrompt(image, everything_results, device=device)

    def everything_prompt(self, output=None):
        prompt_process = self.prompt_process
        ann = prompt_process.everything_prompt()
        prompt_process.plot(annotations=ann, output_path=output)
