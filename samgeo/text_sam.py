import os
import argparse
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import groundingdino.datasets.transforms as T
from PIL import Image
from rasterio.plot import show
from matplotlib.patches import Rectangle
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model

def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image, None)
    return image_transformed

# Class definition for LangSAM
class LangSAM():
    def __init__(self, sam_type="vit_h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_sam(sam_type)

    def build_sam(self, sam_type):
        checkpoint_url = SAM_MODELS[sam_type]
        sam = sam_model_registry[sam_type]()
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
        sam.load_state_dict(state_dict, strict=True)
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, self.device)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil, text_prompt, box_threshold, text_threshold):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits

def main():
    parser = argparse.ArgumentParser(description='LangSAM')
    parser.add_argument('--image', required=True, help='path to the image')
    parser.add_argument('--prompt', required=True, help='text prompt')
    parser.add_argument('--box_threshold', default=0.5, type=float, help='box threshold')
    parser.add_argument('--text_threshold', default=0.5, type=float, help='text threshold')
    args = parser.parse_args()

    with rasterio.open(args.image) as src:
        image_np = src.read().transpose((1, 2, 0))  # Convert rasterio image to numpy array
        transform = src.transform  # Save georeferencing information
        crs = src.crs  # Save the Coordinate Reference System

    model = LangSAM()
    
    image_pil = Image.fromarray(image_np[:, :, :3]) # Convert numpy array to PIL image, excluding the alpha channel
    image_np_copy = image_np.copy()  # Create a copy for modifications

    masks, boxes, phrases, logits = model.predict(image_pil, args.prompt, args.box_threshold, args.text_threshold)

    if boxes.nelement() == 0:  # No "object" instances found
        print('No objects found in the image.')
    else:
        # Create an empty image to store the mask overlays
        mask_overlay = np.zeros_like(image_np[..., 0], dtype=np.int64)  # Adjusted for single channel

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()  # Convert the tensor to a numpy array
            mask = masks[i].cpu().numpy()  # Convert the tensor to a numpy array

            # Add the mask to the mask_overlay image
            mask_overlay += ((mask > 0) * (i + 1))  # Assign a unique value for each mask

    # Normalize mask_overlay to be in [0, 255]
    mask_overlay = ((mask_overlay > 0) * 255).astype(rasterio.uint8)  # Binary mask in [0, 255]

    with rasterio.open(
        'mask.tif',
        'w',
        driver='GTiff',
        height=mask_overlay.shape[0],
        width=mask_overlay.shape[1],
        count=1,
        dtype=mask_overlay.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask_overlay, 1)

if __name__ == '__main__':
    main()