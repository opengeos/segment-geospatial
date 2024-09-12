import torch
import numpy as np
import sam2


class SamGeo:
    """The main class for segmenting geospatial data with the Segment Anything Model 2 (SAM2). See
    https://github.com/facebookresearch/segment-anything-2 for details.
    """

    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
