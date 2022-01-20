# from __future__ import print_function
import torch
# import torch.nn.functional as F
# from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import pandas as pd
# import os, os.path

# from timm.models import create_model
# from sloter.utils.vis import apply_colormap_on_image

def calc_precision(args, model, device, image, label, fname, bboxes):
    """ Calculate the precision for the explanation of the given image. """
    model.to(device)
    model.eval()
    image = image.to(device, dtype=torch.float32)
    _ = model(torch.unsqueeze(image, dim=0))  # Obtain the attention map (saved during forward pass).

    model.train()

    # TODO: LCS if loss_status == -1
    if args.loss_status > 0:
        id = label
    else:
        id = label + 1
    
    # Resize the attention map to be the same size as the image via bilinear interpolation.
    slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png').resize(image.size, resample=Image.BILINEAR), dtype=np.uint8)
    attention_sum = slot_image.sum()  # sum of complete attention map

    max_box_sum = 0
    # Calculate the sum of attention map pixels that fall inside the bounding box.
    for box in bboxes[fname]:
        # TODO: Determine what pixels of the attention map fall into the bounding box.

        # Take their sum.
        box_sum = box_pixels.sum()
        max_box_sum = max(box_sum, max_box_sum)

    precision = max_box_sum / attention_sum
    
    return precision


# For calculating the least similar class:
# https://www.nltk.org/howto/wordnet.html
# https://stackoverflow.com/questions/8077641/how-to-get-the-wordnet-synset-given-an-offset-id