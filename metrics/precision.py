""" Calculate the precision metric. """

import torch
from PIL import Image
import numpy as np
import json

def calc_precision(id, img_size, fname, bboxes):
    """ Calculate the precision for the explanation of the given image. """
    # model.to(device)
    # model.eval()
    # image = image.to(device, dtype=torch.float32)
    # _ = model(torch.unsqueeze(image, dim=0))  # Obtain the attention map (saved during forward pass).

    # model.train()

    # if args.loss_status > 0:
    #     id = label
    # else:
    #     with open('lcs_label_id.json') as json_file:
    #         lcs_dict = json.load(json_file)
    #     label_str = str(label)
    #     id = lcs_dict[label_str]

    # Resize the attention map to be the same size as the image via bilinear interpolation.
    slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png').resize((img_size, img_size), resample=Image.BILINEAR), dtype=np.uint8)
    attention_sum = slot_image.sum()  # sum of complete attention map

    max_box_sum = 0
    # Calculate the sum of attention map pixels that fall inside the bounding box.
    for box in bboxes[fname]:
        # Determine what pixels of the attention map fall into the bounding box.
        box_pixels = slot_image[box[1]:box[3], box[0]:box[2]]
        # Take their sum.
        box_sum = box_pixels.sum()
        max_box_sum = max(box_sum, max_box_sum)  # Take the bounding box with the highest overlap.

    precision = max_box_sum / attention_sum
    
    return precision
