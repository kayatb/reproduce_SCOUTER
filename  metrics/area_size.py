""" Code from scouter/test.py altered to calculate the average attention ratio/area size for an image. """

import torch
from PIL import Image
import numpy as np
import json


def calc_area_size(args, model, device, image, label):
    model.to(device)
    model.eval()
    image = image.to(device, dtype=torch.float32)
    _ = model(torch.unsqueeze(image, dim=0))

    model.train()

    # TODO: least similar class for negative scouter
    if args.loss_status > 0:
        id = label
    else:
        with open('lcs_label_id.json') as json_file:
            lcs_dict = json.load(json_file)
        label_str = str(label)
        id = lcs_dict[label_str]

    slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png'), dtype=np.uint8)
    slot_image_size = slot_image.shape
    attention_ratio = float(slot_image.sum()) / float(slot_image_size[0]*slot_image_size[1]*255)
    
    return attention_ratio
