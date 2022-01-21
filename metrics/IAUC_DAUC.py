""" Calculate the average IAUC and DAUC over all images in the dataset. 
Code partially taken from https://github.com/eclique/RISE. """


import numpy as np
# from matplotlib import pyplot as plt
# from tqdm import tqdm

import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torchvision.datasets as datasets
# import torchvision.models as models
# from torch.nn.functional import conv2d

# import json
from PIL import Image

import RISE.evaluation
import RISE.utils

def calc_iauc_and_dauc(model, image, id, img_size):
    """ Calculate the Insertion Area Under Curve and Deletion Area Under Curve 
    for the given image. """

    slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png').resize((img_size, img_size), resample=Image.BILINEAR), dtype=np.uint8)

    kern = RISE.evaluation.gkern(11, 5).cuda()
    blur = lambda x: nn.functional.conv2d(x, kern, padding=11//2)

    insertion = RISE.evaluation.CausalMetric(model, 'ins', img_size, blur)
    deletion = RISE.evaluation.CausalMetric(model, 'del', img_size, torch.zeros_like)
    
    ins_score = insertion.single_run(torch.unsqueeze(image, dim=0), slot_image)
    del_score = deletion.single_run(torch.unsqueeze(image, dim=0), slot_image)

    return RISE.evaluation.auc(ins_score), RISE.evaluation.auc(del_score)
