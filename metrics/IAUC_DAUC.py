""" Calculate the average IAUC and DAUC over all images in the dataset. 
Code partially taken from https://github.com/eclique/RISE. """


import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import RISE.evaluation
import RISE.utils


def calc_iauc_and_dauc(model, image, id, img_size):
    """Calculate the Insertion Area Under Curve and Deletion Area Under Curve
    for the given image."""

    slot_image = np.array(
        Image.open(f"sloter/vis/slot_{id}.png").resize((img_size, img_size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )

    kern = RISE.evaluation.gkern(11, 5).cuda()
    blur = lambda x: nn.functional.conv2d(x, kern, padding=11 // 2)

    insertion = RISE.evaluation.CausalMetric(model, "ins", img_size, blur)
    deletion = RISE.evaluation.CausalMetric(model, "del", img_size, torch.zeros_like)

    ins_score = insertion.single_run(torch.unsqueeze(image, dim=0), slot_image)
    del_score = deletion.single_run(torch.unsqueeze(image, dim=0), slot_image)

    return RISE.evaluation.auc(ins_score), RISE.evaluation.auc(del_score)


def calc_iauc_and_dauc_batch(model, img_dataloader, exp_dataloader, img_size):
    """Calculate the IAUC and DAUC over batches in the dataloader."""
    model.eval()
    kern = RISE.evaluation.gkern(11, 5).cuda()
    blur = lambda x: nn.functional.conv2d(x.cuda(), kern, padding=11 // 2)

    insertion = RISE.evaluation.CausalMetric(model, "ins", img_size, blur)
    deletion = RISE.evaluation.CausalMetric(model, "del", img_size, torch.zeros_like)

    ins_score = []
    del_score = []

    # images = np.empty((len(img_dataloader), batch_size, 3, img_size, img_size))
    # for i, data in enumerate(img_dataloader):
    #     images[i] = data["image"][0]
    # images.reshape((-1, 3, img_size, img_size))

    # exps = np.empty((len(exp_dataloader), batch_size, 3, img_size, img_size))
    # for i, data in enumerate(exp_dataloader):
    #     exps[i] = data
    # exps.reshape((-1, 3, img_size, img_size))

    exp_iter = iter(exp_dataloader)

    with torch.no_grad():
        for data in img_dataloader:
            imgs = data["image"]
            exps = next(exp_iter)

            ins = insertion.evaluate(imgs, exps, len(imgs))
            ins_score.append(RISE.evaluation.auc(ins.mean(1)))

            dels = deletion.evaluate(imgs, exps, len(imgs))
            del_score.append(RISE.evaluation.auc(dels.mean(1)))

    return np.mean(ins_score), np.mean(del_score)
