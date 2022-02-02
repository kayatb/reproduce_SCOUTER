"""
Calculate the average IAUC and DAUC over all images in the dataset. 

----------------------------------------------------------------------------------------

Code was partially taken and adapted from the following paper:

Petsiuk, V., Das, A., & Saenko, K. (2018). 
Rise: Randomized input sampling for explanation of black-box models. 
arXiv preprint arXiv:1806.07421.

Code available at: https://github.com/eclique/RISE
Commit: d91ea00 on Sep 17, 2018
"""

import numpy as np
import torch
import torch.nn as nn

import metrics.RISE.evaluation as evaluation


def calc_iauc_and_dauc_batch(model, img_dataloader, exp_dataloader, img_size, device):
    """Calculate the IAUC and DAUC over batches in the dataloader."""
    model.eval()
    kern = evaluation.gkern(11, 5).to(device)
    blur = lambda x: nn.functional.conv2d(x.to(device), kern, padding=11 // 2)

    insertion = evaluation.CausalMetric(model, "ins", img_size, blur)
    deletion = evaluation.CausalMetric(model, "del", img_size, torch.zeros_like)

    ins_score = []
    del_score = []

    exp_iter = iter(exp_dataloader)

    with torch.no_grad():
        for data in img_dataloader:
            imgs = data["image"]
            exps = next(exp_iter)

            ins = insertion.evaluate(imgs, exps, len(imgs))
            ins_score.append(evaluation.auc(ins.mean(1)))

            dels = deletion.evaluate(imgs, exps, len(imgs))
            del_score.append(evaluation.auc(dels.mean(1)))

    return np.mean(ins_score), np.mean(del_score)
