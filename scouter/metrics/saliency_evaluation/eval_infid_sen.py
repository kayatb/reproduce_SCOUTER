""" 
Code was taken and adapted from the following paper:

Yeh, C. K., Hsieh, C. Y., Suggala, A., Inouye, D. I., & Ravikumar, P. K. (2019). 
On the (in) fidelity and sensitivity of explanations. 
Advances in Neural Information Processing Systems, 32, 10967-10978.

Code available at: https://github.com/chihkuanyeh/saliency_evaluation
Commit: 44a66e2 on Oct 5, 2020
"""

import numpy as np
import torch

from metrics.saliency_evaluation.infid_sen_utils import evaluate_infid_sen
import metrics.saliency_evaluation.config as config


class Args:
    def __init__(self, args):
        self.source = args
        for key, val in args.items():
            setattr(self, key, val)

    def __repr__(self):
        return repr(self.source)


def calc_infid_and_sens(model, dataloader, exp_path, loss_status, lsc_dict):
    """Calculate the infidelity and sensitivity scores."""
    args = Args(config.args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = model.cuda()

    infid_scores = {}
    max_sen_scores = {}
    for pert in args.perts:
        infid, max_sen = evaluate_infid_sen(
            dataloader, model, exp_path, loss_status, lsc_dict, pert, args.sen_r, args.sen_N
        )
        infid_scores[pert] = infid
        max_sen_scores[pert] = max_sen

    return infid_scores, max_sen_scores
