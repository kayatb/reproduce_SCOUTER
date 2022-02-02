""" 
Code was taken and adapted from the following paper:

Yeh, C. K., Hsieh, C. Y., Suggala, A., Inouye, D. I., & Ravikumar, P. K. (2019). 
On the (in) fidelity and sensitivity of explanations. 
Advances in Neural Information Processing Systems, 32, 10967-10978.

Code available at: https://github.com/chihkuanyeh/saliency_evaluation
Commit: 44a66e2 on Oct 5, 2020
"""

# use the following args to obtain results on local explanations
args = {
    "seed": 0,
    "sen_r": 0.2,
    "sen_N": 50,
    "sg_r": 0.2,
    "sg_N": 500,
    "perts": ["Gaussian"],
}
