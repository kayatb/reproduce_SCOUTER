"""
Calculate the area size of the explanation in a given image.

----------------------------------------------------------------------------------------

Code was partially taken and adapted from the following paper:

Li, L., Wang, B., Verma, M., Nakashima, Y., Kawasaki, R., & Nagahara, H. (2021). 
SCOUTER: Slot attention-based classifier for explainable image recognition. 
In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1046-1055).

Code available at: https://github.com/wbw520/scouter
Commit: 5885b82 on Sep 7, 2021
"""

import numpy as np


def calc_area_size(exp_image):
    slot_image = np.array(exp_image, dtype=np.uint8)
    slot_image_size = slot_image.shape
    attention_ratio = float(slot_image.sum()) / float(slot_image_size[0] * slot_image_size[1] * 255)

    return attention_ratio
