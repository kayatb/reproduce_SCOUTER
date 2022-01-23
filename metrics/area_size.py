""" Code from scouter/test.py altered to calculate the average attention ratio/area size for an image. """
import numpy as np


def calc_area_size(exp_image):
    slot_image = np.array(exp_image, dtype=np.uint8)
    slot_image_size = slot_image.shape
    attention_ratio = float(slot_image.sum()) / float(slot_image_size[0] * slot_image_size[1] * 255)

    return attention_ratio
