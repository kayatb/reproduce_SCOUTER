""" Get the explanation images as a dataset object. """
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def get_exp_filenames(path):
    """Return a list of all explanation image filenames."""
    filenames = []

    for file in os.listdir(path):
        if os.path.isdir(file):
            continue
        filenames.append(os.path.join(path, file))

    return filenames


class ExpData(Dataset):
    """Contains the explanation images."""

    def __init__(self, filenames, img_size, resize=False):
        self.filenames = filenames
        self.resize = resize
        self.img_size = img_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = self.filenames[index]
        img = Image.open(img_path)

        if self.resize:
            img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        return np.array(img, dtype=np.uint8)
