""" Functions related to dealing with the ACRIMA dataset. """

import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torch


def get_data(datadir, train_ratio=0.8):
    """Put the path to each image in the directory together with its label in a list."""
    data = []

    for file in sorted(os.listdir(datadir)):  # Make sure files are always processed in same order.
        path = os.path.join(datadir, file)

        if os.path.isdir(path):
            continue

        # Images with label g have '_g_' in filename.
        # Images with label ng have '_' in filename.
        if len(file) == 18:
            label = 1  # Glaucomatous image
        else:
            label = 0  # Normal image

        data.append((path, label))

    train, val = train_test_split(data, train_size=train_ratio, random_state=29)  # Use same split across runs.

    return train, val


def get_mean_and_std(dataset):
    """Calculate the mean and standard deviation of the images in the dataset per channel.
    Used for normalizing the dataset."""
    channels_sum, channels_squared_sum, num_points = 0, 0, 0

    for data in dataset:
        # Take the mean over the height and width of the image,
        # but not over the channels.
        channels_sum += torch.mean(data["image"], dim=[1, 2])
        channels_squared_sum += torch.mean(data["image"] ** 2, dim=[1, 2])

        num_points += 1

    mean = channels_sum / num_points
    std = (channels_squared_sum / num_points - mean ** 2) ** 0.5

    return mean, std


class ACRIMA(Dataset):
    """Dataset for the ACRIMA dataset."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index][0]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.data[index][1]

        return {"image": image, "label": label, "names": self.data[index][0]}
