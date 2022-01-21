""" The validation set images in the ImageNet dataset are provided in a single folder, but for the SCOUTER code they
need to be ordered in separate folders with the label name. We are provided with a CSV-file that maps the filenames to
their corresponding labels. """

import argparse
import os
import pandas as pd

def get_label(row):
    """ Return the label contained in the row. """
    label = row['PredictionString'].split()[0]

    return label


def sort_imgs(dir, map):
    """ Put all images in the directory into the correct sub-folder (according to its
    label as given by map). """
    for img in os.listdir(dir):
        path = os.path.join(dir, img)
        if os.path.isfile(path):  # Skip directories.
            label = map[img[:-5]]  # Get corresponding label.
            os.rename(path, os.path.join(dir, label, img))  # Move the image to the correct folder.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restructure the ImageNet validation set folder.")
    parser.add_argument('--csv', help='The directory in which the csv-file resides')
    parser.add_argument('--img', help='The directory in which the validation set images reside.')

    args = parser.parse_args()

    df = pd.read_csv(args.csv, sep=',')
    df['label'] = df.apply(lambda row: get_label(row), axis=1)
    labels = df['label'].unique().tolist()  # Each unique label should be a folder.
    mapping = dict(zip(df['ImageId'], df['label']))

    # Create all necessary folders.
    for label in labels:
        path = os.path.join(args.img, label)
        if not os.path.exists(path):  # Check if directory does not exist yet.
            os.mkdir(path)

    sort_imgs(args.img, mapping)
