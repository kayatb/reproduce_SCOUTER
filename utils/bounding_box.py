""" All calculations related to obtaining the bounding boxes for the images. 
The bounding boxes are used in the calculation of the precision metric. """

import argparse
import numpy as np
import pandas as pd
import json
import os
from PIL import Image


def parse_bounding_box(row):
    """Parse the bounding box coordinates contained in the row. Return as a list of tuples."""
    s = row["PredictionString"].split()
    # Delete the first and every fifth element from the list (these contain the label for the bounding box).
    del s[0::5]
    s = [int(i) for i in s]
    it = iter(s)
    # Group each four elements together (i.e. one bounding box)
    return list(zip(it, it, it, it))


def get_bounding_boxes(fname):
    """Parse the bounding box coordinates for the images from the CSV file.
    An image can have multiple bounding boxes.
    Returns a dict with image file names as keys and a list of tuple bounding box coordinates as values."""
    df = pd.read_csv(fname, sep=",")
    df["bounding_box"] = df.apply(lambda row: parse_bounding_box(row), axis=1)

    bboxes = dict(zip(df["ImageId"], df["bounding_box"]))

    return bboxes


def resize_bounding_boxes(target_size, orig_bboxes, data_path, save_file=None):
    """All images are resized to a uniform square shape. The bounding boxes
    should be resized accordingly. Make a json file that contains the resized
    bounding box coordinates based on the original coordinates."""
    # Determine the resize scale in x and y direction for each image
    # Need original size and target size.
    # Simply multiply bounding box coordinates with these scales and round
    resized_bboxes = {}

    for dir in os.listdir(data_path):
        path = os.path.join(data_path, dir)
        if os.path.isdir(path):
            for img in os.listdir(path):
                img_path = os.path.join(path, img)

                img = img[:-5]  # Remove .JPEG extension
                if not os.path.isdir(img_path):
                    image = Image.open(img_path)

                    w, h = image.size
                    x_scale = target_size / w
                    y_scale = target_size / h

                    resized_bboxes[img] = []
                    for box in orig_bboxes[img]:
                        (x1, y1, x2, y2) = box

                        new_box = (
                            int(np.round(x_scale * x1)),
                            int(np.round(y_scale * y1)),
                            int(np.round(x_scale * x2)),
                            int(np.round(y_scale * y2)),
                        )

                        resized_bboxes[img].append(new_box)

    if save_file:
        with open(save_file, "w") as fp:
            json.dump(resized_bboxes, fp)

    return resized_bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resize the bounding boxes of the images")
    parser.add_argument(
        "--csv_loc", type=str, required=True, help="Location of the CSV with the bounding box coordinates"
    )
    parser.add_argument("--img_size", type=int, required=True, help="Target size for the images")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where the data resides")
    parser.add_argument("--save_path", type=str, default=None, help="Location to save the resize dictionary")

    args = parser.parse_args()

    print("Obtaining the original bounding boxes...")
    bboxes = get_bounding_boxes(args.csv_loc)

    print("Resizing the bounding boxes...")
    resize_bounding_boxes(args.img_size, bboxes, args.data_dir, args.save_path)

    if args.save_path:
        print(f"Done! Results were saved at {args.save_path}")
    else:
        print("Done!")
