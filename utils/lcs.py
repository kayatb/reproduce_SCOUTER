""" 
Save the most dissimlar category of each category using the Wu-Palmer Similarity. 
Used to evaluate the explanations of SCOUTER_minus.
"""

import argparse
import json
import os
import math
import nltk

nltk.download("omw-1.4")
nltk.download("wordnet")

from nltk.corpus import wordnet


def get_name(root):
    """Get the names of the categories."""
    for root, dirs, file in os.walk(root):
        return sorted(dirs)


def main(args):
    """This function calculates the least similar category for every category using the
    subset of categories used by the model."""
    all_cat = get_name(args.data_dir)
    used_cat = all_cat[: args.num_classes]

    category_dict = {}
    symmetric_category_dict = {}
    category_label_dict = {}
    word_list = []
    label_id = 0

    # Processing all used categories and their respective synsets.
    for category in used_cat:
        category_dict[category] = wordnet.synset_from_pos_and_offset(category[0], int(category[1:]))
        symmetric_category_dict[category_dict[category]] = category
        category_label_dict[category] = label_id
        word_list.append(category_dict[category])
        label_id += 1

    # lcs_cat_id_dict = {}
    lcs_label_id_dict = {}

    # Loop over every category to determine the least similar category.
    for cat in used_cat:
        word1 = category_dict[cat]

        min_score = math.inf
        for word2 in word_list:
            score = word1.wup_similarity(word2)

            if score < min_score:
                # lcs_cat_id_dict[cat] = symmetric_category_dict[word2]
                lcs_label_id_dict[category_label_dict[cat]] = category_label_dict[symmetric_category_dict[word2]]
                min_score = score

    # Save the results in a .json for ease of use in other scripts.
    # with open("lcs_cat_id.json", 'w') as fp:
    #     json.dump(lcs_cat_id_dict, fp)

    with open(args.save_path, "w") as fp:
        json.dump(lcs_label_id_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate the Least Similar Class for each class in the dataset. ")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where the data resides")
    parser.add_argument("--save_path", type=str, default=None, help="Location to save the resize dictionary")
    parser.add_argument("--num_classes", type=int, default=100, help="How many classes from the dataset to use")

    args = parser.parse_args()

    main(args)
