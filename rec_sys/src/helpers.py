# -*- coding: utf-8 -*-
"""some functions for help."""

import numpy as np
import scipy.sparse as sp
import csv


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def deal_line(line):
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)


def statistics(data):
    row = set([line[0] for line in data])
    col = set([line[1] for line in data])
    return min(row), max(row), min(col), max(col)


def samples_csv_submission(path_dataset):
    # read sample submission to get ids to predict
    data = read_txt(path_dataset)[1:]

    # parse each line of sample_submission
    data = [deal_line(line) for line in data]

    return data


def create_csv_submission(data, item_features, user_features, bias_item,
                          bias_user, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ###############################
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:

        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        for item, user, _ in data:
            # get a prediction for specific users and items.
            item_info = item_features[:, item - 1]
            user_info = user_features[:, user - 1]

            prediction = bias_item[item-1] + bias_user[user-1] + item_info.dot(user_info)

            writer.writerow({'Id': 'r' + str(item) + '_c' + str(user),'Prediction': prediction})
