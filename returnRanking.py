#!/usr/bin/env python3


import main
import pandas as pd
import sys
from train_classifier import impute
sys.path.append('~/libfreenect2_alex/build')


def _calculate(test_vec):

    data = pd.DataFrame
    # index_ = 0
    df = pd.DataFrame

    # i = 0
    # j = 0

    even_list = []  # first frequency
    # odd_list = []  # second frequency
    for mat_type_data in test_vec:
        for index, value in enumerate(mat_type_data):
            even_list.append(value)
        print(mat_type_data)

    # material = training_set[i][j]
    # even_list.append(
    print(data)
    data = pd.DataFrame([even_list])

    # classifiers = give_classifiers()

    # IMPUTE THE DATA
    # data = impute(data, "Iterative")

    # Second classifier = Decision Tree

    return data
