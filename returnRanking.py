#!/usr/bin/env python3


import main
import pandas as pd
import sys
sys.path.append('~/libfreenect2_alex/build')


def calculate_(test_vec):
    even_list = []  # frequency
    for mat_type_data in test_vec:
        for index, value in enumerate(mat_type_data):
            even_list.append(value)

    # material = training_set[i][j]
    # even_list.append(
    data = pd.DataFrame([even_list])

    # classifiers = give_classifiers()

    # IMPUTE THE DATA
    # data = impute(data, "Iterative")

    # Second classifier = Decision Tree

    return data
