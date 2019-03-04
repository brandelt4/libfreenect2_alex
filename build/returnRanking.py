#!/usr/bin/env python3


import main
import pandas as pd
import sys
from train_classifier import impute
from real_time_nn import classifiers
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

    # material = training_set[i][j]
    # even_list.append(

    data = pd.DataFrame([even_list])

    # classifiers = give_classifiers()

    # IMPUTE THE DATA
    data = impute(data, "Iterative")

    # Second classifier = Decision Tree
    ranking = classifiers[2].predict(data)
    
    return ranking
