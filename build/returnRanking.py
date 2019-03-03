import train_classifier
import pandas as pd


def calculate(test_vec):

    data = pd.DataFrame
    # index_ = 0
    df = pd.DataFrame

    i = 0
    j = 0

    even_list = []  # first frequency
    odd_list = []  # second frequency
    for mat_type_data in test_vec:
        for index, value in enumerate(mat_type_data):
            even_list.append(value)

    # material = training_set[i][j]
    # even_list.append(
    #     material[0:(len(material) - 2)])  # last column–target– we append the name removing 01,02, etc.

    data = pd.DataFrame([even_list])

    classifiers = train_classifier.main()

    # Second classifier = Decision Tree
    ranking = classifiers[2].predict(data)
    
    return ranking
