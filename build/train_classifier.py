#!/usr/bin/env python


import dat2png as reader
import sys
sys.path.append('~/libfreenect2_alex/build')
import math
import os
import numpy as np
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import execnet


def cross_validation_nearest_neighbor_classifier(materials, rep=10, max_index=1, num_test=3, num_training=3,
                                                 absolute_depth=True, linear_stage=True, relative_120=True,
                                                 amplitude=False, normalize=True, ignore_80=False):
    confusion = np.zeros((len(materials), len(materials)))
    filename = 'results\confusion_'
    filename += 'alumi-base' if absolute_depth else 'material-only'
    filename += '_linear-stage' if linear_stage else '_depth-base'
    filename += '_120-base' if relative_120 else ''
    filename += '_80-ign' if ignore_80 else ''
    filename += '_with-amp' if amplitude else ''
    filename += '_normalize' if amplitude else ''
    #     for idx in range(rep):
    targ = []
    prob = []
    for m in materials:
        tm = []
        while len(tm) < num_test:
            p = m + str(random.randint(0, max_index)).zfill(2)
            if not p in tm:
                tm.append(p)
        targ.append(tm)  # TEST SET
        pm = []
        while len(pm) < num_training:
            p = m + str(random.randint(0, max_index)).zfill(2)
            if not (p in pm or p in tm):
                pm.append(p)
        prob.append(pm)  # TRAINING SET
    data, testData = nearest_neighbor_classify(targ, prob, confusion, absolute_depth=absolute_depth,
                                               linear_stage=linear_stage, relative_120=relative_120,
                                               normalize=normalize, amplitude=amplitude, ignore_80=ignore_80)

    return data, testData


def nearest_neighbor_classify(test_set, training_set, confusion, verbose=True, absolute_depth=True, linear_stage=True,
                              relative_120=True, amplitude=False, normalize=False, ignore_80=False):
    # GETTING TRAINING DATA
    training_data = []
    for mats in training_set:  # training_set = [[plastic01, ...], []]
        t_data_mat = []
        for m in mats:
            t_data_mat.append(np.vstack(
                load_data(m, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120,
                          normalize_metric=normalize, amplitude=amplitude, ignore_80=ignore_80)).T)
        training_data.append(t_data_mat)

    data = pd.DataFrame
    index_ = 0
    df = pd.DataFrame

    i = 0
    j = 0
    for i, train in enumerate(training_data):
        for j, mat_data in enumerate(train):
            even_list = []  # first frequency
            odd_list = []  # second frequency
            for mat_type_data in mat_data:
                for index, value in enumerate(mat_type_data):
                    if index % 2 == 0:
                        even_list.append(value)

                    else:
                        odd_list.append(value)

            even_list.extend(odd_list)
            material = training_set[i][j]
            even_list.append(
                material[0:(len(material) - 2)])
            if index_ == 0:
                data = pd.DataFrame([even_list])
                index_ += 1
            else:
                df = pd.DataFrame([even_list])
                data = pd.concat([data, df], axis=0)

                index_ += 1

    test_data = []
    # TESTING AND CLASSIFYING
    for idx_test, tests in enumerate(test_set):  # TEST_SET is just a list of materials
        t_data_mat = []
        for test in tests:
            t_data_mat.append(np.vstack(
                load_data(test, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120,
                          normalize_metric=normalize, amplitude=amplitude,
                          ignore_80=ignore_80)).T)  # THE ACTUAL DATA ARRAY
        test_data.append(t_data_mat)

    testData = pd.DataFrame
    index_ = 0
    df = pd.DataFrame

    i = 0
    j = 0
    for i, test in enumerate(test_data):
        for j, mat_data in enumerate(test):
            even_list = []
            odd_list = []
            for mat_type_data in mat_data:
                for index, value in enumerate(mat_type_data):
                    if index % 2 == 0:
                        even_list.append(value)

                    else:
                        odd_list.append(value)

            even_list.extend(odd_list)
            material = test_set[i][j]
            even_list.append(material[0:(len(material) - 2)])

            if index_ == 0:
                testData = pd.DataFrame([even_list])
                index_ += 1
            else:
                df = pd.DataFrame([even_list])
                testData = pd.concat([testData, df], axis=0)

                index_ += 1

    # classify_original(test_set, training_set, confusion, verbose=True, absolute_depth=True, linear_stage=True,
    #                   relative_120=True, amplitude=False, normalize=False, ignore_80=False)

    return data, testData


def have_zero(array):
    return any([True if v == 0 else False for v in array])


def phase2depth(phase, omega_MHz=16., c_mm_ns=300.):
    '''
    Convert phase to depth. The unit of returned depth is milli-meters.

    Parameters
    ----------
    phase: float
        Phase range from 0 to 2PI.
    omega_MHz: float
        Frequency in Mega-Hertz.
    c_mm_ns: float
        Speed of light. milli-meter per nano-second.
    '''
    return c_mm_ns * phase / (2. * math.pi) * 1000. / omega_MHz / 2.


def load_data(targ, base='base00', absolute_depth=True, linear_stage=True, relative_120=True, normalize_metric=True,
              guarantee=None, amplitude=False, ignore_80=False, points=200, relative_center_depth_only=False,
              relative_frequency_only=False, both_axis=False):

    file1_base = os.path.join('data', base, 'phase_depth_0.dat')
    file2_base = os.path.join('data', base, 'phase_depth_1.dat')
    file3_base = os.path.join('data', base, 'phase_depth_2.dat')
    file1_targ = os.path.join('data', targ, 'phase_depth_0.dat')
    file2_targ = os.path.join('data', targ, 'phase_depth_1.dat')
    file3_targ = os.path.join('data', targ, 'phase_depth_2.dat')
    file1a_targ = os.path.join('data', targ, 'amp_depth_0.dat')
    file2a_targ = os.path.join('data', targ, 'amp_depth_1.dat')
    file3a_targ = os.path.join('data', targ, 'amp_depth_2.dat')
    # acc = reader.read_float_file(os.path.join('data', targ, 'accumurate_depth.dat'))
    # depths = reader.read_float_file(os.path.join('data', targ, 'depth_data.dat'))

    d16_base = phase2depth(reader.read_float_file(file2_base), 16.)
    d80_base = phase2depth(reader.read_float_file(file1_base), 80.)
    d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
    d16 = phase2depth(reader.read_float_file(file2_targ), 16.)
    d80 = phase2depth(reader.read_float_file(file1_targ), 80.)
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.)
    a16 = reader.read_float_file(file2a_targ)
    a80 = reader.read_float_file(file1a_targ)
    a120 = reader.read_float_file(file3a_targ)

    if relative_center_depth_only:
        center_idx = int(len(d16) // 2)
        new_depths = depths - depths[center_idx]
        new_d80 = d80 - d80[center_idx]
        return new_d80 - new_depths

    if relative_frequency_only:  # IF I WANT TO RETURN RELATIVE FREUQUENCIES?
        center_idx = int(len(d16) // 2)
        return np.array((d120[center_idx] - d80[center_idx], d120[center_idx] - d16[center_idx]))

    if both_axis:
        center_idx = int(len(d16) // 2)
        new_depths = depths - depths[center_idx]
        new_d80 = d80 - d80[center_idx] - new_depths
        new_d120 = d120 - d120[center_idx] - new_depths
        new_d16 = d16 - d16[center_idx] - new_depths
        return np.hstack((new_d80, new_d120, new_d16))

    if absolute_depth:
        d16 -= d16_base
        d80 -= d80_base
        d120 -= d120_base

    # if not linear_stage:
    #        d16, d80, d120, a16, a80, a120, acc = convert_axis_S2D(d16, d80, d120, a16, a80, a120, acc, depths, points=points)


    if relative_120:
        d16 -= d120
        d80 -= d120
        d120 -= d120
        a16 = np.array([0 if d == 0 else v / d for v, d in zip(a16, a120)])
        a80 = np.array([0 if d == 0 else v / d for v, d in zip(a80, a120)])
        a120 = np.array([0 if v == 0 else 1. for v in a120])

    mean_normalizer = np.zeros(6)
    std_normalizer = np.ones(6)
    #    if normalize_metric:
    #        cond_num = condition_number(absolute_depth, linear_stage, relative_120, amplitude, ignore_80)
    #        f = open('results/normalization_coefficients.pickle', 'rb')
    #        norm_coef = pickle.load(f)
    #        mean_normalizer, std_normalizer = norm_coef[cond_num]

    if isinstance(guarantee, int):
        #        d16 = np.array([v for v, a in zip(d16, acc) if a > guarantee])
        #        d80 = np.array([v for v, a in zip(d80, acc) if a > guarantee])
        #        d120 = np.array([v for v, a in zip(d120, acc) if a > guarantee])
        #        a16 = np.array([v for v, a in zip(a16, acc) if a > guarantee])
        #        a80 = np.array([v for v, a in zip(a80, acc) if a > guarantee])
        #        a120 = np.array([v for v, a in zip(a120, acc) if a > guarantee])
        d16 = np.array([0 if a < guarantee else v for v, a in zip(d16, acc)])
        d80 = np.array([0 if a < guarantee else v for v, a in zip(d80, acc)])
        d120 = np.array([0 if a < guarantee else v for v, a in zip(d120, acc)])
        a16 = np.array([0 if a < guarantee else v for v, a in zip(a16, acc)])
        a80 = np.array([0 if a < guarantee else v for v, a in zip(a80, acc)])
        a120 = np.array([0 if a < guarantee else v for v, a in zip(a120, acc)])

    if relative_120:
        if amplitude:
            if not ignore_80:
                return ((seq - m) / s for seq, m, s in zip((d16, d80, a16, a80), mean_normalizer, std_normalizer))
            else:
                return ((seq - m) / s for seq, m, s in zip((d16, a16), mean_normalizer, std_normalizer))
        else:
            if not ignore_80:
                return ((seq - m) / s for seq, m, s in zip((d16, d80), mean_normalizer, std_normalizer))
            else:
                return (d16 - mean_normalizer[0]) / std_normalizer[0]

    if amplitude:
        return ((seq - m) / s for seq, m, s in zip((d16, d80, d120, a16, a80, a120), mean_normalizer, std_normalizer))
    else:
        return ((seq - m) / s for seq, m, s in zip((d16, d80, d120), mean_normalizer, std_normalizer))


def valid_l2_norm(vec1, vec2, ave=False):
    l2 = np.linalg.norm(vec1 - vec2, axis=1)
    valid = np.array([0 if have_zero(t) or have_zero(p) else 1 for t, p in zip(vec1, vec2)])
    if not ave:
        return sum(l2 * valid)
    else:
        return sum(l2 * valid) / sum(valid) / vec1.shape[1]


def valid_l2_norm2(vec1, vec2, ave=False):
    dif = vec1 - vec2
    l2 = np.sqrt(dif * dif)
    valid = np.array([0 if t == 0 or p == 0 else 1 for t, p in zip(vec1, vec2)])
    if not ave:
        return sum(l2 * valid)
    else:
        return sum(l2 * valid) / sum(valid) / vec1.shape[1]


def preprocess(data):
    plastics = ['polystyrene', 'epvc','pvc', 'pp', 'acryl', 'acryl3mm', 'acryl2mm', 'acryl1mm']
    counter = 0
    for material in data.iloc[:, 3400]:
        if material in plastics:
            data.iloc[counter, 3400] = 'plastic'
        else:
            data.iloc[counter, 3400] = 'residual'
        counter+=1
    return data

#
# def call_python_version(Version, Module, Function, ArgumentList):
#     gw = execnet.makegateway("Popen//python=python%s" % Version)
#     channel = gw.remote_exec("""
#         from %s import %s as the_function
#         channel.send(the_function(*channel.receive()))
#     """ % (Module, Function))
#     channel.send(ArgumentList)
#     return channel.receive()


def main_f():
    # What materials to train with?
    mats = ['polystyrene', 'epvc','pvc', 'pp', 'acryl', 'acryl3mm', 'acryl2mm', 'acryl1mm',
            'alumi',  'copper', 'ceramic',
            'plaster','paper', 'blackpaper', 'wood',
            'cork', 'mdf', 'bamboo', 'cardboard',
            'fabric', 'fakeleather', 'leather', 'carpet',
            'silicone',
            'whiteglass', 'sponge']


    print(mats)
    # Retreive the data
    trainData, testData = cross_validation_nearest_neighbor_classifier(mats, rep=20, max_index=12, num_training=4,
                                                                       absolute_depth=False)

    print("02 PREPROCESSING DATA")
    # Preprocess
    trainData = preprocess(trainData)
    # testData = preprocess(testData)

    # Convert to float and Replace zeros with NaN
    for row in range(trainData.shape[1] - 1):
        trainData[row] = trainData[row].astype(float)
    #
    # for row in range(trainData.shape[1] - 1):
    #     testData[row] = testData[row].astype(float)

    for row in range(testData.shape[1] - 1):
        trainData[row] = trainData[row].astype(float)

    # for row in range(testData.shape[1] - 1):
    #     testData[row] = testData[row].astype(float)

    trainData[:] = trainData[:].replace({0.000000: np.nan, 0: np.nan})
    # testData[:] = testData[:].replace({0.000000: np.nan, 0: np.nan})


    # -------------- CLASIFICATION PROCESS ----------------

    imputation = 'Iterative'
    _range = 1

    # Impute the values
    # testData.iloc[:, 0:3400] = impute(testData, imputation)

    from imputer import impute



    # python2_command = "imputer.py trainData imputation"
    # process = subprocess.Popen(python2_command.split(), stdout=subprocess.PIPE)
    # trainData.iloc[:, 0:3400], error = process.communicate()


    # trainData.iloc[:, 0:3400] = call_python_version("2.6", "imputer", "impute", [trainData, imputation])

    trainData.iloc[:, 0:3400] = impute(trainData, imputation)

    X_train = trainData.iloc[:, 0:3400]
    y_train = trainData.iloc[:, 3400]

    # X_test = testData.iloc[:, 0:3400]
    # y_test = testData.iloc[:, 3400]

    # Normalise
    # X_train.iloc[:, :] = (X_train.iloc[:, :] - np.nanmean(X_train.iloc[:, :], axis=0))/np.nanstd(X_train.iloc[:, :], axis=0)
    # X_test.iloc[:, :] = (X_test.iloc[:, :] - np.nanmean(X_test.iloc[:, :], axis=0))/np.nanstd(X_test.iloc[:, :], axis=0)

    # accuracy_lr = []
    # y_lr = {}
    LogisticRegression_clf = LogisticRegression(random_state=200)
    LogisticRegression_clf.fit(X_train, y_train)
    # y_lr = LogisticRegression_clf.predict(X_test)
    # accuracy_lr.append(accuracy_score(y_test, y_lr))

    # accuracy_svc = []
    # y_svc = {}
    SVC_clf = SVC(C=1.0, gamma='auto', kernel='rbf')
    SVC_clf.fit(X_train, y_train)
    # y_svc = SVC_clf.predict(X_test)
    # accuracy_svc.append(accuracy_score(y_test, y_svc))

    # accuracy_tree = []
    # y_tree = {}
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=10)
    DecisionTree_clf.fit(X_train, y_train)
    # y_tree = DecisionTree_clf.predict(X_test)
    # accuracy_tree.append(accuracy_score(y_test, y_tree))

    # accuracy_neigh = []
    KNN_clf = KNeighborsClassifier(n_neighbors=2, weights='distance')
    KNN_clf.fit(X_train, y_train)
    # y_neigh = KNN_clf.predict(X_test)
    # accuracy_neigh.append(accuracy_score(y_test, y_neigh))

    # print("Logistic Regression: {}".format(sum(accuracy_lr) / _range))
    # print("SVC: {}".format(sum(accuracy_svc) / _range))
    #
    # print("Tree: {}".format(sum(accuracy_tree) / _range))
    #
    # print("Neighbourhood: {}".format(sum(accuracy_neigh) / _range))


    classifiers = [LogisticRegression_clf, SVC_clf, DecisionTree_clf, KNN_clf]

    return classifiers

if __name__ == '__main__':

    pass