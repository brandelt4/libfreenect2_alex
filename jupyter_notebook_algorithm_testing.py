#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler, SimpleFill
import matplotlib.pyplot as plt
from scipy import signal
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler, IterativeSVD
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split




counterr=0


# # Defining functions

# In[10]:


def cross_validation_nearest_neighbor_classifier(materials, rep=10, max_index=1, num_test=0, num_training=3, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, normalize=True, ignore_80=False):
    confusion = np.zeros((len(materials), len(materials)))
    filename = 'results/confusion_'
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
        targ.append(tm)                    # TEST SET
        pm = []
        while len(pm) < num_training:
            p = m + str(random.randint(0, max_index)).zfill(2)
            if not (p in pm or p in tm):
                pm.append(p)
        prob.append(pm)                    # TRAINING SET
    data, testData = nearest_neighbor_classify(targ, prob, confusion, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize=normalize, amplitude=amplitude, ignore_80=ignore_80)


    return data, testData
    


def nearest_neighbor_classify(test_set, training_set, confusion, verbose=True, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, normalize=False, ignore_80=False):
    # GETTING TRAINING DATA
    testData = None
    training_data = []
    for mats in training_set: # training_set = [[plastic01, ...], []]
        t_data_mat = []
        for m in mats:
            t_data_mat.append(np.vstack(load_data(m, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize_metric=normalize, amplitude=amplitude, ignore_80=ignore_80)).T)
        training_data.append(t_data_mat)

        
    data = pd.DataFrame
    index_ = 0
    df = pd.DataFrame
    
    i=0
    j=0
    for i, train in enumerate(training_data):
        for j, mat_data in enumerate(train):
            even_list = [] # first frequency
            odd_list = [] # second frequency
            for mat_type_data in mat_data:
                for index, value in enumerate(mat_type_data):
                    if index % 2 == 0:
                        even_list.append(value)

                    else:
                        odd_list.append(value)

            even_list.extend(odd_list)
            material = training_set[i][j]
            even_list.append(material[0:(len(material)-2)])  # last column–target– we append the name removing 01,02, etc.

            if index_ == 0:
                data = pd.DataFrame([even_list])
                index_ += 1
            else:
                df = pd.DataFrame([even_list])
                data = pd.concat([data,df],axis=0)

                index_ += 1



    test_data = []
#     # TESTING AND CLASSIFYING
#     for idx_test, tests in enumerate(test_set): # TEST_SET is just a list of materials
#         t_data_mat = []
#         for test in tests:
#             t_data_mat.append(np.vstack(load_data(test, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize_metric=normalize, amplitude=amplitude, ignore_80=ignore_80)).T)  # THE ACTUAL DATA ARRAY
#         test_data.append(t_data_mat)
        
        
    
    
#     testData = pd.DataFrame
#     index_ = 0
#     df = pd.DataFrame
    
#     i=0
#     j=0
#     for i, test in enumerate(test_data):
#         for j, mat_data in enumerate(test):
# #             print(mat_data)
#             even_list = []
#             odd_list = []
#             for mat_type_data in mat_data:
                
#                 for index, value in enumerate(mat_type_data):
#                     if index % 2 == 0:
#                         even_list.append(value)

#                     else:
#                         odd_list.append(value)

#             even_list.extend(odd_list)
#             material = test_set[i][j]
#             even_list.append(material[0:(len(material)-2)])

#             if index_ == 0:
#                 testData = pd.DataFrame([even_list])
#                 index_ += 1
#             else:
#                 df = pd.DataFrame([even_list])
#                 testData = pd.concat([testData,df],axis=0)

#                 index_ += 1
                
#     classify_original(test_set, training_set, confusion, verbose=True, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, normalize=False, ignore_80=False)
            
    
    return data, testData



# In[12]:


def classify_original (test_set, training_set, confusion, verbose=True, absolute_depth=True, linear_stage=True, relative_120=True, amplitude=False, normalize=False, ignore_80=False):
    success = 0
    failure = 0
    training_data = []
    for mats in training_set: # training_set = [[plastic01, ...], []]
        t_data_mat = []
        for m in mats:
            t_data_mat.append(np.vstack(load_data(m, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize_metric=normalize, amplitude=amplitude, ignore_80=ignore_80)).T)
        training_data.append(t_data_mat)
    
    
    show = True
    for idx_test, tests in enumerate(test_set): # TEST_SET is just a list of materials
        for test in tests:
            test_vec = np.vstack(load_data(test, absolute_depth=absolute_depth, linear_stage=linear_stage, relative_120=relative_120, normalize_metric=normalize, amplitude=amplitude, ignore_80=ignore_80)).T  # THE ACTUAL DATA ARRAY
            class_costs = []
            for idx_ref, materials in enumerate(training_set):
                costs = []
                for idx_tmp, ref in enumerate(materials):
                    if show == True:
                        print(test_vec)
                        show = False
                    costs.append(valid_l2_norm(test_vec, training_data[idx_ref][idx_tmp]))  # Find l2-norm, for a particular material idx_ref of particular sample id_tmp, and add to costs
                class_costs.append(min(costs))    # Contains all (minimum for each material sample) l2-norm values (test_vec - training material)
            nn = np.argmin(class_costs)   # Returns indices of the minimum value from class_costs
            if idx_test == nn:
                success += 1
            else:
                failure += 1

    print(success)
    print(failure)

    
    
    


# In[13]:


def have_zero(array):
    return any([True if v==0 else False for v in array])
    
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
    
        
def load_data(targ, base='base00', absolute_depth=True, linear_stage=True, relative_120=True, normalize_metric=True, guarantee=None, amplitude=False, ignore_80=False, points=200, relative_center_depth_only=False, relative_frequency_only=False, both_axis=False):
    ''' ステージパルス基準で計測したデータを読み込む．
    
    Parameters
    ----------
    absolute_depth : bool
        If True, returns the relative depth distortion against the 'base' material.
    linear_stage
        Not used.
    relative_120
        If True, returned is 2 relative values from the measurement of 120MHｚ. Otherwise, 3 absolute values.
    normalize_metric
        Not used.
        
    '''
    file1_base = os.path.join('data', base, 'phase_depth_0.dat')        
    file2_base = os.path.join('data', base, 'phase_depth_1.dat')        
    file3_base = os.path.join('data', base, 'phase_depth_2.dat')      
    file1_targ = os.path.join('data', targ, 'phase_depth_0.dat')        
    file2_targ = os.path.join('data', targ, 'phase_depth_1.dat')        
    file3_targ = os.path.join('data', targ, 'phase_depth_2.dat')      
    file1a_targ = os.path.join('data', targ, 'amp_depth_0.dat')        
    file2a_targ = os.path.join('data', targ, 'amp_depth_1.dat')        
    file3a_targ = os.path.join('data', targ, 'amp_depth_2.dat')        
    acc = reader.read_float_file(os.path.join('data', targ, 'accumurate_depth.dat'))
    depths = reader.read_float_file(os.path.join('data', targ, 'depth_data.dat'))
    
    d16_base  = phase2depth(reader.read_float_file(file2_base), 16.)
    d80_base  = phase2depth(reader.read_float_file(file1_base), 80.)
    d120_base = phase2depth(reader.read_float_file(file3_base), 120.)
    d16  = phase2depth(reader.read_float_file(file2_targ), 16.)
    d80  = phase2depth(reader.read_float_file(file1_targ), 80.) 
    d120 = phase2depth(reader.read_float_file(file3_targ), 120.) 
    a16  = reader.read_float_file(file2a_targ)
    a80  = reader.read_float_file(file1a_targ)
    a120 = reader.read_float_file(file3a_targ)

    if relative_center_depth_only:
        center_idx = int(len(d16) // 2)
        new_depths = depths - depths[center_idx]
        new_d80 = d80 - d80[center_idx]
        return new_d80 - new_depths
        
    if relative_frequency_only:  #IF I WANT TO RETURN RELATIVE FREUQUENCIES?
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
        d120-= d120_base
    
#    if not linear_stage:
#        d16, d80, d120, a16, a80, a120, acc = convert_axis_S2D(d16, d80, d120, a16, a80, a120, acc, depths, points=points)

        
    if relative_120:
        d16 -= d120
        d80 -= d120
        d120 -= d120
        a16 = np.array([0 if d == 0 else v / d for v, d in zip(a16, a120)])
        a80 = np.array([0 if d == 0 else v / d for v, d in zip(a80, a120)])
        a120 = np.array([0 if d == 0 else 1. for v in a120])
    
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
        d16  = np.array([0 if a < guarantee else v for v, a in zip(d16 , acc)])
        d80  = np.array([0 if a < guarantee else v for v, a in zip(d80 , acc)])
        d120 = np.array([0 if a < guarantee else v for v, a in zip(d120, acc)])
        a16  = np.array([0 if a < guarantee else v for v, a in zip(a16 , acc)])
        a80  = np.array([0 if a < guarantee else v for v, a in zip(a80 , acc)])
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


# In[14]:


def valid_l2_norm(vec1, vec2, ave=False):
    l2 = np.linalg.norm(vec1 - vec2, axis=1)
    valid = np.array([0 if have_zero(t) or have_zero(p) else 1 for t, p in zip(vec1, vec2)])
    if not ave:
        return sum(l2 * valid)
    else:
        return sum(l2 * valid) / sum(valid) / vec1.shape[1]
    
def valid_l2_norm2(vec1, vec2, ave=False):
    dif = vec1 - vec2
    l2 = np.sqrt(dif*dif)
    valid = np.array([0 if t==0 or p==0 else 1 for t, p in zip(vec1, vec2)])
    if not ave:
        return sum(l2 * valid)
    else:
        return sum(l2 * valid) / sum(valid) / vec1.shape[1]


# In[15]:


mats = ['polystyrene', 'epvc','pvc', 'pp', 
        'acryl', 'acryl3mm', 'acryl2mm', 'acryl1mm', 
       'alumi',  'copper', 'ceramic', 
        'plaster','paper', 'blackpaper',  'wood', 
        'cork', 'mdf', 'bamboo', 'cardboard',
        'fabric', 'fakeleather', 'leather', 'carpet',
         'silicone',
          'whiteglass', 'sponge']
        
        


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


def remove_outliers_smooth(newData):
        print('-'*40)
        print(type(newData))
        print('-'*40)
        df2 = newData.iloc[:, 0:3400].rolling(30).mean()

        b, a = signal.butter(3, 0.05)
        y = signal.filtfilt(b,a, newData.iloc[:, 0:3400].values)

        df3 = pd.DataFrame(y, index=df2.index)
        
#         print(df3)
        
        return df3

def impute(data, imputation):

    # Imputation technique
    newData = data.copy()
    newData1 = data.copy()

    _newData = newData.values
    
    if imputation == 'Iterative':
#         newData.iloc[:, 0:3400] = IterativeImputer(n_iter=50, random_state=0).fit_transform(data.iloc[:,0:3400])
        newData1.iloc[:, 0:3400] = IterativeImputer().fit_transform(data.iloc[:,0:3400])

#         print(newData)
        return remove_outliers_smooth(newData1)
#         return newData1

    elif imputation == 'KNN':
        newData.iloc[:, 0:3400] = KNN(k=3).fit_transform(data.iloc[:,0:3400])
#         return remove_outliers_smooth(newData)
        return newData

    elif imputation == 'IterativeSVD':
        newData.iloc[:, 0:3400] = IterativeSVD().fit_transform(data.iloc[:,0:3400])
        return remove_outliers_smooth(newData)
    
    elif imputation == 'simple':
        lastAvailable = None
        print(newData.shape[0], newData.shape[1])
        for row in range(newData.shape[0]):
            print(row)
            for column in range(newData.shape[1]-1, 0, -1):
                print(type(newData.iloc[row, column]))
                if newData.iloc[row, column] == 'nan':
                    print('NONE')
                    if lastAvailable is not None:
                        newData.iloc[row, column] = lastAvailable
                        print('Assigned: {}'.format(lastAvailable))
                    else:
                        print('Trying to find...')
                         # Iterate until you find the next value
                        for _column in range(newData.shape[1]-1, 0, -1): 
                            if newData.iloc[row, _column] is not None:
                                print('Found!')
                                newData.iloc[row, column] = newData.iloc[row, _column]
                            else:
                                continue
                else:
                    print('Normal lastAvailable')
                    lastAvailable = newData.iloc[row, column]
                    
                    
        return newData



# Read the data

items = ['corona', 'evian','foil','foodbox','lemsip','napkin','paperbag','teabox','wafflebox']


# In[11]:

if __name__ == '__main__':

    all_data = pd.DataFrame
    i = 0
    for item in items:
        for trial in ['', '1', '2', '3', '4', '5']:

            df = pd.read_excel('C:\\libfreenect2_alex\\build\\raised_data\\{}{}\\{}.xlsx'.format(item, trial, item))

            if i == 0:
                all_data = df

            else:
                all_data = pd.concat([all_data, df], ignore_index=True)

            i+=1


        
        
        
        


    all_data['material'] = 'mat'


    items = ['corona', 'evian','foil','foodbox','lemsip','napkin','paperbag','teabox','wafflebox']
    row = 0
    for item in items:
        for trial in ['', '1', '2', '3', '4', '5']:
            if item in ['evian', 'wafflebox','foodbox']:
                all_data.iloc[row, -1] = 'plastic'
            else:
                all_data.iloc[row, -1] = 'residual'

            row+=1


    imputation = 'Iterative'

    # test = testData.copy()
    train = all_data.copy()


    # Impute the values
    # test.iloc[:, 0:3400] = impute(testData, imputation )
    train.iloc[:, 0:3400] = impute(train.iloc[:, 0:3400], imputation )
    train = train.drop(columns=['Unnamed: 0'])
    train.to_excel('all_data_explore.xlsx')

    # Normalise
    train.iloc[:, 0:1700] = (train.iloc[:, 0:1700] - np.nanmean(train.iloc[:, 0:1700], axis=0))/np.nanstd(train.iloc[:, 0:1700], axis=0)
    # test.iloc[:, 0:1700] = (test.iloc[:, 0:1700] - np.nanmean(test.iloc[:, 0:1700], axis=0))/np.nanstd(test.iloc[:, 0:1700], axis=0)
    train.iloc[:, 1700:3400] = (train.iloc[:, 1700:3400] - np.nanmean(train.iloc[:, 1700:3400], axis=0))/np.nanstd(train.iloc[:, 1700:3400], axis=0)
    # test.iloc[:, 1700:3400] = (test.iloc[:, 1700:3400] - np.nanmean(test.iloc[:, 1700:3400], axis=0))/np.nanstd(test.iloc[:, 1700:3400], axis=0)


    X_all = train.iloc[:, 0:3400]
    y_all = train.iloc[:, 3400]

    # X_test = test.iloc[:, 0:3400]
    # y_test = test.iloc[:, 3400]



    # TRAINING THE DATA

    accuracy_lr = []
    y_lr = {}

    clf_A = LogisticRegression(random_state=200)
    clf_A.fit(X_all, y_all)


    accuracy_svc = []
    y_svc = {}
    clf_B = SVC(C=1.0, gamma=0.001200, kernel='rbf')
    # clf_B = SVC(C=1.0, gamma='auto', kernel='rbf')
    clf_B.fit(X_all, y_all)


    accuracy_tree = []
    y_tree = {}
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=321)
    clf.fit(X_all, y_all)


    accuracy_neigh = []
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=12, weights='distance')
    neigh.fit(X_all, y_all)


    accuracy_xgb = []
    xgb_model = XGBClassifier()
    xgb_model.fit(X_all, y_all)



    _range = 1


    print("Logistic Regression: {}".format(sum(accuracy_lr)/_range))
    print("SVC: {}".format(sum(accuracy_svc)/_range))
    print("Tree: {}".format(sum(accuracy_tree)/_range))
    print("Neighbourhood: {}".format(sum(accuracy_neigh)/_range))
    print("XGB: {}".format(sum(accuracy_xgb)/_range))



    # In[72]:


    # Convert to float
    for row in range(all_data.shape[1]-1):
        all_data[row] = all_data[row].astype(float)

    print(type(all_data.iloc[0,0]))


    # In[73]:


    import pickle

    classifiers = [clf_A, clf_B, clf, neigh]


    with open('classifiers_latest.pkl', 'wb') as output:
        pickle.dump(classifiers, output, pickle.HIGHEST_PROTOCOL)

    with open('train_data.pkl', 'wb') as output:
        pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)



