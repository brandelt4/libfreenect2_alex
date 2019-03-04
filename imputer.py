#!/usr/bin/env python
from fancyimpute import KNN, IterativeImputer, IterativeSVD
from scipy import signal
import pandas as pd



def remove_outliers_smooth(newData):
    df2 = newData.iloc[:, 0:3400].rolling(20).mean()

    b, a = signal.butter(3, 0.05)
    y = signal.filtfilt(b, a, newData.iloc[:, 0:3400].values)

    df3 = pd.DataFrame(y, index=df2.index)

    print(df3)

    return df3

def impute(data, imputation):
    # Imputation technique

    print("IM IMPUTING!!!!!!!!")
    newData = data.copy()
    _newData = newData.values

    if imputation == 'Iterative':
        newData.iloc[:, 0:3400] = IterativeImputer().fit_transform(data.iloc[:, 0:3400])
        print(newData)
        return remove_outliers_smooth(newData)

    elif imputation == 'KNN':
        newData.iloc[:, 0:3400] = KNN(k=3).fit_transform(data.iloc[:, 0:3400])
        return remove_outliers_smooth(newData)

    elif imputation == 'IterativeSVD':
        newData.iloc[:, 0:3400] = IterativeSVD().fit_transform(data.iloc[:, 0:3400])
        return remove_outliers_smooth(newData)