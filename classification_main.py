#!/usr/bin/env python

    
import sys
import os
import time
import dat2png as reader
import math
import pickle
from preprocessing import main_f, preprocess, impute, replace_zeros_with_nan, impute_test_vec, normalise
# from auto_invoke_demos import start_kinect
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import numpy as np
from PyQt5 import QtCore
# ARDUINO
from returnRanking import calculate_

global app


# mats = ['alumi',   'copper', 'ceramic', #'stainless',
#         'paper', 'blackpaper',  'wood',     'cork', 'mdf', 'bamboo', 'cardboard',
#          'fabric', 'fakeleather', 'leather', 'carpet',
#         #'banana', 'fakebanana', 'fakeapple',
#         'plaster', 'polystyrene', 'epvc', #  'pvc', 'silicone', 'pp',
#         'acryl', 'acryl3mm', 'acryl2mm', 'acryl1mm',  'whiteglass']

mats = ['plastic', 'residual']
        
# mat_label = ['Metal - Aluminum',   'Metal - Copper', 'Ceramic', #'stainless',
#         'Paper', 'Flock paper',  'Wood - Natural', 'Wood - Cork', 'Wood - MDF', 'Wood - Bamboo', 'Paper - Cardboard',
#          'Fabric - Cotton', 'Fabric - Fake leather', 'Fabric - Leather', 'Fabric - Carpet',
#         #'Plant - Banana', 'Plastic - Unknown', 'Plastic - Unknown',
#         'Plaster', 'Plastic - PS', 'Plastic - E-PVC', #  'Plastic - PVC', 'Plastic - Silicone', 'Plastic - PP',
#         'Plastic - Acryl', 'Plastic - Acryl, 3mm', 'Plastic - Acryl, 2mm', 'Plastic - Acryl, 1mm',  'Diffusion glass']

mat_label = ['Plastic Waste', 'Residual Waste']
        
test_mats = ['paper', 'plaster', 'acryl']
ignored = ['fakebanana', 'fakeapple', 'banana', 'cardboard', 'polystyrene']
iteration=0

#pi2 = math.pi #/ 2.
def phase2depth(phase, omega_MHz=16., c_mm_ns=300.):
    '''
    Convert phase to depth. The unit of returned depth is milli-meters.
    
    Parameters
    ----------
    phase: float
        Phase ranges from 0 to 2PI.
    omega_MHz: float
        Frequency in Mega-Hertz.
    c_mm_ns: float
        Speed of light. milli-meter per nano-second.
    '''
#    if omega_MHz > 100:
#        phase = np.array([p + 2.* math.pi if p < pi2 else p for p in phase])
    return c_mm_ns * phase / (2. * math.pi) * 1000. / omega_MHz / 2.
    
 
def have_zero(array):
    return any([True if v==0 else False for v in array])
 
def valid_l2_norm(vec1, vec2):
    l2 = np.linalg.norm(vec1 - vec2, axis=0)
    valid = np.array([0 if have_zero(t) or have_zero(p) else 1 for t, p in zip(vec1.T, vec2.T)])
    return sum(l2 * valid)          

class AppFormNect():
    ''' Main application GUI form for scatter plot. Watches Protonect output files and calculate phase values.
    
    Attributes
    ----------
    x_file : str
        Filename of the x values of plot data.
    y_file : str
        Filename of the y values of plot data.
    wait_for_file_close : float
        Wait time between file modified detection and file open for load data.
    scatterplot : ScatterPlot
        Plot widget wrapping matplotlib.

    '''

    def __init__(self, parent=None, file1='phase_depth_0.dat', 
                                    file2='phase_depth_1.dat', 
                                    file3='phase_depth_2.dat', 
                                    wait_for_file_close=.01,
                                    accuracy=10,
                                    debug=False):
        # QMainWindow.__init__(self, parent)
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.wait_for_file_close = wait_for_file_close
        self.accuracy = accuracy

        # Retreive the ID of the current item
        with open('material_number.txt', 'r') as file:
            self.i = int(file.readline())
        with open('material_number.txt', 'w') as file:
            file.write(str(self.i+1))

        # Creating file directory for storage
        self.folder_name = 'Tr-lipton'
        # os.mkdir('raised_data/{}'.format(self.folder_name))

        created = False
        i = 1
        while created is not True:
            try:
                os.mkdir('raised_data/{}{}'.format(self.folder_name, i))
                self.folder_name = self.folder_name + str(i)
                created = True
            except:
                i += 1
                continue

        self.estimate_material()


    def _on_file_changed(self):
        time.sleep(self.wait_for_file_close)
        self.estimate_material()
        

    def load_file(self):
        flag = True
        flag &= os.path.exists(self.file1)
        flag &= os.path.exists(self.file2)
        flag &= os.path.exists(self.file3)
        self.all_file_exists = flag
        if flag:
            self.p16  = phase2depth(reader.read_float_file(self.file2), 16.)
            self.p80  = phase2depth(reader.read_float_file(self.file1), 80.)
            self.p120 = phase2depth(reader.read_float_file(self.file3), 120.)
            self.acc = reader.read_float_file('accumurate_depth.dat')
            self.d80 = np.array([0 if a < self.accuracy else b - c for a, b, c in zip(self.acc, self.p80, self.p120)])
            self.d16 = np.array([0 if a < self.accuracy else b - c for a, b, c in zip(self.acc, self.p16, self.p120)])

        
    def estimate_material(self):
        # Kill kinect
        os.system("taskkill /im Protonect.exe")

        # while numOfNan > 1000:
        # Load the data from the file and save to self.d18 and self.80
        self.load_file()

        # Getting the collected data
        test_vec = np.vstack((self.d16, self.d80))

        # Formatting
        print('Formatting the data...')
        test_vec = replace_zeros_with_nan(calculate_(test_vec))
        test_vec.to_excel('raised_data/{}/raw_data.xlsx'.format(self.folder_name,self.folder_name, self.i))
        numOfNan = test_vec.isna().sum().sum()
        print('Currently unknown: {}'.format(numOfNan))

        train_data = pd.read_pickle("train_data.pkl")
        train_data = train_data.reset_index(drop=True)
        train_data = train_data.drop(['material'], axis=1)

        test_vec = pd.concat([test_vec, train_data.loc[1:10]], ignore_index=True)


        # global iteration

        # Checking if enough data was collected
        if numOfNan < 2500:

            # Imputing data
            print("Imputing the data...")
            # changeActivity('Imputting the data...')

            array = impute_test_vec(test_vec, "Iterative")
            test_vec = array

            test_vec = test_vec.drop([1,2,3,4,5,6,7,8,9,10], axis=0)

            l = []
            for i in range(1,55):
                l.append(i)

            # Normalise
            test_vec = pd.concat([test_vec, train_data], ignore_index=True)
            test_vec = normalise(test_vec)
            test_vec = test_vec.drop(l, axis=0)

            try:
                test_vec = test_vec.drop([3400], axis=1)
                test_vec.drop(test_vec.columns[len(test_vec.columns)-1], axis=1, inplace=True)
            except:
                pass

            test_vec.drop(test_vec.columns[len(test_vec.columns) - 1], axis=1, inplace=True)

            with open('classifiers_latest.pkl', 'rb') as input:
                classifiers = pickle.load(input)

            # rankingLR = classifiers[0].predict(test_vec)
            rankingSVC = classifiers[1].predict(test_vec)
            # rankingDT = classifiers[2].predict(test_vec)
            # rankingKNN = classifiers[3].predict(test_vec)


            print('-' * 40)
            print("-----------CURRENT BEST PREDICTION------------")
            # print("Logistic Regression: {}".format(rankingLR))
            print("SVC: {}".format(rankingSVC))
            # print("Decision Tree: {}".format(rankingDT))
            # print("KNN: {}".format(rankingKNN))
            print('-' * 40)

            # send_plastic()

        else:
            print("Not enough data was collected. Number of NaN: {}".format(numOfNan))


        
def main(args):
    # app = QApplication(args)
    global app
    app = AppFormNect()




if __name__ == "__main__":
    main(sys.argv)


