#!/usr/bin/env python

    
import sys
import os
import time
import dat2png as reader
import math
import pickle
from train_classifier import main_f, preprocess, impute, replace_zeros_with_nan, impute_test_vec
# from auto_invoke_demos import start_kinect
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import numpy as np
from PyQt5 import QtCore

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
        
    Examples
    --------
    # >>> app = QApplication(sys.argv)
    # >>> form = AppForm()
    # >>> form.show()
    # >>> sys.exit(app.exec_())
    '''
#    def __init__(self, parent=None, file1='phase_depth_0_rt.dat', 
#                                    file2='phase_depth_1_rt.dat', 
#                                    file3='phase_depth_2_rt.dat', 
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

        # Creating file watcher

        # logging.basicConfig(level=logging.INFO,
        #                     format='%(asctime)s - %(message)s',
        #                     datefmt='%Y-%m-%d %H:%M:%S')
        # path = sys.argv[1] if len(sys.argv) > 1 else '.'
        # event_handler = Event(self)
        # observer = Observer()
        # observer.schedule(event_handler, path, recursive=True)
        # observer.start()
        # try:
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     observer.stop()
        # observer.join()

        while True:
            print("PLEASE PUT YOUR MATERIAL")
            time.sleep(10)
            self.estimate_material()



#        self.creat_main_window()
#         self.create_label_window()
#
        # Add watchdog for each file
        # if not debug:
        #     self.watcher = QtCore.QFileSystemWatcher()
        #
        #     self.directory_changed = self.watcher.directoryChanged
        #     self.file_changed = self.watcher.fileChanged
        #
        #     self.watcher.addPath(self.file1)
        #     self.watcher.addPath(self.file2)
        #     self.watcher.addPath(self.file3)
        #     # self.load_database()
        #     self.estimate_material()

        
        
    # def create_label_window(self):
    #     # window
    #     self.main_frame = QWidget()
    #     self.setGeometry(750, 0, 800, 500)
    #     self.setWindowTitle('Material Classifier')
    #
    #     # layout
    #     vbox = QVBoxLayout()
    #
    #     # widgets
    #     self.label = QLabel('Put material.')
    #     self.label.setFont(QFont('SansSerif', 40))
    #     self.mat2 = QLabel('rank2')
    #     self.mat2.setFont(QFont('SansSerif', 32))
    #     # self.mat3 = QLabel('rank3')
    #     # self.mat3.setFont(QFont('SansSerif', 28))
    #     # self.mat4 = QLabel('rank4')
    #     # self.mat4.setFont(QFont('SansSerif', 24))
    #     # self.mat5 = QLabel('rank5')
    #     # self.mat5.setFont(QFont('SansSerif', 20))
    #
    #     # set all
    #     vbox.addWidget(self.label)
    #     vbox.addWidget(self.mat2)
    #     # vbox.addWidget(self.mat3)
    #     # vbox.addWidget(self.mat4)
    #     # vbox.addWidget(self.mat5)
    #     self.main_frame.setLayout(vbox)
    #     self.setCentralWidget(self.main_frame)
    
    def _on_file_changed(self):
        time.sleep(self.wait_for_file_close)
        self.estimate_material()
        
    def load_database(self):
        pass
        # self.materials = []
        # self.training = []
        # for idx, mat in enumerate(mats):
        #     if mat in ignored:
        #         continue
        #     self.materials.append(mat_label[idx])
        #     self.training.append(np.load('data/'+mat+'/3mm.npy'))
#        self.materials = mat_label
#        self.training = [np.load('data/'+m+'/3mm.npy') for m in mats]
        
    def load_file(self):
        flag = True
        flag &= os.path.exists(self.file1)
        flag &= os.path.exists(self.file2)
        flag &= os.path.exists(self.file3)
        self.all_file_exists = flag
        print(flag)
        if flag:
            self.p16  = phase2depth(reader.read_float_file(self.file2), 16.)
            self.p80  = phase2depth(reader.read_float_file(self.file1), 80.)
            self.p120 = phase2depth(reader.read_float_file(self.file3), 120.)
            self.acc = reader.read_float_file('accumurate_depth.dat')
            self.d80 = np.array([0 if a < self.accuracy else b - c for a, b, c in zip(self.acc, self.p80, self.p120)])
            self.d16 = np.array([0 if a < self.accuracy else b - c for a, b, c in zip(self.acc, self.p16, self.p120)])
        
    # def clear_labels(self):
    #     self.mat2.setText('')
        # self.mat3.setText('')
        # self.mat4.setText('')
        # self.mat5.setText('')
        
    def estimate_material(self):
        self.load_file()
        if not self.all_file_exists:
            # self.clear_labels()
            print('****    Empty. Put material.    *****')
            return
        
        valid_pixels = len([True for v in self.acc if v > self.accuracy])
        # print("NUMBER OF VALID PIXELS IS {}".format(valid_pixels))
        # for v in self.acc:
        #     print(v, end='  ')
        if valid_pixels < 20:
            # self.clear_labels()
            if valid_pixels == 0:
                print('****    Put material.    *****')
            else:
                print('****    Measuring.    *****')
            return

        # REFORMAT THIS INTO WHAT WE NEED
        test_vec = np.vstack((self.d16, self.d80))
        # training = self.training

        # THIS SHOULD CALL
        # costs = [valid_l2_norm(test_vec, v) for v in self.training]
        # argmin = np.argmin(costs)

        # THIS SHOULD CALL returnRANKING function

        test_vec = replace_zeros_with_nan(calculate_(test_vec))
        # print("TYPE IS : " + str(type(test_vec)))
        # print('\n')
        # print('Test Vector:')
        # print(test_vec)
        numOfNan = test_vec.isna().sum().sum()
        global iteration

        if numOfNan < 1000:
            array = impute_test_vec(test_vec, "IterativeSVD")
            test_vec = pd.DataFrame(array)

            # Writing to excel file
            writer = pd.ExcelWriter('test_vector.xlsx', engine='openpyxl')
            test_vec.to_excel(writer, index=False)
            test_vec.to_excel(writer, startrow=iteration, index=False)
            writer.save()
            iteration+=3


            # print("FINALLLLLYYYYYY:")
            # print(test_vec)
            #
            # print(test_vec.iloc[:, 3350:3400])
            # print("Are there any NaN?")
            # print(test_vec.isna().sum().sum())
            with open('classifiers.pkl', 'rb') as input:
                classifiers = pickle.load(input)

            rankingLR = classifiers[0].predict(test_vec)
            rankingSVC = classifiers[1].predict(test_vec)
            rankingDT = classifiers[2].predict(test_vec)
            rankingKNN = classifiers[3].predict(test_vec)


            print('-' * 40)
            print("-----------CURRENT BEST PREDICTIONS------------")
            print("Logistic Regression: {}".format(rankingLR))
            print("SVC: {}".format(rankingSVC))
            print("Decision Tree: {}".format(rankingDT))
            print("KNN: {}".format(rankingKNN))
            print('-' * 40)


        else:
            print("Still long way to go... {}".format(numOfNan))



        # self.label.setText(self.materials[ranking])
        # self.mat2.setText(self.materials[ranking[1]])
        # self.mat3.setText(self.materials[ranking[2]])
        # self.mat4.setText(self.materials[ranking[3]])
        # self.mat5.setText(self.materials[ranking[4]])
        
def main(args):
    # app = QApplication(args)
    global app
    app = AppFormNect()
    input("Enter")

    # form.show()
    # sys.exit(app.exec_())


class Event(LoggingEventHandler):
    def __init__(self, application):
        self.application = application

    def on_modified(self, event):
        global app
        time.sleep(1)
        AppFormNect._on_file_changed(self.application)



if __name__ == "__main__":
    # print("----------- RETREIVING DATA ------------")
    #
    # # classifiers = main_f()
    #
    # print("----------- CLASSIFIERS TRAINED ------------")
    # start_kinect()
    main(sys.argv)

    input("Press ENTER to exit")

