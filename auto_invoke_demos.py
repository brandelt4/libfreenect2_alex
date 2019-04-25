#!/usr/bin/env python

import sys
sys.path.append('~/libfreenect2_alex/build')
import sys
import os
import time
import subprocess



# def start_kinect(com='bin/RelWithDebInfo/Protonect'):
#     p = subprocess.Popen([com, 'cpu']) # running background.
#     # time.sleep(3)
#     # p3 = subprocess.Popen(['python', 'real_time_plot_acc.py'])
#     # time.sleep(1)
#
#     # Train the classifiers
#     # p1 = subprocess.Popen(['python', 'preprocessing.py'])
#
#     # Open real-time classification
#     # p2 = subprocess.Popen(['python', '-i', 'classification_main.py'])
#
#     p_stdout = p.communicate()[0]
#     # p2.terminate()
#     # p3.terminate()

def classify():
    p2 = subprocess.Popen(['python', '-i', 'classification_main.py'])



if __name__ == "__main__":
    pass
     
