#!/usr/bin/env python

import sys
sys.path.append('~/libfreenect2_alex/build')
import sys
import os
import time
import subprocess



def invoke_demo(com='bin/RelWithDebInfo/Protonect'):
    p = subprocess.Popen([com, 'cpu']) # running background.
    time.sleep(3)
    # p3 = subprocess.Popen(['python', 'real_time_plot_acc.py'])
    # time.sleep(1)

    # Train the classifiers
    # p1 = subprocess.Popen(['python', 'train_classifier.py'])

    # Open real-time classification
    p2 = subprocess.Popen(['python', 'real_time_nn.py'])

    p_stdout = p.communicate()[0]
    p2.terminate()
    p3.terminate()



if __name__ == "__main__":
    invoke_demo()
     
