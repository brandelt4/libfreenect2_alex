# from auto_invoke_demos import invoke_demo
from preprocessing import main_f
from training import main_f
import subprocess
import os

global classifiers


def _main(com='bin/RelWithDebInfo/Protonect'):

    invoke_demo()


def give_classifiers():
    global classifiers
    return classifiers


if __name__ == "__main__":

    # If files already exist, delete them
    try:
        os.remove('accumurate_depth.dat')
        os.remove('depth_bins.dat')
        os.remove('phase_depth_0.dat')
        os.remove('phase_depth_1.dat')
        os.remove('phase_depth_2.dat')
    except:
        print('Files were not removed, because no files found.')

    p3 = subprocess.Popen(['python', '-i', 'arduino2.py'])
    p_stdout = p3.communicate()[0]
