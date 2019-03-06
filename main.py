from auto_invoke_demos import invoke_demo
from train_classifier import main_f
from training import main_f
import subprocess

global classifiers


def _main(com='bin/RelWithDebInfo/Protonect'):

    # global classifiers
    # classifiers = main_f()

    # print("----------- CLASSIFIERS TRAINED ------------")
    print("----------- RETREIVING DATA ------------")

    # main_f()

    print("----------- CLASSIFIERS TRAINED ------------")



    print("----------- TURNING ON KINECT ------------")
    invoke_demo()


def give_classifiers():
    global classifiers
    return classifiers


if __name__ == "__main__":
    # print("----------- RETREIVING DATA ------------")

    p3 = subprocess.Popen(['python', '-i', 'arduino2.py'])