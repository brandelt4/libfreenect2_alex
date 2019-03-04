from auto_invoke_demos import invoke_demo
from train_classifier import main_f

def main(com='bin/RelWithDebInfo/Protonect'):

    global classifiers
    print("----------- RETREIVING DATA ------------")
    # classifiers = main_f()

    print("----------- CLASSIFIERS TRAINED ------------")




    print("----------- TURNING ON KINECT ------------")
    invoke_demo()


def give_classifiers():
    return classifiers


if __name__ == "__main__":
    main()