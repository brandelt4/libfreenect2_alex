import serial
import time
from auto_invoke_demos import classify
import subprocess
import os
import window
# from window import w

# import tkinter
# windowCreated = False
# try: mainWindow
# except NameError: mainWindow = None
# if mainWindow is None:
#     mainWindow = tkinter.Tk()
#     windowCreated = True
#
# v = tkinter.StringVar()

DETACHED_PROCESS = 0x00000008
DETACHED_PROCESS2 = 0x00000009
DETACHED_PROCESS3 = 0x00000007



"""
1. // Yellow button
2. A: void move_one()
3. // Green button
4. A: void move_all()
5. A: void send_kinect()
6. A: void start_platform()
7. Py: collect_data()
8. A: void send_classify()
9. Py: kill_kinect_data_collection()
10. Py: classify()
11. A: void put_the_item_in_the_right_slot()
12. RESET

"""

# port = '/dev/cu.
ports = []


# class Window():
#
#     def __init__(self):
#         global label2
#         v.set('nothing...')
#         label = tkinter.Label(mainWindow, text='Current Activity: ')
#         label.pack()
#         label2 = tkinter.Label(mainWindow, textvariable=v)
#         label2.pack()
#         mainWindow.mainloop()
#
#
#     def changeActivity(self, mess):
#         global v
#         v.set(mess)


def send_plastic():
    arduinoSerialData = serial.Serial('COM4', 9600)
    arduinoSerialData.write("rsdl")

# def createWindow():
#     global label2
#     v.set('nothing...')
#     label = tkinter.Label(mainWindow, text='Current Activity: ')
#     label.pack()
#     label2 = tkinter.Label(mainWindow, textvariable=v)
#     label2.pack()
#     mainWindow.mainloop()


# def changeActivity(message):
#     global v
#     v.set(message)


if __name__ == '__main__':

    # Starting the main window
    # p3 = subprocess.Popen(['python', 'window.py'], creationflags=DETACHED_PROCESS).pid

    port = 'COM4'
    arduinoSerialData = serial.Serial(port, 9600)
    ports.append(arduinoSerialData)


    while True:

        if arduinoSerialData.in_waiting > 0:
            mySignal = arduinoSerialData.readline()
            mySignal = mySignal.decode('utf-8')
            print(mySignal)

            if 'k_on' in mySignal:
                print("Received: k_on")
                window.w.changeActivity('Received: k_on')
                p = subprocess.Popen(['python', '-i', 'start_kinect.py'], creationflags=DETACHED_PROCESS).pid
                # p_stdout = p.communicate()[0]

            elif 'class' in mySignal:
                print("Received: class")
                # window.Window('Received: class')
                window.w.changeActivity('Received: class')

                p2 = subprocess.Popen(['python', '-i', 'classify.py'], creationflags=DETACHED_PROCESS2).pid
                # p2_stdout = p2.communicate()[0]

            elif 'hitme' in mySignal:
                print("Received: hit me")
                # window.Window('Received: hit me')
                window.w.changeActivity('Received: hit me')

                print("Replying...")
                arduinoSerialData.write(2)


