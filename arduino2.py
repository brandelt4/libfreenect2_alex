import serial
import time
from auto_invoke_demos import classify
import subprocess
import os
DETACHED_PROCESS = 0x00000008

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

# port = '/dev/cu.usbmodemFD121'
port = 'COM4'

arduinoSerialData = serial.Serial(port, 9600)



while True:
    if arduinoSerialData.in_waiting > 0:
        mySignal = arduinoSerialData.readline()
        mySignal = mySignal.decode('utf-8')
        print(mySignal)



        if 'k_on' in mySignal:
            print("Received: k_on")
            p = subprocess.Popen(['python', '-i', 'start_kinect.py'], creationflags=DETACHED_PROCESS).pid
            # p_stdout = p.communicate()[0]





        elif 'class' in mySignal:
            print("Received: class")
            p2 = subprocess.Popen(['python', '-i', 'classify.py'])
            p2_stdout = p2.communicate()[0]


        continue