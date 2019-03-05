import serial
import time
from main import _main


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

port = '/dev/cu.usbmodemFD121'

arduinoSerialData = serial.Serial(port, 9600)



while True:

    if arduinoSerialData.in_waiting > 0:
        mySignal = arduinoSerialData.readline()



        if mySignal == 'k_on':
            print("starting kinect...")
            _main()


        elif mySignal == 'classify':
            pass
