from pyfirmata import Arduino, util
import time

board = Arduino('device manager->ports->COM3')

iterator = util.Iterator(board)
iterator.start()

# Read voltage of pin 0 analogue
Tv1 = board.get_pin('a:0:i')

# Digital pin
D1 = board.get_pin('d:3:p')
D1.write(1)

time.sleep(30)



time.sleep(1)


print(Tv1.read())