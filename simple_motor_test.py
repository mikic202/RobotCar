import gpiod
import time
LED_PIN = 14
chip = gpiod.Chip('gpiochip4')
motor_line = chip.get_line(LED_PIN)
motor_line.request(consumer="Motor", type=gpiod.LINE_REQ_DIR_OUT)

LED_PIN = 15
motor_line2 = chip.get_line(LED_PIN)
motor_line2.request(consumer="Motor2", type=gpiod.LINE_REQ_DIR_OUT)
try:
   while True:
       motor_line.set_value(1)
       motor_line2.set_value(0)
       time.sleep(1)
       motor_line.set_value(0)
       motor_line2.set_value(1)
       time.sleep(1)
finally:
   motor_line.set_value(0)
   motor_line2.set_value(0)
   motor_line.release()
   motor_line2.release()