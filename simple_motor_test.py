import gpiod
import time
from HardwareClasses.Motor import Motor

motor = Motor(14, 15)

try:
   while True:
       motor.set_pwm(0.7)
       time.sleep(1)
       motor.set_pwm(0)
       time.sleep(1)
       motor.set_pwm(-0.70)
       time.sleep(1)
finally:
   motor.reset()