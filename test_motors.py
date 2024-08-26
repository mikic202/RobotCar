from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.Motor import Motor
from time import sleep

if __name__ == "__main__":
    motor_drive = MotorDrive([Motor(14, 15), Motor(23, 24)])
    for pwm in range(10):
        sleep(0.5)
        print(f"Setting left pwm to {pwm/10}")
        motor_drive.set_pwms([0, -pwm/10])
    for pwm in range(10):
        sleep(0.5)
        print(f"Setting right pwm to {pwm/10}")
        motor_drive.set_pwms([-pwm/10, 0])