from typing import List
from HardwareClasses.Motor import Motor
from multiprocessing import Manager


class MotorDrive:
    def __init__(self, motors: List[Motor]) -> None:
        self._motors = motors
        self._pwms = Manager().list()
        for motor in self._motors:
            motor.reset()

    def set_pwms(self, pwms: List[float]):
        self._pwms[:] = pwms
        for motor, pwm in zip(self._motors, pwms):
            motor.set_pwm(pwm)

    def get_pwms(self):
        return self._pwms
