from typing import List
from HardwareClasses.Motor import Motor


class MotorDrive:
    def __init__(self, motors: List[Motor]) -> None:
        self._motors = motors
        for motor in self._motors:
            motor.reset()

    def set_pwms(self, pwms: List[float]):
        for motor, pwm in zip(self._motors, pwms):
            motor.set_pwm(pwm)

    def get_pwms(self):
        return [motor.get_pwm() for motor in self._motors]
