from HardwareClasses.Motor import Motor
from multiprocessing import Manager


class MotorDrive:
    def __init__(self, motors: list[Motor]) -> None:
        self._motors = motors
        self._pwms = Manager().list()
        for motor in self._motors:
            motor.reset()

    def set_pwms(self, pwms: list[float]) -> None:
        self._pwms[:] = pwms
        for motor, pwm in zip(self._motors, pwms):
            motor.set_pwm(pwm)

    def get_pwms(self) -> list[float]:
        return self._pwms
