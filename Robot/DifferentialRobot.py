from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Robot.Robot import Robot


class DifferentialRobot(Robot):
    def __init__(
        self,
        motors_drive: MotorDrive,
        sensor_array: SensorArray,
        sensor_logger: Logger,
        motor_logger: Logger,
        regulator: Regulator,
        controll_loop_timer,
    ):
        super().__init__(
            motors_drive,
            sensor_array,
            sensor_logger,
            motor_logger,
            regulator,
            controll_loop_timer,
        )

    def _apply_new_controls(self):
        constant_value = 0.4
        controll = self._regulator.get_controll([self._sensor_array()[0][0]]) / 10000
        print(f"controll: {[constant_value + controll, constant_value - controll]}")
        self._motor_drive.set_pwms(
            [constant_value + controll, constant_value - controll]
        )
