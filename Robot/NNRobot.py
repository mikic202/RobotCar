from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Robot.Robot import Robot


class NNRobot(Robot):
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
        with self._lock:
            array_values = self._sensor_array()
        controll = self._regulator.get_controll(
            [array_values[0][0], array_values[1][0], array_values[2][0], array_values[3][0], array_values[4][0]]
        )
        self._motor_drive.set_pwms(controll[0])
