from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Robot.Robot import Robot


class TwoRegRobot(Robot):
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
        # lock causes some instabilities in how often the controll is aplied
        with self._lock:
            array_values = self._sensor_array()
        dist_diff = array_values[0][0] - array_values[-1][0]
        input_variables = [array_values[0][0] / 1000, array_values[-1][0] / 1000]
        self._motor_drive.set_pwms(self._regulator.get_controll(input_variables))
