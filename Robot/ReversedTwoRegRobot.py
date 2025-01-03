from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Robot.Robot import Robot


class ReversedTwoRegRobot(Robot):
    def __init__(
        self,
        motors_drive: MotorDrive,
        sensor_array: SensorArray,
        sensor_logger: Logger,
        motor_logger: Logger,
        regulator: Regulator,
        control_loop_timer,
    ) -> None:
        super().__init__(
            motors_drive,
            sensor_array,
            sensor_logger,
            motor_logger,
            regulator,
            control_loop_timer,
        )

    def _apply_new_controls(self) -> None:
        with self._lock:
            array_values = self._sensor_array()
        dist_diff = array_values[0][0] - array_values[-1][0]
        input_variables = [array_values[0][0] / 1000, array_values[-1][0] / 1000]
        control = [
            value + 0.4 for value in self._regulator.get_control(input_variables)
        ]
        self._motor_drive.set_pwms(control)
