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
        constant_value = 0.5
        with self._lock:
            array_values = self._sensor_array()
        dist_diff = array_values[0][0] - array_values[-1][0]
        control = self._regulator.get_control([dist_diff / 1000])
        self._motor_drive.set_pwms([constant_value + control, constant_value - control])
