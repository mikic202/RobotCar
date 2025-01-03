from multiprocessing import Process
from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Robot.Robot import Robot


class HumanControlledRobot(Robot):
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

    def _read_sensor_data(self) -> None:
        with self._lock:
            self._sensor_array()

    def _apply_new_controls(self) -> None:
        Process(target=self._read_sensor_data).start()
        self._motor_drive.set_pwms(self._regulator.get_control([]))
