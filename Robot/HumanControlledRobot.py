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

    def _read_sensor_data(self):
        self._sensor_array()

    def _apply_new_controls(self):
        Process(target = self._read_sensor_data).start()
        self._motor_drive.set_pwms(self._regulator.get_controll([]))