from abc import ABC, abstractmethod
from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Parser.RobotDataParser import RobotDataParser
from Timers.Timer import Timer
from time import sleep
from multiprocessing import Process, Lock
import copy


class Robot(ABC):
    def __init__(
        self,
        motors_drive: MotorDrive,
        sensor_array: SensorArray,
        sensor_logger: Logger,
        motor_logger: Logger,
        regulator: Regulator,
        control_loop_timer: Timer,
    ):
        self._motor_drive = motors_drive
        self._sensor_array = sensor_array
        self._sensor_logger = sensor_logger
        self._motor_logger = motor_logger
        self._regulator = regulator
        self._control_loop_timer = control_loop_timer
        self._log_loop_timer = copy.deepcopy(control_loop_timer)
        self._lock = Lock()

    def log_sensor_data(self, iteration: int):
        with self._lock:
            sensor_data = self._sensor_array.get_latest_data()[:]
        self._sensor_logger.log(
            RobotDataParser.convert_sensor_data_to_dict(sensor_data), iteration
        )

    def log_motor_data(self, iteration: int):
        with self._lock:
            control_data = self._motor_drive.get_pwms()[:]
        self._motor_logger.log(
            RobotDataParser.convert_motor_data_to_dict(control_data), iteration
        )

    def _start_loggers(self):
        logger_iteration = 0
        sleep(0.1)
        try:
            while True:
                if self._log_loop_timer():
                    self.log_sensor_data(logger_iteration)
                    self.log_motor_data(logger_iteration)
                    logger_iteration += 1
        finally:
            self._sensor_logger.close()
            self._motor_logger.close()

    def _run(self):
        try:
            while True:
                if self._control_loop_timer():
                    self._apply_new_controls()
                sleep(0.01)
        finally:
            self._motor_drive.set_pwms([0] * len(self._motor_drive.get_pwms()))
            self._sensor_array.reset_addresses()

    def __call__(self):
        Process(target=self._start_loggers).start()
        self._run()

    @abstractmethod
    def _apply_new_controls(self):
        pass
