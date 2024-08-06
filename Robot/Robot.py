from abc import ABC, abstractmethod
from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Timers.Timer import Timer
from time import sleep
from threading import Thread
import json


def convert_sensor_data_to_dict(data):
    return [{"angle": reading[1], "value": reading[0]} for reading in data]


def convert_motor_data_to_dict(data):
    return [{"control_name": motor, "value": value} for motor, value in enumerate(data)]


class Robot(ABC):
    def __init__(
        self,
        motors_drive: MotorDrive,
        sensor_array: SensorArray,
        sensor_logger: Logger,
        motor_logger: Logger,
        regulator: Regulator,
        controll_loop_timer: Timer,
    ):
        self._motor_drive = motors_drive
        self._sensor_array = sensor_array
        self._sensor_logger = sensor_logger
        self._motor_logger = motor_logger
        self._regulator = regulator
        self._controll_loop_timer = controll_loop_timer

    def log_sensor_data(self):
        self._sensor_logger.log(
            json.dumps(
                convert_sensor_data_to_dict(self._sensor_array.get_latest_data())
            )
        )

    def log_motor_data(self):
        self._motor_logger.log(
            json.dumps(convert_motor_data_to_dict(self._motor_drive.get_pwms()))
        )

    def _start_loggers(self):
        try:
            while True:
                self.log_sensor_data()
                self.log_motor_data()
                sleep(0.5)
        finally:
            self._sensor_logger.close()
            self._motor_logger.close()

    def _run(self):
        try:
            while True:
                if self._controll_loop_timer():
                    self._apply_new_controls()
                sleep(0.01)
        finally:
            self._motor_drive.set_pwms([0] * len(self._motor_drive.get_pwms()))
            self._sensor_array.reset_addresses()

    def __call__(self):
        Thread(target=self._start_loggers).start()
        self._run()

    @abstractmethod
    def _apply_new_controls(self):
        pass
