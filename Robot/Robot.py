from abc import ABC, abstractmethod
from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator


def convert_sensor_data_to_dict(data):
    return [{"angle": reading[1], "value": reading[0]}for reading in data]


class Robot(ABC):
    def __init__(self, motors_drive: MotorDrive, sensor_array: SensorArray, sensor_logger: Logger, motor_logger: Logger, regulator: Regulator):
        self._motor_drive = motors_drive
        self._sensor_array = sensor_array
        self._sensor_logger = sensor_logger
        self._motor_logger = motor_logger
        self._regulator = regulator

    def log_sensor_data(self):
        self._sensor_logger.log(convert_sensor_data_to_dict(self._sensor_array.get_latest_data()))

    def log_motor_data(self):
        pass

    @abstractmethod
    def __call__(self):
        pass