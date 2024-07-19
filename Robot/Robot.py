from abc import ABC, abstractmethod
from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray

class Robot(ABC):
    def __init__(self, motors_drive: MotorDrive, sensor_array: SensorArray):
        self._motor_drive = motors_drive
        self._sensor_array = sensor_array

    @abstractmethod
    def __call__(self):
        pass