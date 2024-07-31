from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.SensorArray import SensorArray
from Loggers.Logger import Logger
from Regulators.Regulator import Regulator
from Robot.Robot import Robot

class DifferentialRobot(Robot):
    def __init__(self, motors_drive: MotorDrive, sensor_array: SensorArray, sensor_logger: Logger, motor_logger: Logger, regulator: Regulator):
        super().__init__(motors_drive, sensor_array, sensor_logger, motor_logger, regulator)

    def _apply_new_controls(self):
        sonstant_value = 0.4
        controll = self._regulator.get_controll([self._sensor_array()[0][0]]) /10000
        print(f"controll: {[sonstant_value + controll, sonstant_value - controll]}")
        self._motor_drive.set_pwms([sonstant_value + controll, sonstant_value - controll])