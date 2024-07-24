from HardwareClasses.SensorArray import SensorArray
from Loggers.RemoteLogger import RemoteLogger
from HardwareClasses.MotorDrive import MotorDrive
from Robot.Robot import Robot
from HardwareClasses.Motor import Motor
from Regulators.RemoteRegulator import RemoteRegulator
from Regulators.PID import PID
from Loggers.LocalLogger import LocalLogger
import time


if __name__ == "__main__":
    try:
        motor_drive = MotorDrive([Motor(14, 15), Motor(23, 24)])
        array = SensorArray([(5, -90), (25, -45), (12, 0), (1, 45), (21, 90)])
        print("Sensors established")
        # sensor_logger = RemoteLogger("192.168.0.164", 65432)
        # time.sleep(0.5)
        # controll_logger = RemoteLogger("192.168.0.164", 65433)
        sensor_logger = LocalLogger("sensor.txt")
        time.sleep(0.5)
        controll_logger = LocalLogger("contr.txt")
        print("Loggers established")
        remote_regulator = PID(1, 1, 1, 0.5, 5, [0, 0, 0, 0, 0])
        robot = Robot(motor_drive, array, sensor_logger, controll_logger, remote_regulator)
        robot()
    finally:
        array.reset_addresses()
    array.reset_addresses()
