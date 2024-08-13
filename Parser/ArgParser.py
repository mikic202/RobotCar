import argparse
from Loggers.RemoteLogger import RemoteLogger
from Loggers.LocalLogger import LocalLogger
from Loggers.Logger import Logger
from Timers.Timer import Timer
from Regulators.Regulator import Regulator
from Regulators.RemoteRegulator import RemoteRegulator
from Regulators.DMC import DMC
from Robot.DifferentialRobot import DifferentialRobot
from Robot.Robot import Robot
from Regulators.PID import PID
from HardwareClasses.SensorArray import SensorArray
from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.Motor import Motor

parser = argparse.ArgumentParser(description='Parse robot arguments')

parser.add_argument('--logger', choices=['l', 'r'], help="Type of logger used by robot", default='l')
parser.add_argument('--Tp', type=float, help="Timer period for regulation loop", default=0.5)
parser.add_argument('--robot', choices=['diff', 'sing'], help="Type of robot used", default='diff')
parser.add_argument('--regulator', choices=['PID', 'human', 'DMC', 'NN'], help="Type of regulator used by robot", default='PID')
parser.add_argument('--reg_args', metavar='N', type=float, nargs='*', help='Arguments for regulator')


def init_robot_from_args(args: argparse.Namespace) -> Robot:
    motor_drive = MotorDrive([Motor(14, 15), Motor(23, 24)])
    array = SensorArray([(5, -90), (25, -45), (12, 0), (1, 45), (21, 90)])

    sensor_logger: Logger
    control_logger: Logger
    if args.logger == 'r':
        sensor_logger = RemoteLogger()
        control_logger = RemoteLogger()
    else:
        sensor_logger = LocalLogger("sensor.log")
        control_logger = LocalLogger("control.log")

    timer = Timer(args.Tp)

    regulator: Regulator

    if args.regulator == 'PID':
        regulator = PID(*args.reg_args, args.Tp, 1, [150])
    elif args.regulator == 'human':
        regulator = RemoteRegulator()
    elif args.regulator == 'DMC':
        regulator = DMC(*args.reg_args, args.Tp, 1)

    if args.robot == 'diff':
        return DifferentialRobot(motor_drive, array, sensor_logger, control_logger, regulator, timer)

