import argparse
from Loggers.RemoteLogger import RemoteLogger
from Loggers.LocalLogger import LocalLogger
from Loggers.CsvLogger import CsvLogger
from Loggers.Logger import Logger
from Timers.Timer import Timer
from Regulators.Regulator import Regulator
from Regulators.RemoteRegulator import RemoteRegulator
from Regulators.DMC import DMC
from Robot.DifferentialRobot import DifferentialRobot
from Robot.HumanControlledRobot import HumanControlledRobot
from Robot.NNRobot import NNRobot
from Robot.ReversedTwoRegRobot import ReversedTwoRegRobot
from Robot.Robot import Robot
from Regulators.PID import PID
from Regulators.FuzzyPID import FuzzyPID
from Regulators.NeuralNetworkRegulator import NeuralNetworkRegulator
from HardwareClasses.SensorArray import SensorArray
from HardwareClasses.MotorDrive import MotorDrive
from HardwareClasses.Motor import Motor

def parse_args():
    parser = argparse.ArgumentParser(description="Parse robot arguments")

    parser.add_argument(
        "--logger",
        choices=["l", "r", "c"],
        help="Type of logger used by robot",
        default="l",
    )
    parser.add_argument(
        "--Tp", type=float, help="Timer period for regulation loop", default=0.5
    )
    parser.add_argument(
        "--robot",
        choices=["diff", "human", "NN", "two_reg", "fuzzy_diff"],
        help="Type of robot used",
        default="diff",
    )
    parser.add_argument(
        "--regulator",
        choices=["PID", "human", "DMC", "NN"],
        help="Type of regulator used by robot",
        default="PID",
    )
    parser.add_argument(
        "--reg_args", metavar="N", type=str, nargs="*", help="Arguments for regulator"
    )

    parser.add_argument(
        "--log_location", type=str, help="Location where data will be logged", default="log"
    )

    return parser.parse_args()


def init_robot_from_args() -> Robot:
    args = parse_args()

    sensor_logger: Logger
    control_logger: Logger
    if args.logger == "r":
        sensor_logger = RemoteLogger()
        control_logger = RemoteLogger()
    elif args.logger == "c":
        sensor_logger = CsvLogger(f"{args.log_location}/sensor.csv")
        control_logger = CsvLogger(f"{args.log_location}/control.csv")
    else:
        sensor_logger = LocalLogger(f"{args.log_location}/sensor.log")
        control_logger = LocalLogger(f"{args.log_location}/control.log")

    timer = Timer(args.Tp)
    motor_drive = MotorDrive([Motor(14, 15), Motor(23, 24)])


    if args.robot == "diff":
        return DifferentialRobot(
            motor_drive,
            array,
            sensor_logger,
            control_logger,
            PID(*[[float(arg)] for arg in args.reg_args], args.Tp, 1, [0]),
            timer,
        )
    elif args.robot == "fuzzy_diff":
        reg = FuzzyPID(args.reg_args[0], Tp=args.Tp)
        array = SensorArray([(5, -90), (25, -45), (12, 0), (1, 45), (21, 90)])
        return DifferentialRobot(
            motor_drive,
            array,
            sensor_logger,
            control_logger,
            reg,
            timer,
        )
    elif args.robot == "human":
        return HumanControlledRobot(
            motor_drive, array, sensor_logger, control_logger, RemoteRegulator(), timer
        )
    elif args.robot == "NN":
        return NNRobot(
            motor_drive,
            array,
            sensor_logger,
            control_logger,
            NeuralNetworkRegulator(f"Regulators/{args.reg_args[0]}"),
            timer,
        )
    elif args.robot == "two_reg":
        return ReversedTwoRegRobot(
            motor_drive,
            array,
            sensor_logger,
            control_logger,
            PID(*[[float(arg), float(arg)] for arg in args.reg_args], args.Tp, 2, [0, 0]),
            timer,
        )
