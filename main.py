from Parser.ArgParser import init_robot_from_args, parser


if __name__ == "__main__":
    try:
        robot = init_robot_from_args(parser.parse_args())
        robot()
    finally:
        robot._sensor_array.reset_addresses()
    robot._sensor_array.reset_addresses()
