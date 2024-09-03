from Parser.ArgParser import init_robot_from_args, parser


if __name__ == "__main__":
    robot = init_robot_from_args(parser.parse_args())
    robot()
