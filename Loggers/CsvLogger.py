from Loggers.Logger import Logger
import csv
from typing import Any
from Parser.RobotDataParser import RobotDataParser


class CsvLogger(Logger):
    def __init__(self, filename) -> None:
        self.filename = filename
        self.file = open(filename, "w")
        self.writer = csv.writer(self.file)

    def log(self, log_values: list[dict[str, Any]], sample_index: int) -> None:
        self.writer.writerows(
            RobotDataParser.convert_data_dict_to_csv_rovs(log_values, sample_index)
        )

    def close(self) -> None:
        self.file.close()

    def __del__(self) -> None:
        self.close()
