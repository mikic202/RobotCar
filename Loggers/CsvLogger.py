from Loggers.Logger import Logger
import csv
from typing import Dict, Any, List
from Parser.RobotDataParser import RobotDataParser


class CsvLogger(Logger):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        self.writer = csv.writer(self.file)

    def log(self, log_values: List[Dict[str, Any]], sample_index: int):
        self.writer.writerows(
            RobotDataParser.convert_data_dict_to_csv_rovs(log_values, sample_index)
        )

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()
