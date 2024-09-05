from Loggers.Logger import Logger
import csv
from typing import Dict, Any, List


class CsvLogger(Logger):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        self.writer = csv.writer(self.file)

    def log(self, log_values: List[Dict[str, Any]], sample_index: int):
        self.writer.writerows(
            [
                [*[value for value in reading.values()], sample_index]
                for reading in log_values
            ]
        )

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()
