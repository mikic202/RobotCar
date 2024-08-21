from Loggers.Logger import Logger
import csv
from typing import List, Any

class CsvLogger(Logger):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w')
        self.writer = csv.writer(self.file)

    def log(self, log_values: List[Any]):
        self.writer.writerow(log_values)

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()