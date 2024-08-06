from Loggers.Logger import Logger
from typing import Dict, Any
import logging
import json


class LocalLogger(Logger):
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._logger = logging.getLogger(self._file_path)

    def log(self, message: str):
        self._logger.info(message)

    def log(self, data: Dict[str, Any]):
        self._logger.info(json.dumps(data))

    def close(self):
        pass
