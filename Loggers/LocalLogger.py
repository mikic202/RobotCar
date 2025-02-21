from Loggers.Logger import Logger
from typing import Dict, Any
import logging
import json


class LocalLogger(Logger):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._logger = logging.getLogger(self._file_path)

    def log(self, data: list[dict[str, Any]], sample_index: int) -> None:
        self._logger.info(json.dumps(data))

    def close(self) -> None:
        pass
