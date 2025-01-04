from Loggers.Logger import Logger
from typing import Any


class MultiLogger(Logger):
    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = loggers

    def log(self, log_values: list[dict[str, Any]], sample_index: int) -> None:
        for logger in self._loggers:
            logger.log(log_values, sample_index)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()

    def __del__(self) -> None:
        for logger in self._loggers:
            logger.close()
