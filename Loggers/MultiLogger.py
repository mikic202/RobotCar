from Loggers.Logger import Logger

class MultiLogger(Logger):
    def __init__(self, loggers: list[Logger]):
        self._loggers = loggers

    def log(self, log_values: List[Dict[str, Any]], sample_index: int):
        for logger in self._loggers:
            logger.log(log_values, sample_index)

    def close(self):
        for logger in self._loggers:
            logger.close()

    def __del__(self):
        for logger in self._loggers:
            logger.close()