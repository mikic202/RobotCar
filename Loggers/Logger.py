from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Logger(ABC):
    def __init__(self):
        pass

    def __enter__(self):
        return self

    @abstractmethod
    def log(self, data: List[Dict[str, Any]], sample_index: int):
        pass

    @abstractmethod
    def close(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()