from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Logger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log(self, message: str):
        pass

    @abstractmethod
    def log(self, data: List[Dict[str, Any]], sample_index: int):
        pass

    @abstractmethod
    def close(self):
        pass
