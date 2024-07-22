from abc import ABC, abstractmethod
from typing import Dict, Any


class Logger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log(self, message: str):
        pass

    @abstractmethod
    def log(self, data: Dict[str, Any]):
        pass

    @abstractmethod
    def close(self):
        pass
