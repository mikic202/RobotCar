from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    def __init__(self) -> None:
        pass

    def __enter__(self):
        return self

    @abstractmethod
    def log(self, data: list[dict[str, Any]], sample_index: int) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
