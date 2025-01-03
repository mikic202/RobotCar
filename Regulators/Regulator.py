from abc import ABC, abstractmethod
from typing import List


class Regulator(ABC):
    @abstractmethod
    def get_control(self, input: List[float]) -> List[float]:
        pass
