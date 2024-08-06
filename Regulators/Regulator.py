from abc import ABC, abstractmethod
from typing import List


class Regulator(ABC):
    @abstractmethod
    def get_controll(self, input: List[float]) -> List[float]:
        pass
