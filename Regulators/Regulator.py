from abc import ABC, abstractmethod


class Regulator(ABC):
    @abstractmethod
    def get_control(self, input: list[float]) -> list[float]:
        pass
