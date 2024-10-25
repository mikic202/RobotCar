from abc import ABC, abstractmethod


class MembershipFunction(ABC):
    @abstractmethod
    def __call__(self, input: float) -> float:
        pass
