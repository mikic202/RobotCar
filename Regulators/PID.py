from typing import List
import numpy as np


class PID:
    def __init__(self, K: float, Ti: float, Td: float, Tp: float, number_of_inputs: int, setpoints: List[float]) -> None:
        self.K = K
        self.Ti = Ti
        self.Td = Td
        self.Tp = Tp
        self._r_0 = K * (1 + Tp / (2 * Ti) + Td / Tp)
        self._r_1 = K * (Tp / (2 * Ti) - 2 * Td / Tp - 1)
        self._r_2 = K * Td / Tp
        self._e = [np.zeros(number_of_inputs) for _ in range(4)]
        self._stp = np.array(setpoints)
        self._u = 0

    def get_controll(self, input: List[float]):
        self._e = [self._stp - np.array(input), self._e[0], self._e[1], self._e[2]]
        self._u = (
            self._r_0 * (self._e[1] - self._e[0])
            + self._r_1 * (self._e[2] - self._e[1])
            + self._r_2 * (self._e[3] - self._e[2])
            + self._u
        )
        return self._u
