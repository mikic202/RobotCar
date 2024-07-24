from typing import List


class PID:
    def __init__(self, K: float, Ti: float, Td: float, Tp: float) -> None:
        self.K = K
        self.Ti = Ti
        self.Td = Td
        self.Tp = Tp
        self._r_0 = K * (1 + Tp / (2 * Ti) + Td / Tp)
        self._r_1 = K * (Tp / (2 * Ti) - 2 * Td / Tp - 1)
        self._r_2 = K * Td / Tp
        self._y = [0, 0, 0, 0]
        self._u = 0

    def get_controll(self, input: List[float]):
        self._y = [input, self._y[0], self._y[1], self._y[2]]
        self._u = (
            self._r_0 * (self._y[1] - self._y[0])
            + self._r_1 * (self._y[2] - self._y[1])
            + self._r_2 * (self._y[3] - self._y[2])
            + self._u
        )
        return self._u
