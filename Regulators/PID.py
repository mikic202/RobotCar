from Regulators.Regulator import Regulator
import numpy as np


class PID(Regulator):
    def __init__(
        self,
        K: list[float],
        Ti: list[float],
        Td: list[float],
        Tp: float,
        number_of_inputs: int,
        setpoints: list[float],
    ) -> None:
        self.K = np.array(K)
        self.Ti = np.array(Ti)
        self.Td = np.array(Td)
        self.Tp = Tp
        self._calculate_r_values()
        self._e = [np.zeros(number_of_inputs) for _ in range(4)]
        self._setpoints = np.array(setpoints)
        self._u = np.zeros(number_of_inputs)

    def get_control(self, input: list[float]) -> list[float]:
        self._e = [
            self._setpoints - np.array(input),
            self._e[0],
            self._e[1],
            self._e[2],
        ]
        self._u = (
            self._r_0 * (self._e[1] - self._e[0])
            + self._r_1 * (self._e[2] - self._e[1])
            + self._r_2 * (self._e[3] - self._e[2])
            + self._u
        )
        return self._u

    def set_Tp(self, Tp: float):
        self.Tp = Tp
        self._calculate_r_values()

    def set_K(self, K: list[float]):
        self.K = np.array(K)
        self._calculate_r_values()

    def set_Ti(self, Ti: list[float]):
        self.Ti = np.array(Ti)
        self._calculate_r_values()

    def set_Td(self, Td: list[float]):
        self.Td = np.array(Td)
        self._calculate_r_values()

    def set_setpoints(self, setpoints: list[float]):
        self._setpoints = np.array(setpoints)

    def _calculate_r_values(self):
        self._r_0 = self.K * (1 + self.Tp / (2 * self.Ti) + self.Td / self.Tp)
        self._r_1 = self.K * (self.Tp / (2 * self.Ti) - 2 * self.Td / self.Tp - 1)
        self._r_2 = self.K * self.Td / self.Tp
