from Regulators.MembershipFunctions.MembershipFunction import MembershipFunction
import numpy as np


class TrapeizodalMembershipFunction(MembershipFunction):
    def __init__(self, params: list[float]) -> None:
        self._params = params
        self._positive_slope_coefficients = np.polyfit([params[0], params[1]], [0, 1], 1)
        self._negative_slope_coefficients = np.polyfit([params[2], params[3]], [1, 0], 1)

    def __call__(self, input: float) -> float:
        if input <= self._params[0] or input >= self._params[-1]:
            return 0
        if input >= self._params[1] and input <= self._params[2]:
            return input
        if input < self._params[1]:
            return np.polyval(self._positive_slope_coefficients, input)
        return np.polyval(self._negative_slope_coefficients, input)