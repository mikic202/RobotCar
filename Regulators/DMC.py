from Regulators.Regulator import Regulator

class DMC(Regulator):
    def __init__(self, D, N, Nu, lambda_, step_response):
        self._D = D
        self._N = N
        self._Nu = Nu
        self._lambda = lambda_
        self._step_response = step_response