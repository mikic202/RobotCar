from Regulators.Regulator import Regulator
from Regulators.FuzzyPID import FuzzyPID


class MultipleFuzzyPID(Regulator):
    def __init__(self, init_files):
        super().__init__()
        self._fuzzy_pids = [FuzzyPID(filname) for filname in init_files]

    def get_controll(self, inputs: list[float]):
        return [fuzzy_pid(input) for fuzzy_pid, input in zip(self._fuzzy_pids, inputs)]