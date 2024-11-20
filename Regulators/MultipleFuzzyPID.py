from Regulators.Regulator import Regulator
from Regulators.FuzzyPID import FuzzyPID


class MultipleFuzzyPID(Regulator):
    def __init__(self, init_files, Tp):
        super().__init__()
        self._Tp = Tp
        self._fuzzy_pids = [FuzzyPID(filname, Tp) for filname in init_files]

    def get_controll(self, inputs: list[float]):
        return [fuzzy_pid.get_controll([input]) for fuzzy_pid, input in zip(self._fuzzy_pids, inputs)]