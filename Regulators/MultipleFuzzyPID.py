from Regulators.Regulator import Regulator
from Regulators.FuzzyPID import FuzzyPID


class MultipleFuzzyPID(Regulator):
    def __init__(self, init_files, Tp) -> None:
        super().__init__()
        self._Tp = Tp
        self._fuzzy_pids = [FuzzyPID(filname, Tp) for filname in init_files]

    def get_control(self, regulator_inputs: list[float]) -> list[float]:
        return [
            fuzzy_pid.get_control([regulator_input])
            for fuzzy_pid, regulator_input in zip(self._fuzzy_pids, regulator_inputs)
        ]
