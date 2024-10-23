from Regulators.PID import PID
import json
import numpy as np


class FuzzyPID(PID):
    def __init__(self, param_json_file: str, Tp: float = 1):
        self._param_file = param_json_file
        self._Tp = Tp
        self._fuzzy_functions = []
        self.load_params_from_json(param_json_file)

    def load_params_from_json(self, param_json_file: str):
        if not param_json_file.endswith(".json"):
            raise ValueError("File must be a json file")

        with open(param_json_file, 'r') as file:
            data = json.load(file)

        K_params, Ti_params, Td_params, setpoints = self.parse_pid_json_data(data)
        super().__init__(K_params, Ti_params, Td_params, len(data), self._Tp, setpoints)

    def parse_pid_json_data(self, data: dict):
        K_params = []
        Ti_params = []
        Td_params = []
        setpoints = []
        for pid_params in data:
            setpoints.append(pid_params["setpoint"])
            K_params.append(pid_params["K"])
            Ti_params.append(pid_params["Ti"])
            Td_params.append(pid_params["Td"])
            self._fuzzy_functions.append([some_fuzzy_function, pid_params["fuzzy_functions"]])
        return K_params, Ti_params, Td_params, setpoints

    def get_controll(self, input: list):
        raw_controll_output = super().get_controll(input)
        controll_output = []
        for controll_value, fuzzy_functions in zip(input, self._fuzzy_functions):
            controll_output.append(fuzzy_functions[0](controll_value, fuzzy_functions[1]))
        return controll_output


def some_fuzzy_function(x, func_params: list[float]):
    func_params = sorted(func_params)
    if x <= func_params[0] or x >= func_params[-1]:
        return 0
    if x >= func_params[1] and x <= func_params[2]:
        return x
    if x < func_params[1]:
        coefficients = np.polyfit([func_params[0], func_params[1]], [0, 1], 1)
        return np.polyval(coefficients, x)
    coefficients = np.polyfit([func_params[2], func_params[3]], [1, 0], 1)
    return np.polyval(coefficients, x)
