from Regulators.PID import PID
import json


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
            self._fuzzy_functions.append(pid_params["fuzzy_functions"])
        return K_params, Ti_params, Td_params, setpoints

    def get_controll(self, input: list):
        raw_controll_output = super().get_controll(input)
        controll_output = []
        for controll_value, fuzzy_functions in zip(raw_controll_output, self._fuzzy_functions):
            controll_output.append(self.calculate_fuzzy_output(controll_value, fuzzy_functions))
