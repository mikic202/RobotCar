from Regulators.PID import PID
from Regulators.MembershipFunctions.TrapeizodalMembershipFunction import (
    TrapeizodalMembershipFunction,
)
from Regulators.MembershipFunctions.MembershipFunction import MembershipFunction
import json
import numpy as np


class FuzzyPID(PID):
    def __init__(self, param_json_file: str, Tp: float = 1):
        self._param_file = param_json_file
        self._Tp = Tp
        self._fuzzy_functions: list[MembershipFunction] = []
        self.load_params_from_json(param_json_file)

    def load_params_from_json(self, param_json_file: str):
        if not param_json_file.endswith(".json"):
            raise ValueError("File must be a json file")

        with open(param_json_file, "r") as file:
            data = json.load(file)

        K_params, Ti_params, Td_params, setpoints, self._fuzzy_functions = self.parse_pid_json_data(data)
        super().__init__(K_params, Ti_params, Td_params, self._Tp, len(data), setpoints)

    def parse_pid_json_data(self, data: dict):
        K_params = []
        Ti_params = []
        Td_params = []
        setpoints = []
        fuzzy_functions = []
        for pid_params in data:
            setpoints.append(pid_params["setpoint"])
            K_params.append(pid_params["K"])
            Ti_params.append(pid_params["Ti"])
            Td_params.append(pid_params["Td"])
            fuzzy_functions.append(
                TrapeizodalMembershipFunction(pid_params["fuzzy_functions"])
            )
        return K_params, Ti_params, Td_params, setpoints, fuzzy_functions

    def get_controll(self, inputs: list):
        # figure out how to split for different inputs
        raw_controll_output = super().get_controll(inputs)
        controll_output = []
        for controll_value, fuzzy_functions in zip(
            raw_controll_output, self._fuzzy_functions
        ):
            controll_output.append(fuzzy_functions(inputs[0]) * controll_value)
        return sum(controll_output)
