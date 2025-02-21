from Regulators.PID import PID
from Regulators.MembershipFunctions.TrapeizodalMembershipFunction import (
    TrapeizodalMembershipFunction,
)
from Regulators.MembershipFunctions.MembershipFunction import MembershipFunction
import json
import numpy as np


class FuzzyPID(PID):
    def __init__(self, param_json_file: str, Tp: float = 1) -> None:
        self._param_file = param_json_file
        self._Tp = Tp
        self._fuzzy_functions: list[MembershipFunction] = []
        self.load_params_from_json(param_json_file)

    def load_params_from_json(self, param_json_file: str) -> None:
        if not param_json_file.endswith(".json"):
            raise ValueError("File must be a json file")

        with open(param_json_file, "r") as file:
            data = json.load(file)

        (
            K_params,
            Ti_params,
            Td_params,
            setpoints,
            self._fuzzy_functions,
        ) = self.parse_pid_json_data(data)
        super().__init__(K_params, Ti_params, Td_params, self._Tp, len(data), setpoints)

    def parse_pid_json_data(self, data: dict) -> tuple[list[float], list[float], list[float], list[float], list[MembershipFunction]]:
        K_params = []
        Ti_params = []
        Td_params = []
        setpoints = []
        fuzzy_functions = []
        for pid_params in data:
            setpoints.append(float(pid_params["setpoint"]))
            K_params.append(float(pid_params["K"]))
            Ti_params.append(float(pid_params["Ti"]))
            Td_params.append(float(pid_params["Td"]))
            fuzzy_functions.append(
                TrapeizodalMembershipFunction(pid_params["fuzzy_functions"])
            )
        return K_params, Ti_params, Td_params, setpoints, fuzzy_functions

    def get_control(self, regulator_inputs: list) -> float:
        raw_control_output = super().get_control(regulator_inputs)
        control_output = []
        for control_value, fuzzy_functions in zip(
            raw_control_output, self._fuzzy_functions
        ):
            control_output.append(fuzzy_functions(regulator_inputs[0]) * control_value)
        return sum(control_output)
