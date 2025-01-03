from Regulators.Regulator import Regulator
import torch
import torch.nn as nn

NUMBER_OF_INPUTS = 5
NUMBER_OF_OUTPUTS = 2


class NeuralNetworkRegulator(Regulator):
    def __init__(self, model_file: str) -> None:
        class RobotNet(nn.Module):
            def __init__(self):
                super(RobotNet, self).__init__()
                self.fc1 = nn.Linear(NUMBER_OF_INPUTS, 2048)
                self.act1 = nn.Tanh()
                self.d1 = nn.Dropout(0.05)
                self.fc5 = nn.Linear(2048, NUMBER_OF_OUTPUTS)

            def forward(self, x):
                x = self.act1(self.fc1(x))
                x = self.d1(x)
                x = self.fc5(x)
                return x

        self._model_file = model_file
        self.__model = RobotNet()
        self.__model.eval()
        device = torch.device("cpu")
        self.__model.load_state_dict(torch.load(self._model_file, map_location=device))

    def get_control(self, regulator_inputs: list[float]) -> list[float]:
        regulator_inputs = torch.tensor([regulator_inputs]).to(torch.float)
        data = self.__model(regulator_inputs).tolist()
        return data
