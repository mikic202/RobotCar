from Regulators.Regulator import Regulator
import torch
import torch.nn as nn


class NeuralNetworkRegulator(Regulator):
    class RobotNet(nn.Module):
        def __init__(self):
            super(RobotNet, self).__init__()
            self.fc1 = nn.Linear(1, 8)
            self.fc2 = nn.Linear(8, 16)
            self.fc3 = nn.Linear(16, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def __init__(self, model_file: str):
        self._model_file = model_file
        self.__model = torch.load(model_file)
