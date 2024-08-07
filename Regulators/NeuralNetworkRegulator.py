from Regulators.Regulator import Regulator
import torch
import torch.nn as nn


class NeuralNetworkRegulator(Regulator):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetworkRegulator.NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1, 1)

        def forward(self, x):
            x = self.fc1(x)
            return x

    def __init__(self, model_file: str):
        self._model_file = model_file
        self.__model = torch.load(model_file)
