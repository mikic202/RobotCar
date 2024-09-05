from Regulators.Regulator import Regulator
import torch
import torch.nn as nn


class NeuralNetworkRegulator(Regulator):
    def __init__(self, model_file: str):
        class RobotNet(nn.Module):
            def __init__(self):
                super(RobotNet, self).__init__()
                self.fc1 = nn.Linear(2, 16)
                self.act1 = nn.LeakyReLU()
                self.dropout1 = nn.Dropout(0.001)
                self.act2 = nn.LeakyReLU()
                self.fc2 = nn.Linear(16, 32)
                self.dropout2 = nn.Dropout(0.001)
                self.fc3 = nn.Linear(32, 2)

            def forward(self, x):
                x = self.act1(self.fc1(x))
                # x = self.dropout1(x)
                x = self.act1(self.fc2(x))
                # x = self.dropout2(x)
                x = self.fc3(x)
                return x

        self._model_file = model_file
        self.__model = RobotNet()
        device = torch.device("cpu")
        self.__model.load_state_dict(torch.load(self._model_file, map_location=device))

    def get_controll(self, sensor_data):
        sensor_data = torch.tensor([sensor_data]).to(torch.float)
        data = self.__model(sensor_data).tolist()
        return data
