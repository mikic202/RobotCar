from Regulators.Regulator import Regulator
import torch
import torch.nn as nn

NUMBER_OF_INPUTS = 5
NUMBER_OF_OUTPUTS = 2


class NeuralNetworkRegulator(Regulator):
    def __init__(self, model_file: str):
        class RobotNet(nn.Module):
            def __init__(self):
                super(RobotNet, self).__init__()
                self.fc1 = nn.Linear(NUMBER_OF_INPUTS, 128)
                self.act1 = nn.GELU()
                self.act2 = nn.GELU()
                self.act3 = nn.LeakyReLU()
                self.act4 = nn.GELU()
                self.fc2 = nn.Linear(128, 512)
                self.fc3 = nn.Linear(512, 1024)
                self.fc4 = nn.Linear(1024, 1024)
                self.fc5 = nn.Linear(1024, NUMBER_OF_OUTPUTS)

            def forward(self, x):
                x = self.act1(self.fc1(x))
                x = self.act2(self.fc2(x))
                x = self.act3(self.fc3(x))
                x = self.act4(self.fc4(x))
                x = self.fc5(x)
                return x

        self._model_file = model_file
        self.__model = RobotNet()
        device = torch.device("cpu")
        self.__model.load_state_dict(torch.load(self._model_file, map_location=device))

    def get_controll(self, sensor_data):
        sensor_data = torch.tensor([sensor_data]).to(torch.float)
        data = self.__model(sensor_data).tolist()
        return data
