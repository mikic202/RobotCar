from HardwareClasses.DistanceSensor import DistanceSensor
from gpiozero import DigitalOutputDevice
from typing import List
import board

class SensorArray:
    def __init__(self, sensor_pins_list: List[int]):
        self._sensors_list = []
        self._i2c = board.I2C()
        self._init_sensors(sensor_pins_list)

    def _init_sensors(self, sensor_pins_list: List[int]):
        for adress, (pin, angle) in enumerate(sensor_pins_list):
            sensor = DistanceSensor(self._i2c, DigitalOutputDevice(pin), angle)
            # sensor.set_address(0x30 + adress)
            self._sensors_list.append(sensor)

    def __call__(self):
        return [sensor() for sensor in self._sensors_list]


if __name__ == "__main__":
    array = SensorArray([(5,-90), (25,-45), (12, 0), (1, 45), (21, 90)])

    print(array())