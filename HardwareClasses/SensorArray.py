from HardwareClasses.DistanceSensor import DistanceSensor
from typing import List
import time
import board
from adafruit_vl53l0x import VL53L0X

class SensorArray:
    def __init__(self, sensor_list: List[DistanceSensor]):
        self._sensors_list = sensor_list
        self._init_sensors()

    def _init_sensors(self):
        for sensor, adress in iter(self._sensors_list):
            sensor.set_address(0x30 + adress)

    def __call__(self):
        return [sensor() for sensor in self._sensors_list]


if __name__ == "__main__":
    i2c = board.I2C()
    array = SensorArray([DistanceSensor(i2c, 25), DistanceSensor(i2c, 11), DistanceSensor(i2c, 1)])

    print(array())