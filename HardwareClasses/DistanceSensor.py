import time
import busio
from gpiozero import DigitalOutputDevice

import adafruit_vl53l0x


class DistanceSensor:
    def __init__(self, i2c_bus: busio.I2C, off_pin: int) -> None:
        self._i2c_bus = i2c_bus
        self._off_pin = DigitalOutputDevice(off_pin)
        self._vl53l0x = adafruit_vl53l0x.VL53L0X(self._i2c_bus)

    def __call__(self) -> float:
        return self._vl53l0x.range

    def set_address(self, addres: int):
        self._off_pin.on()
        time.sleep(1.0)
        self._vl53l0x.set_address(addres)

    def off(self):
        self._off_pin.off()
