from HardwareClasses.DistanceSensor import DistanceSensor
from gpiozero import DigitalOutputDevice
import board
from multiprocessing import Manager

DEFAULT_ADDRESS = 0x29


class SensorArray:
    def __init__(self, sensor_pins_list: list[int]) -> None:
        self._sensors = []
        self._i2c = board.I2C()
        try:
            self._init_sensors(sensor_pins_list)
        except Exception as e:
            self.reset_addresses()
            raise e
        self._latest_data = Manager().list()

    def _init_sensors(self, sensor_pins_list: list[int]) -> None:
        for adress, (pin, angle) in enumerate(sensor_pins_list):
            print(f"Creating sensor with pin {pin} and angle {angle}")
            sensor = DistanceSensor(self._i2c, DigitalOutputDevice(pin), angle)
            sensor.set_address(DEFAULT_ADDRESS + 1 + adress)
            self._sensors.append(sensor)

    def __call__(self) -> list[float]:
        self._latest_data[:] = []
        [
            self._latest_data.append((sensor(), sensor._angle))
            for sensor in self._sensors
        ]
        return self._latest_data

    def get_latest_data(self) -> list[float]:
        return self._latest_data

    def reset_addresses(self) -> None:
        for sensor in self._sensors:
            sensor.set_address(DEFAULT_ADDRESS)
