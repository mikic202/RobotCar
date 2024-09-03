from HardwareClasses.SensorArray import SensorArray
from time import sleep

if __name__ == "__main__":
    try:
        array = SensorArray([(5, -90), (25, -45), (12, 0), (1, 45), (21, 90), (20, 180)])
        for _ in range(5):
            sleep(1)
            print(f"Sensor values: {array()}")
    finally:
        array.reset_addresses()