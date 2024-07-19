from HardwareClasses.SensorArray import SensorArray
from Loggers.RemoteLogger import RemoteLogger
import json
import time


def convert_to_dict(data):
    return [{"angle": reading[1], "value": reading[0]}for reading in data]


if __name__ == "__main__":
    logger = RemoteLogger("192.168.0.164", 65432)
    time.sleep(1)
    logger2 = RemoteLogger("192.168.0.164", 65433)
    array = SensorArray([(5, -90), (25, -45), (12, 0), (1, 45), (21, 90)])
    try:
        while True:
            time.sleep(0.5)
            logger.log(json.dumps(convert_to_dict(array())))
    finally:
        array.reset_addresses()