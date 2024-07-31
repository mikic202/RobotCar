from time import time
from typing import Any


class Timer:
    def __init__(self, time_period: float):
        self._tim_period = time_period
        self.__last_update = time()

    def __call__(self) -> bool:
        current_time = time()
        if current_time - self.__last_update > self._tim_period:
            self.__last_update = current_time
            return True
        return False