from time import time
from typing import Any


class Timer:
    def __init__(self, time_period: float) -> None:
        self._tim_period = time_period
        self.__last_update = time()

    def __call__(self) -> bool:
        current_time = time()
        if current_time - self.__last_update > self._tim_period:
            self.__last_update = current_time
            return True
        return False

    def reset(self) -> None:
        self.__last_update = time()

    @property
    def timer_period(self):
        return self._tim_period

    @timer_period.setter
    def timer_period(self, time_period: float) -> None:
        self._tim_period = time_period
        self.reset()
