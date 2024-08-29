from gpiozero import PWMOutputDevice, DigitalOutputDevice


class Motor:
    def __init__(
        self, logic_input_1: int, logic_input_2: int, max_pwm: float = 0.7
    ) -> None:
        self.logic_input_1 = logic_input_1
        self.logic_input_2 = logic_input_2
        self._pwm_output = PWMOutputDevice(logic_input_1)
        self._enable_pin = DigitalOutputDevice(logic_input_2)
        self.reset()
        self.max_pwm = max_pwm
        self._current_pwm = 0

    def reset(self):
        self._enable_pin.off()
        self._pwm_output.off()

    def breaks(self):
        self._enable_pin.on()
        self._pwm_output.on()

    def set_pwm(self, pwm_value: float):
        self._current_pwm = pwm_value
        if abs(pwm_value) < 0.4:
            self.reset()
        elif pwm_value > 0:
            self._enable_pin.off()
            self._pwm_output.value = min(abs(pwm_value), self.max_pwm)
        else:
            self._enable_pin.on()
            self._pwm_output.value = 1 - min(abs(pwm_value), self.max_pwm)

    def get_pwm(self):
        return self._current_pwm
