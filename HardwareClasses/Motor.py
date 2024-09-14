from gpiozero import PWMOutputDevice, DigitalOutputDevice


class Motor:
    def __init__(
        self, logic_input_1: int, logic_input_2: int, max_pwm: float = 0.9
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
        pwm_value = max(-self.max_pwm, min(self.max_pwm, pwm_value))
        self._current_pwm = pwm_value
        if abs(pwm_value) < 0.3:
            self.reset()
        elif pwm_value > 0:
            self._enable_pin.off()
            self._pwm_output.value = abs(pwm_value)
        else:
            self._enable_pin.on()
            self._pwm_output.value = 1 - abs(pwm_value)

    def get_pwm(self):
        return self._current_pwm
