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

    def reset(self):
        self._enable_pin.off()
        self._pwm_output.off()

    def set_pwm(self, pwm_value: float):
        if pwm_value < 0.4:
            self.reset()
        if pwm_value > 0:
            self._enable_pin.off()
            self._pwm_output.value = min(abs(pwm_value), self.max_pwm)
        else:
            self._enable_pin.off()
            self._pwm_output.value = min(1-abs(pwm_value), self.max_pwm)
