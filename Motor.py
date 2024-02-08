from gpiozero import PWMOutputDevice, DigitalOutputDevice


class Motor:
    def __init__(
        self, first_enable_pin: int, second_enable_pin: int, pwm_pin: int, max_pwm: float = 0.9
    ) -> None:
        self.first_enable_pin = first_enable_pin
        self.second_enable_pin = second_enable_pin
        self.pwm_pin = pwm_pin
        self._pwm_output = PWMOutputDevice(pwm_pin)
        self._second_enable_output = DigitalOutputDevice(second_enable_pin)
        self._first_enable_output = DigitalOutputDevice(first_enable_pin)
        self.reset()
        self.max_pwm = max_pwm

    def reset(self):
        self._first_enable_output.off()
        self._second_enable_output.off()

    def set_pwm(self, pwm_value: float):
        if pwm_value == 0:
            self.reset()
        if pwm_value > 0:
            self._first_enable_output.on()
            self._second_enable_output.off()
        else:
            self._first_enable_output.off()
            self._second_enable_output.on()
        self._pwm_output.value = min(abs(pwm_value), self.max_pwm)
        print(self._pwm_output.value)
