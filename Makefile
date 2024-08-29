CONFIG = Makefile.conf
include ${CONFIG}

default: run

black:
	@black .

mypy:
	@mypy .

run:
	@python3 -m main --Tp $(TIMER_PERIOD)  --logger $(LOGGER) --robot $(ROBOT) --regulator $(REGULATOR) --reg_args $(REG_ARGS)

test_sensors:
	@python3 -m test_sensors

test_motors:
	@python3 -m test_motors