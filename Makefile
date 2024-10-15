CONFIG = Makefile.conf
include ${CONFIG}

default: run

black:
	@black .

mypy:
	@mypy .

run:
	@mkdir -p log/$(TEST_NAME)
	@python3 -m main --Tp $(TIMER_PERIOD)  --logger $(LOGGER) --robot $(ROBOT) --regulator $(REGULATOR) --reg_args $(REG_ARGS) --log_location log/$(TEST_NAME)

test_sensors:
	@python3 -m test_sensors

test_motors:
	@python3 -m test_motors

postprocess_logs:
	@python3 -m Parser.LogsPostprocessor