default: run

black:
	@black .

mypy:
	@mypy .

run:
	@python3 -m main