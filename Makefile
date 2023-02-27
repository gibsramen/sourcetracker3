all: stylecheck test

stylecheck:
	flake8 st3/*.py setup.py st3/tests/*.py

test:
	pytest st3/tests
