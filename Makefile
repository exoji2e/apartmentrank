run:
	pipenv run python3 main.py

test:
	pipenv run python3 -m pytest main.py

typecheck:
	pipenv run mypy --ignore-missing-imports main.py
