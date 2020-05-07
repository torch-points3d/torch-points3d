POETRY=${HOME}/.poetry/bin/poetry

.PHONY: staticchecks
staticchecks:
	${POETRY} run flake8 . --count --select=E9,F402,F6,F7,F5,F8,F9 --show-source --statistics
	${POETRY} run mypy torch_points3d
