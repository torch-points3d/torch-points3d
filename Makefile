.PHONY: staticchecks
staticchecks:
	flake8 . --count --select=E9,F402,F6,F7,F5,F8,F9 --show-source --statistics
	mypy torch_points3d
