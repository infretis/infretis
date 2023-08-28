.PHONY:
	clean
	coverage
	coverage-print
	ruff
	black
	mypy

coverage:
	pytest --cov-report html --cov=infretis test/

coverage-print:
	pytest --cov-report html --cov=infretis -rP test/

ruff:
	poetry run ruff infretis

black:
	poetry run black --check infretis

mypy:
	poetry run mypy -p infretis

clean:
	find -name \*.pyc -delete
	find -name \*.pyo -delete
	find -name __pycache__ -delete
	find -name \*.so -delete
	@if [ -f ".coverage" ]; then \
		echo "\033[31mRemove .coverage\033[0m"; \
		rm .coverage; \
	fi
	@if [ -d "htmlcov" ]; then \
		echo "\033[31mRemove htmlcov directory\033[0m"; \
		rm -rf htmlcov; \
	fi
