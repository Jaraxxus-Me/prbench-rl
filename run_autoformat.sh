#!/bin/bash
python -m black . --exclude third-party
docformatter -i -r . --exclude venv third-party
isort . --skip-gitignore --extend-skip-glob="third-party/*"
