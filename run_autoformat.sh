#!/bin/bash
python -m black . --exclude "(third-party|\.venv)"
docformatter -i -r . --exclude venv .venv third-party
isort . --skip-gitignore --extend-skip-glob="third-party/*,.venv/*"
