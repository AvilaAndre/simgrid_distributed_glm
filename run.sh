#!/bin/bash

# Get data
Rscript Data-01.R

# Run python3 with the correct environment
source .venv/bin/activate
python3 main.py "$@"
