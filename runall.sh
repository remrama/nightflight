#!/bin/bash

# This script runs all the scripts in the correct order.
# Before running this script,
#   - make sure you have the necessary packages installed (see environment.yml)
#   - make sure you have the directories properly configured (see config.py)
#   - make sure you have the necessary source data files present

set -e                  # Exit immediately if a command exits with a non-zero status
set -x                  # Print each command before executing it

python prepare.py       # -> ./archive/*
python describe.py      # -> ./descriptives/tables/*
python visualize.py     # -> ./descriptives/plots/*
