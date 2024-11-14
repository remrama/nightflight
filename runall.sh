#!/bin/bash

set -e                  # Exit immediately if a command exits with a non-zero status
set -x                  # Print each command before executing it

python prepare.py       # -> ./archive/*
python describe.py      # -> ./descriptives/tables/*
python visualize.py     # -> ./descriptives/plots/*
