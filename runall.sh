#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Preparation scripts
python prep_corpus.py
python prep_annotation_labels.py
python prep_annotation_spans.py

# Calculation scripts
python calc_counts.py
python calc_subsources.py
python calc_lengths.py

# Plotting scripts
python plot_sources.py
python plot_authors.py
python plot_wordcount.py
python plot_annotation.py
python plot_agreement.py

