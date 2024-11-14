# Dream Flight Archive

This repository contains code related to the preparation, description, and visualization of the **Dream Flight Archive**.

The main content of the **Dream Flight Archive** is a manually-curated text corpus of flying dreams and observations about them. The archive additionally contains various human annotations, including binary labels of whether a dream is lucid and highlighted text spans of when lucidity or flying occur in a dream.

> The archive will soon to be available for public use on [Borealis](https://borealisdata.ca/).

## Scripts

* `environment.yaml` was used to create the conda environment where all scripts were run.
* `config.yaml` contains configuration stuff.
* `prepare.py` was used to prepare the raw data for public release on Borealis.
* `describe.py` was used to calculate descriptive statistics of the Borealis archive.
* `visualize.py` was used to plot visualizations of the Borealis archive.
* `runall.sh` was used to run all Python files in order.
