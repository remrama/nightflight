# Dream Flight Archive

This repository contains code related to the curation, preparation, description, and visualization of the **Dream Flight Archive**.

The main content of the **Dream Flight Archive** is a manually-curated text corpus of flying dreams and observations about them. The archive additionally contains various human annotations, including binary labels of whether a dream is lucid and highlighted text spans of when lucidity or flying occur in a dream.

> The archive will soon to be available for public use on [Borealis](https://borealisdata.ca/).


## Scripts

This repository contains all scripts/code used to prepare and describe the dataset.

* **Preparation scripts:** Files matching the pattern `prep*_.py` were used to prepare the raw datasets for public release on Borealis. These were used to read and manipulate internal source data that is not otherwise available, and were only used once to prepare the archive. So I could consider these scripts internal and can mostly be ignored now. They are not useful for analyses presented in the descriptor paper.
* **Calculation scripts:** Files matching the pattern `calc_*.py` were used to calculate descriptive statistics of the publicly released version of the archive and are included in the paper.
* **Plotting scripts:** Files matching the pattern `plot_*.py` were used to plot visualizations of the publicly released version of the archive and are included in the paper.
* `config.yaml` contains configuration stuff.
* `environment.yaml` was used to create the conda environment where all scripts were run.
* The `runall.sh` file will run all Python files in order.
