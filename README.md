# NightFlight

This repository contains code related to the curation, preparation, description, and visualization of the Dream Flight Archive (soon to be available for public use on [Borealis](https://borealisdata.ca/)).

The main content of the archive is a manually-curated text corpus of flying dreams and observations about them. The archive additionally contains various human annotations, including binary labels of whether a dream is lucid and highlighted text spans of when lucidity or flying occur in a dream. We plan to publish a dataset descriptor with more details soon.

### TODO

- [ ] Attach derivatives output as GitHub assets with submitted version
- [ ] Update with Borealis link
- [ ] Update with preprint link
- [ ] Add "Accessing data files" section with short example


## Files

See `runall.sh` for a list of all scripts run to prepare the dataset and generate the statistics and figures for the descriptor paper.

#### Preparation scripts

Files matching the pattern `prep*_.py` were used to prepare the raw datasets for public release on Borealis. These were used to read and manipulate internal source data that is not otherwise available, and were only used once to prepare the archive. So I could consider these scripts internal and can mostly be ignored now. They are not useful for analyses presented in the descriptor paper.

#### Calculation scripts

Files matching the pattern `calc_*.py` were used to calculate descriptive statistics of the publicly released version of the archive and are included in the paper.

#### Plotting scripts

Files matching the pattern `plot_*.py` were used to plot visualizations of the publicly released version of the archive and are included in the paper.
