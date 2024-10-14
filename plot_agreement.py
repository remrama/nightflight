"""
Visualize agreement between islucid coders.
"""

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils


utils.set_global_matplotlib_settings()

export_path = utils.derivatives_directory / "agreement.png"

# Load human codings
coder1 = utils.load_codings("annotator1").rename("Coder 1")
coder2 = utils.load_codings("annotator2").rename("Coder 2")

# Get confusion matrix
confusion_matrix = (
    pd.crosstab(coder1, coder2)
    .sort_index(axis=0)
    .sort_index(axis=1)
    .rename({False: "Non-lucid", True: "Lucid"}, axis=0)
    .rename({False: "Non-lucid", True: "Lucid"}, axis=1)
)

# Draw heatmap
cmap = cc.cm["gray_r"]
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=cmap, ax=ax)

# Save figure
utils.save_and_close_fig(export_path, include_hires=True)
