"""
Visualize the frequency of narratives and observations across all data sources.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import utils

import_path = utils.derivatives_directory / "counts.tsv"
export_path = utils.derivatives_directory / "counts_sources.png"


utils.set_global_matplotlib_settings()

palette = utils.get_color_palette()

counts = pd.read_table(import_path)
report_counts = (
    counts.pivot(index="source", columns="report_type", values="n_reports")
    .pipe(lambda df: df.assign(total=df.sum(axis=1)))
    .sort_values("total", ascending=False)
    .drop(columns="total")
)

type_order = ["narrative", "observation"]

fig, ax = plt.subplots(figsize=(3, 3.5), constrained_layout=True)
bar_kwargs = dict(width=0.8, linewidth=0.5, edgecolor="black")

xvals = np.arange(len(report_counts))
ybottom = np.zeros_like(xvals)
for report_type in type_order:
    yvals = report_counts[report_type].values
    cvals = palette[report_type]
    ax.bar(xvals, yvals, bottom=ybottom, color=cvals, **bar_kwargs)
    ybottom += yvals

ax.set_xticks(xvals)
ax.set_xticklabels(report_counts.index.to_list(), rotation=40, ha="right", fontsize=8)
ax.set_xlabel("Data source")
ax.set_ylabel("Number of reports")
# ax.yaxis.set(major_locator=plt.MultipleLocator(500), minor_locator=plt.MultipleLocator(100))
ax.set_yscale("log")
ax.set_ybound(upper=10000)
ax.yaxis.grid(True, which="major", linestyle="solid", linewidth=0.5, color="black", alpha=1)
ax.yaxis.grid(True, which="minor", linestyle="solid", linewidth=0.5, color="black", alpha=0.1)
ax.set_axisbelow(True)

patch_kwargs = dict(edgecolor=bar_kwargs["edgecolor"], linewidth=bar_kwargs["linewidth"])
handles = [
    Patch(facecolor=palette[report_type], label=report_type, **patch_kwargs)
    for report_type in type_order
]

legend = ax.legend(
    handles=handles,
    loc="upper right",
    title="Report type",
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)

legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels


utils.save_and_close_fig(export_path, include_hires=True)
