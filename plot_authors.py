"""
Visualize relationship between report and author sample size for each source and report type.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import utils


utils.set_global_matplotlib_settings()

import_path = utils.derivatives_directory / "counts.tsv"
export_path = utils.derivatives_directory / "counts_authors.png"

palette = utils.get_color_palette()
markers = utils.get_marker_palette()

counts = pd.read_table(import_path, index_col=["source", "report_type"])

type_order = ["narrative", "observation"]

scatter_kwargs = dict(s=90, clip_on=False, edgecolor="white", linewidth=0.5)

fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

for (source, report_type), row in counts.iterrows():
    x = row["n_reports"]
    y = row["n_authors"]
    c = palette[report_type]
    m = markers[source]
    z = type_order.index(report_type) + 1
    if x and y:
        ax.scatter(x, y, color=c, marker=m, zorder=z, label=report_type, **scatter_kwargs)

ax.plot([0, 1], [0, 1], "--", lw=1, color="black", zorder=0, transform=ax.transAxes)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of reports")
ax.set_ylabel("Number of authors")
ax.set_xbound(upper=10000)
ax.set_ybound(upper=10000)
ax.set_aspect(1)


patch_kwargs = dict(linewidth=0)
line2d_kwargs = dict(markersize=6, color="gray", markerfacecolor="gray", linewidth=0)
legend_kwargs = dict(
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)

report_type_handles = [
    Patch(facecolor=palette[report_type], label=report_type.capitalize(), **patch_kwargs)
    for report_type in type_order
]
source_handles = [
    Line2D([0], [0], marker=markers[x], label=x, **line2d_kwargs)
    for x in counts.index.get_level_values("source").unique()
]
handles = report_type_handles + source_handles
legend = ax.legend(handles=handles, loc="upper left", **legend_kwargs)
legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels


utils.save_and_close_fig(export_path, include_hires=True)
