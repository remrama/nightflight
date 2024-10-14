"""
Visualize lemma counts of all reports by source and report type.
"""

import matplotlib.pyplot as plt

import utils


utils.set_global_matplotlib_settings()

export_path = utils.derivatives_directory / "lengths_lemmas.png"

corpus = utils.load_corpus()

palette = utils.get_color_palette()

# Get lengths
corpus["n_chars"] = corpus["report_text"].str.len()
corpus["n_lemmas"] = corpus["report_text"].str.split().str.len()


box_kwargs = dict(
    widths=0.2,
    showmeans=True,
    meanline=True,
    meanprops=dict(color="black", linestyle="dashed", linewidth=0.5),
    medianprops=dict(color="black", linewidth=0.5),
    flierprops=dict(marker="_", markersize=3, markerfacecolor="black"),
    # flierprops=dict(marker=".", markersize=3, markerfacecolor="black", markeredgecolor="none"),
    capprops=dict(linewidth=0.5),
    whiskerprops=dict(linewidth=0.5),
    patch_artist=True,
)

xnudge = box_kwargs["widths"] / 2

type_order = ["narrative", "observation"]
# xtick_labels = list(corpus["source"].unique())
xtick_labels = ["Reddit", "LD4all", "SDDb", "DreamBank"]

fig, ax = plt.subplots(figsize=(3.5, 3), constrained_layout=True)

for source, df in corpus.query("source.isin(@xtick_labels)").groupby("source"):
    for report_type in type_order:
        dvals = df.query(f"report_type == '{report_type}'")["n_chars"].to_numpy()
        xval = xtick_labels.index(source)
        xval = xval - xnudge if report_type == type_order[0] else xval + xnudge
        ax.boxplot(
            dvals,
            positions=[xval],
            boxprops=dict(linewidth=0.5, facecolor=palette[report_type]),
            **box_kwargs,
        )

# ax.set_yscale("log")
ax.set_xlabel("Data source", labelpad=1)
ax.set_ylabel(r"$n$ lemmas", labelpad=1)
ax.set_xticks(range(len(xtick_labels)))
ax.set_xticklabels(xtick_labels, rotation=40, ha="right", fontsize=8)


# min_lemma_count = 10
# max_lemma_count = 4000
# n_bins = 20
# bin_min = 0
# bin_max = max_lemma_count
# bins = np.linspace(bin_min, bin_max, n_bins + 1)
# minor_xtick_loc = np.diff(bins).mean()
# major_xtick_loc = minor_xtick_loc * 5

# ymax = 10000

# # line_kwargs = dict(linewidth=0.5, alpha=1, color="black", linestyle="dashed", clip_on=False, zorder=0)
# bar_kwargs = dict(bins=bins, edgecolor="black", linewidth=0.5, alpha=0.3, clip_on=True)


# fig, ax = plt.subplots(figsize=(3.5, 2), constrained_layout=True)

# for report_type, df in corpus.groupby("report_type"):
#     distvals = df.wordcount.to_numpy()
#     zorder = type_order.index(report_type) + 1
#     ax.hist(distvals, facecolor=palette[report_type], **bar_kwargs)
#     # ax.hist(
#     #     distvals,
#     #     density=True,
#     #     histtype="step",
#     #     zorder=zorder,
#     #     edgecolor=linecolor,
#     #     linewidth=linewidth,
#     #     **bar_kwargs,
#     # )

# ax.set_yscale("log")
# ax.set_ylabel(r"$n$ reports", labelpad=1)
# ax.set_ybound(upper=ymax)
# ax.set_xlabel(r"$n$ lemmas", labelpad=1)
# # ax.spines[["top", "right"]].set_visible(False)
# ax.set_xlim(0, max_lemma_count)
# # ax.tick_params(axis="y", which="both", direction="in")
# ax.xaxis.set(major_locator=plt.MultipleLocator(major_xtick_loc), minor_locator=plt.MultipleLocator(minor_xtick_loc))
# # ax.yaxis.set(major_locator=plt.MultipleLocator(ymax),
# #                 major_formatter=plt.FuncFormatter(c.no_leading_zeros))
# # ax.yaxis.set(major_locator=plt.MultipleLocator(ymax / 2),
# #                 minor_locator=plt.MultipleLocator(ymax / 10))


# handle_kwargs = dict(alpha=bar_kwargs["alpha"], edgecolor=bar_kwargs["edgecolor"], linewidth=bar_kwargs["linewidth"])
# handles = [Patch(label=report_type, facecolor=palette[report_type], **handle_kwargs) for report_type in type_order]
# legend = ax.legend(
#     handles=handles,
#     bbox_to_anchor=(1.0, 0.95),
#     loc="upper right",
#     frameon=False,
#     borderaxespad=0,
#     labelspacing=0.1,
#     handletextpad=0.2,
# )

# # Grab some values for text.
# n, mean, sd, median = ser.describe().loc[["count", "mean", "std", "50%"]]

# # Draw text.
# text_list = [
#     fr"$n={n:.0f}$",
#     fr"$\bar{{x}}={mean:.0f}$",
#     fr"$\sigma_{{\bar{{x}}}}={sd:.1f}$",
#     fr"$\tilde{{x}}={median:.0f}$",
# ]
# if i != 2:
#     for j, txt in enumerate(text_list):
#         ytop = 1 - 0.2 * j
#         ax.text(1, ytop, txt, transform=ax.transAxes, ha="right", va="top")

# # Draw a line and other text.
# ax.axvline(c.MIN_WORDCOUNT, **line_kwargs)
# ax.text(
#     c.MIN_WORDCOUNT + 10,
#     1,
#     "min word cutoff",
#     transform=ax.get_xaxis_transform(),
#     ha="left",
#     va="top",
# )


utils.save_and_close_fig(export_path, include_hires=True)
