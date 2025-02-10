"""
Plotting functions for visualizing the descriptive statistics of the archive.
"""

import argparse

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from spacy import displacy

import config
from describe import load_dfa


def set_rcparams(**kwargs) -> None:
    """
    Set global matplotlib settings with custom parameters.

    This function updates the global matplotlib settings with a predefined set of custom settings.
    Additional settings can be provided as keyword arguments, which will override the default settings.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional matplotlib settings to override the default custom settings.
    """
    rcparams = {
        "interactive": False,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
    rcparams.update(kwargs)
    plt.rcParams.update(rcparams)


def get_palette():
    return {
        "narrative": "#0c7bdc",
        "observation": "#ffc20a",
        "lucid": "#29c9e6",  # #29cae7
        "flying": "#fca094",  # #fda094
    }


def get_markers():
    return {
        "AlchemyForums": "h",
        "DreamBank": "o",
        "IDreamOfCovid": "<",
        "IntArchivDreams": "v",
        "LD4all": "s",
        "LucidityInstitute": "8",
        "Reddit": "^",
        "SDDb": "*",
        "StraightDope": "p",
        "Twitter": ">",
    }


def save_and_close_fig(fstem, suffixes=False, **kwargs):
    fp = config.directories["descriptives"] / "plots" / fstem
    for suff in suffixes:
        plt.savefig(fp.with_suffix(suff), **kwargs)
    plt.close()


def plot_agreement():
    """
    Visualize agreement between islucid coders.
    """
    replacements = {"non-lucid": False, "lucid": True}
    labels = load_dfa("labels").pivot(index="id", columns="annotator", values="label")
    coder1 = labels[labels.columns[0]].rename("Coder 1").map(replacements).astype(bool)
    coder2 = labels[labels.columns[1]].rename("Coder 2").map(replacements).astype(bool)
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


def plot_annotation():
    """
    Visualize annotated report.
    """
    # annotator1["textlength"] = annotator1.report.str.len()
    # annotator1.sort_values("textlength", ascending=True).head(30)
    id_ = "report-e74b"
    annotator = "CPD"
    corpus = load_dfa("corpus", index_col="id")
    spans = load_dfa("spans", index_col=["id", "annotator"])
    text = corpus.at[id_, "text"]
    annotator1 = spans.loc[(id_, annotator)]
    # annotator1 = load_annotations("annotator1", drop_text=False)
    text = annotator1.at[id_, "report"]
    labels = annotator1.at[id_, "labels"]
    # Convert character indices to token indices
    nlp = spacy.blank("en")
    doc = nlp(text)
    char_to_token = {token.idx: token.i for token in doc}
    for i in range(len(text)):
        if i not in char_to_token:
            char_to_token[i] = char_to_token[i - 1]
    # Update labels with token indices
    spans = [
        {
            "start_token": char_to_token[start],
            "end_token": char_to_token[end],
            "label": label.capitalize(),
        }
        for start, end, label in labels
    ]
    palette = {k.capitalize(): v for k, v in get_palette().items()}
    tokens = [token.text for token in doc]
    # tokens = list(doc)
    data = {"text": text, "spans": spans, "tokens": tokens}
    options = {"colors": palette}
    html = displacy.render(data, style="span", manual=True, options=options)
    html = html.replace(
        'ltr"', 'ltr; text-align: center; max-width: 6in; margin: 0 auto;"'
    )
    return html


def plot_authors():
    """
    Visualize relationship between report and author sample size for each source and report type.
    """
    import_path = config.directories["descriptives"] / "tables" / "counts.tsv"
    palette = get_palette()
    markers = get_markers()
    counts = pd.read_table(import_path, index_col=["source", "type"])
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
            ax.scatter(
                x, y, color=c, marker=m, zorder=z, label=report_type, **scatter_kwargs
            )
    ax.plot([0, 1], [0, 1], "--", lw=1, color="black", zorder=0, transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of reports")
    ax.set_ylabel("Number of authors")
    ax.set_xbound(upper=10000)
    ax.set_ybound(upper=10000)
    ax.set_aspect(1)
    patch_kwargs = dict(linewidth=0)
    line2d_kwargs = dict(
        markersize=6, color="gray", markerfacecolor="gray", linewidth=0
    )
    legend_kwargs = dict(
        frameon=False,
        title_fontsize=8,
        fontsize=8,
        handletextpad=0.4,  # space between legend marker and label
        labelspacing=0.1,  # vertical space between the legend entries
        borderaxespad=0,
    )
    report_type_handles = [
        Patch(
            facecolor=palette[report_type],
            label=report_type.capitalize(),
            **patch_kwargs,
        )
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


def plot_sources():
    """
    Visualize the frequency of narratives and observations across all data sources.
    """
    import_path = config.directories["descriptives"] / "tables" / "counts.tsv"
    palette = get_palette()
    counts = pd.read_table(import_path)
    report_counts = (
        counts.pivot(index="source", columns="type", values="n_reports")
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
    ax.set_xticklabels(
        report_counts.index.to_list(), rotation=40, ha="right", fontsize=8
    )
    ax.set_xlabel("Data source")
    ax.set_ylabel("Number of reports")
    # ax.yaxis.set(major_locator=plt.MultipleLocator(500), minor_locator=plt.MultipleLocator(100))
    ax.set_yscale("log")
    ax.set_ybound(upper=10000)
    ax.yaxis.grid(
        True, which="major", linestyle="solid", linewidth=0.5, color="black", alpha=1
    )
    ax.yaxis.grid(
        True, which="minor", linestyle="solid", linewidth=0.5, color="black", alpha=0.1
    )
    ax.set_axisbelow(True)
    patch_kwargs = dict(
        edgecolor=bar_kwargs["edgecolor"], linewidth=bar_kwargs["linewidth"]
    )
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


def plot_wordcount():
    """
    Visualize lemma counts of all reports by source and report type.
    """
    corpus = load_dfa("corpus", index_col="id")
    palette = get_palette()
    # Get lengths
    corpus["n_chars"] = corpus["text"].str.len()
    corpus["n_lemmas"] = corpus["text"].str.split().str.len()
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
    fig, ax = plt.subplots(figsize=(2.5, 3), constrained_layout=True)
    for source, df in corpus.query("source.isin(@xtick_labels)").groupby("source"):
        for report_type in type_order:
            dvals = df.query(f"type == '{report_type}'")["n_chars"].to_numpy()
            xval = xtick_labels.index(source)
            xval = xval - xnudge if report_type == type_order[0] else xval + xnudge
            ax.boxplot(
                dvals,
                positions=[xval],
                boxprops=dict(linewidth=0.5, facecolor=palette[report_type]),
                **box_kwargs,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Data source", labelpad=1)
    ax.set_ylabel(r"$n$ lemmas", labelpad=1)
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=40, ha="right", fontsize=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=["png"],
        choices=["png", "svg", "pdf", "eps"],
        help="Output formats",
    )
    args = parser.parse_args()

    img_extensions = args.extensions
    img_suffixes = [f".{ext}" for ext in img_extensions]

    descriptives_directory = config.directories["descriptives"]
    descriptives_directory.mkdir(parents=False, exist_ok=True)
    plots_directory = descriptives_directory / "plots"
    plots_directory.mkdir(parents=False, exist_ok=True)

    set_rcparams()

    plot_agreement()
    save_and_close_fig("agreement", img_suffixes)

    plot_authors()
    save_and_close_fig("counts_authors", img_suffixes)

    plot_sources()
    save_and_close_fig("counts_sources", img_suffixes)

    plot_wordcount()
    save_and_close_fig("lengths_lemmas", img_suffixes)

    # html = plot_annotation()
    # with (plots_directory / "annotation.html").open("w", encoding="utf-8") as f:
    #     f.write(html)
