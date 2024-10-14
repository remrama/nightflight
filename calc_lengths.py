"""
Calculate report length across all sources and report types.
"""

import utils

export_path = utils.derivatives_directory / "lengths.tsv"

corpus = utils.load_corpus()

# Get lengths
corpus["n_chars"] = corpus["report_text"].str.len()
corpus["n_lemmas"] = corpus["report_text"].str.split().str.len()

# Aggregate by report type
lengths = (
    corpus
    # Get descriptive statistics for character and lemma counts
    .groupby(["source", "report_type"])[["n_chars", "n_lemmas"]]
    .describe()
    # Create rows for empty report types
    .unstack("report_type")
    .swaplevel(0, 1, axis=1)
    .fillna({"count": 0})
    .stack([2, 1], future_stack=True)
    .sort_index()
    # Convert multi-level columns to rows
    .reset_index(2)
    .rename(columns={"level_2": "unit"})
    .replace({"unit": {"n_chars": "characters", "n_lemmas": "lemmas"}})
)

lengths.to_csv(export_path, sep="\t", encoding="utf-8")
