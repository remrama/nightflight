"""
Calculate frequency of report types and authors across all data sources.
"""

import utils


# Set export path
export_path = utils.derivatives_directory / "counts.tsv"

# Load corpus
corpus = utils.load_corpus()

# Calculate counts
counts = (
    corpus.groupby(["source", "report_type"])["author_id"]
    .agg(["nunique", "count"])
    .unstack("report_type", fill_value=0)
    .stack("report_type", future_stack=True)
    .rename(columns={"nunique": "n_authors", "count": "n_reports"})
    .sort_index(axis="index")
    .sort_index(axis="columns")
)

# Export counts
counts.to_csv(export_path, sep="\t", encoding="utf-8")
