"""
Prepare the raw Doccano annotations for the published dataset.
"""

import pandas as pd

import utils


# Define annotators
annotators = ["annotator1", "annotator2"]

# Load the corpus
corpus = utils.load_corpus(index_col=None)

for ann in annotators:
    # Define import and export paths
    import_path = utils.sourcedata_directory / "annotations" / f"{ann}.jsonl"
    export_path = utils.archive_directory / "annotations" / "spans" / f"{ann}.jsonl"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    # Load the annotations
    df = pd.read_json(import_path, lines=True)
    # Check data quality
    assert df.notna().all(axis=None), f"{ann} has missing values."
    assert df.index.notna().all(), f"{ann} has missing report IDs."
    assert df.index.is_unique, f"{ann} has duplicate report IDs."
    assert df["text"].is_unique, f"{ann} has duplicate text reports."
    assert df["Comments"].str.len().isin([0, 1]).all(), f"{ann} has extra comments."
    # Update to new report_id format
    df["report_id"] = df["report_id"].str.replace("fly-", "report-")
    # Filter to verify all report IDs are in the corpus and clean a bit
    df = (
        df.query("report_id.isin(@corpus.report_id)")
        .reindex(columns=["report_id", "label", "text"])
        .rename(columns={"label": "labels", "text": "report"})
        .sort_values("report_id")
    )
    # Export as JSONL
    df.to_json(export_path, lines=True, orient="records", force_ascii=False)
