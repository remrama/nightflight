"""
Export subsource names and frequencies
"""

import utils

export_path = utils.derivatives_directory / "subsources.tsv"

corpus = utils.load_corpus()

subsources = (
    corpus.fillna({"subsource": "[none]"})
    .groupby("source")["subsource"]
    .value_counts()
    .sort_index()
)

subsources.to_csv(export_path, sep="\t", encoding="utf-8")
