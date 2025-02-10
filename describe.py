"""
Calculation functions for descriptive statistics of the archive.
"""

import pandas as pd
from sklearn import metrics

import config
from prepare import export_tabular


# from frictionless import Package
# def load_dfa(name, update_dtypes=True, primary_key_as_index=True):
#     assert name in ["corpus", "labels", "spans"]
#     package = Package(config.directories["archive"] / "datapackage.json")
#     assert name in package.resource_names, f"Resource {name} not found in archive."
#     resource = package.get_table_resource(name)
#     df = resource.to_pandas()
#     df.resource = resource
#     df.metadata = resource.to_dict()
#     for field in resource.schema.fields:
#         dtype = "Int64" if field.type == "integer" else field.type
#         if field.name == resource.schema.primary_key:
#             df.index = df.index.astype(field.type)
#         else:
#             df[field.name] = df[field.name].astype(dtype)
#     if not primary_key_as_index and resource.schema.primary_key is not None:
#         df = df.reset_index(drop=False)
#     return df


def load_dfa(name, **kwargs):
    assert name in ["corpus", "labels", "spans"]
    import_path = config.directories["archive"] / f"dfa-{name}.tsv"
    return pd.read_table(import_path, **kwargs)


def calculate_counts():
    """
    Calculate frequency of report types and authors across all data sources.
    """
    corpus = load_dfa("corpus", index_col="id")
    counts = (
        corpus.groupby(["source", "type"])["author"]
        .agg(["nunique", "count"])
        .unstack("type", fill_value=0)
        .stack("type", future_stack=True)
        .rename(columns={"nunique": "n_authors", "count": "n_reports"})
        .sort_index(axis="index")
        .sort_index(axis="columns")
    )
    return counts


def calculate_label_agreement():
    """
    Evaluate agreement between islucid coders.
    """
    coder1, coder2 = (
        load_dfa("labels").pivot(index="id", columns="annotator").T.to_numpy()
    )
    return pd.DataFrame(
        [
            {"metric": "accuracy", "value": metrics.accuracy_score(coder1, coder2)},
            {"metric": "kappa", "value": metrics.cohen_kappa_score(coder1, coder2)},
        ]
    )


def calculate_lengths():
    """
    Calculate report length across all sources and report types.
    """
    corpus = load_dfa("corpus", index_col="id")
    # Get lengths
    corpus["n_chars"] = corpus["text"].str.len()
    corpus["n_lemmas"] = corpus["text"].str.split().str.len()
    # Aggregate by report type
    lengths = (
        corpus
        # Get descriptive statistics for character and lemma counts
        .groupby(["source", "type"])[["n_chars", "n_lemmas"]]
        .describe()
        # Create rows for empty report types
        .unstack("type")
        .swaplevel(0, 1, axis=1)
        .fillna({"count": 0})
        .stack([2, 1], future_stack=True)
        .sort_index()
        # Convert multi-level columns to rows
        .reset_index(2)
        .rename(columns={"level_2": "unit"})
        .replace({"unit": {"n_chars": "characters", "n_lemmas": "lemmas"}})
    )
    return lengths


def calculate_subsources():
    """
    Export subsource names and frequencies
    """
    corpus = load_dfa("corpus", index_col="id")
    subsources = (
        corpus.fillna({"subsource": "[none]"})
        .groupby("source")["subsource"]
        .value_counts()
        .sort_index()
    )
    return subsources


if __name__ == "__main__":
    descriptives_directory = config.directories["descriptives"]
    descriptives_directory.mkdir(parents=False, exist_ok=True)
    tables_directory = descriptives_directory / "tables"
    tables_directory.mkdir(parents=False, exist_ok=True)

    counts = calculate_counts()
    subsources = calculate_subsources()
    lengths = calculate_lengths()
    agreement = calculate_label_agreement()

    data = {
        "counts": counts,
        "subsources": subsources,
        "lengths": lengths,
        "agreement": agreement,
    }

    for name, dataframe in data.items():
        export_tabular(
            data=dataframe,
            filepath=tables_directory / f"{name}.tsv",
            index=False if name == "agreement" else True,
        )
