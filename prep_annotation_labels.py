"""
Go from raw sourcedata manual ratings to what will be the published codings file.
"""

import pandas as pd

import utils


# Load corpus to make sure all report IDs from codings are present in final corpus
corpus = utils.load_corpus()


coder_filepath = utils.sourcedata_directory / "Inter-rater_100dreams.xlsx"
# coder_filepath = Path("./sourcedata") / "Inter-rater_Lucid_100dreams.xlsx"
mapping_filepath = utils.sourcedata_directory / "report2gpt_id_mapping.xlsx"


codings = pd.read_excel(coder_filepath)
mapping = pd.read_excel(mapping_filepath, index_col="gpt_id").squeeze().to_dict()

# Map GPT IDs to report IDs
codings["report_id"] = codings["GPT_ID_500"].map(mapping)

# # Print number of missing IDs
# n_missing_report_ids = codings["report_id"].isna().sum()
# print("Number of missing report IDs:", n_missing_report_ids)

# Drop missing IDs
codings = codings.dropna(subset=["report_id"]).set_index("report_id")

# Clean up by dropping irrelevant columns
codings = codings.filter(like="Lucid").drop(columns="Lucidity_GPT")

# Extract and replace lucidity codings
islucid = codings[["Lucidity_CPD", "Lucidity_TM"]].rename(
    columns={"Lucidity_CPD": "coder1", "Lucidity_TM": "coder2"}
)
assert islucid.notna().all(axis=None), "Missing values in lucidity dataframe."
assert islucid.isin([0, 1]).all(axis=None), "Invalid lucidity values."
islucid = islucid.astype(bool)
islucid = islucid.sort_index(axis=0).sort_index(axis=1)

# # Ensure all report IDs are present in final corpus
# assert lucidity.index.isin(corpus.index).all(), "Missing report IDs in final corpus."

# Drop to only report IDs in final corpus
islucid.index = islucid.index.str.replace("fly-", "report-")
islucid = islucid.loc[islucid.index.intersection(corpus.index)]

# timelines = codings.filter(like="Timeline").rename(columns={"Lucidity_Timeline_CPD": "rater1_timeline", "Lucidity_Timeline_TM": "rater2_timeline"})
# # assert timelines.notna().all(axis=None), "Missing values in lucidity dataframe."
# # assert timelines.isin([1, 2]).all(axis=None), "Invalid lucidity values."
# timelines = timelines.dropna(how="all").replace({1: "lucid_first", 2: "flying_first"})
# timelines.sort_index(axis=0).sort_index(axis=1)
# lucidity = lucidity.melt(ignore_index=False, value_name="islucid", var_name="rater").set_index("rater", append=True).sort_index()
# timelines = timelines.melt(ignore_index=False, value_name="timeline", var_name="rater").set_index("rater", append=True).sort_index()

# Save codings
for coder in islucid:
    export_path = (
        utils.archive_directory
        / "annotations"
        / "labels"
        / f"{coder}.tsv".replace("coder", "annotator")
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)
    ser = islucid[coder].rename("islucid")
    ser.to_csv(export_path, sep="\t", encoding="utf-8")
