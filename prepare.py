"""
Prepare all archive files for upload to Borealis.

This script processes the raw sourcedata files into the first official version of the corpus.

1. Load/validate/clean corpus file.
2. Load/validate/clean label annotations.
3. Load/validate/clean span annotations.
4. Fill missing values in datapackage template.
5. Export processed files to archive directory.
6. Validate the exported archive.
"""

import argparse
import hashlib
import json
import mimetypes
import random
import string
from contextlib import chdir
from datetime import datetime, timezone

import pandas as pd
from frictionless import Package

import config



def export_json(data, filepath, **kwargs):
    kwargs.setdefault("indent", 4)
    kwargs.setdefault("ensure_ascii", True)
    kwargs.setdefault("sort_keys", False)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)
    return

def export_tabular(data, filepath, **kwargs):
    kwargs.setdefault("index", False)
    kwargs.setdefault("sep", "\t")
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("doublequote", True)
    kwargs.setdefault("quoting", 2)  # 0 = minimal, 1 = all, 2 = nonnumeric, 3 = none
    kwargs.setdefault("quotechar", '"')
    kwargs.setdefault("escapechar", "\\")
    kwargs.setdefault("decimal", ".")
    kwargs.setdefault("lineterminator", "\n")
    kwargs.setdefault("na_rep", "N/A")
    data.to_csv(filepath, **kwargs)
    return


def get_file_hash(filepath, alg="md5"):
    hash_func = hashlib.new(alg)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def generate_identifier(length: int = 4, seed: int = None) -> str:
    assert length >= 2, "Length must be at least 2"
    odd_chars = string.ascii_uppercase
    even_chars = string.digits
    rd = random.Random(seed)
    id_ = "".join(rd.choice(odd_chars if i % 2 == 0 else even_chars) for i in range(length))
    assert id_.isidentifier(), f"Generated ID is not a valid Python identifier: {id_}"
    return id_


def generate_identifier_mapping(unique_values, nchars=4, seed=32) -> dict:
    assert len(unique_values) == len(set(unique_values)), "Non-unique values found"
    mapping = {}
    for value in unique_values:
        while (id_ := generate_identifier(nchars, seed)) in set(mapping.values()):
            seed += 1
        mapping[value] = id_
    return mapping


def get_archive_paths():
    archive_directory = config.directories["archive"]
    datapackage_template_fp = config.directories["sourcedata"] / config.source_filenames["datapackage"]
    with datapackage_template_fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    resources = data["resources"]
    filepaths = {r["name"]: archive_directory / r["path"] for r in resources}
    filepaths["datapackage"] = archive_directory / "datapackage.json"
    # filenames = {
    #     "datapackage": "datapackage.json",
    #     "corpus": f"{archive_acronym}-corpus.tsv",
    #     "labels": f"{archive_acronym}-labels.tsv",
    #     "spans": f"{archive_acronym}-spans.tsv",
    # }
    # filepaths = {name: archive_directory / fn for name, fn in filenames.items()}
    return filepaths

def check_for_archive_files(overwrite: bool = False):
    assert isinstance(overwrite, bool), "Overwrite must be a boolean."
    fpaths = get_archive_paths().values()
    parent = config.directories["archive"]
    parent_exists = parent.exists()
    file_exists = any(fp.exists() for fp in fpaths)
    if parent_exists:
        assert parent.is_dir(), f"{parent} is not a directory."
    for fp in fpaths:
        if fp.exists():
            assert fp.is_file(), f"{fp} is not a file."
    if overwrite:
        if file_exists:
            for fp in fpaths:
                if fp.exists():
                    fp.unlink()
        if parent_exists:
            parent.rmdir()
    elif not overwrite:
        assert not file_exists, "Archive files already exist. Use -o/--overwrite to replace them."
        assert not parent_exists, "Archive directory already exists. Use -o/--overwrite to replace it."
    return


def check_for_source_files():
    """
    Check if the source files are present in the sourcedata directory.

    Raises
    ------
    FileNotFoundError
        If the source files are not present in the sourcedata directory.
    """
    sourcedata_directory = config.directories["sourcedata"]
    if not sourcedata_directory.exists():
        raise FileNotFoundError(f"Source data directory not found at '{sourcedata_directory}'.")
    source_fnames = []
    for value in config.source_filenames.values():
        if isinstance(value, list):
            source_fnames.extend(value)
        elif isinstance(value, str):
            source_fnames.append(value)
        else:
            raise TypeError(f"Unexpected type {type(value)} for source filename {value}.")
    for fn in source_fnames:
        if not (sourcedata_directory / fn).exists():
            raise FileNotFoundError(f"Source file '{fn}' not found in '{sourcedata_directory}'.")


def process_corpus():
    import_path = config.directories["sourcedata"] / config.source_filenames["corpus"]
    CHAR_LENGTH_BETWEEN = (50, 5000)
    COLUMN_ORDER = [
        "id",
        "author",
        "thread",
        "source",
        "subsource",
        "text",
        "type",
    ]
    COLUMN_SORT_ORDER = [
        "source",
        "subsource",
        "author",
        "thread",
        "type",
        "text",
    ]
    DROP_COLUMNS = [
        "failure",  # incomplete
        "plane_vehicle",  # incomplete
        "impossible_object_flight",  # incomplete
        "unassisted_flight_other",  # incomplete
        "age",  # unreliable
        "sex",  # unreliable
        "race",  # unreliable
        "date",  # unreliable
        "user_info",  # redundant with author_id column (slightly different but less reliable, eg [deleted])
    ]
    RENAME_COLUMNS = {
        "report_id": "id",
        "author_id": "author",
        "report": "text",
        "report_type": "type",
        "thread_keywords": "thread",
    }
    DROP_DUPLICATES = [
        "text",
    ]
    REPLACE_VALUES = {
        "type": {
            "dream": "narrative",
            "comment": "observation",
        },
        "sex": {
            1: "male",
            2: "female",
            4: "they",
            "Female": "female",
            "unspecified": "unspecified",
            pd.NA: "unspecified",  # or use .fillna({"sex": "unspecified"})
        },
        "source": {
            "Alchemy forums": "AlchemyForums",
            "International Archive of Dreams": "IntArchivDreams",
            "I Dream of Covid (IDoC)": "IDreamOfCovid",
            "Reddit": "Reddit r/offbeat",  # correcting missing values (r/offbeat/comments/8089p/how_many_people_have_had_true_flying_dreams_where)
            "SDDB Flying Dreams - Export": "SDDb",
            "Straight Dope Message Board > Main > In My Humble Opinion (IMHO) >": "StraightDope IMHO",
            "The Lucidity Institute, Stephen Laberge": "LucidityInstitute",
            "Twitter : @CovidDreams 30mar2020_11jul2022 (33834) for Tobi v2": "Twitter @CovidDreams",
        },
    }
    REPLACE_STRINGS = {
        "thread": {
            r"Barb Sanders #\d+": "Barb Sanders",
        },
        "source": {
            r"^Reddit R/": "Reddit r/",
            r"^Reddit: r/": "Reddit r/",
            r"^Redditr r/": "Reddit r/",
            r"^Reddits r/": "Reddit r/",
            r"^Reddit r/ ": "Reddit r/",
            r"^LD4all\.com : Lucid adventures": "LD4all.com : Lucid Adventures",
            r"^LD4all\.com :Lucid Adventures": "LD4all.com : Lucid Adventures",
            r"^LD4all\.com : quest for lucidity": "LD4all.com : Quest for Lucidity",
            r"r/askreddit": "r/AskReddit",
            r"r/Dream$": "r/Dreams",
            r"r/dreams$": "r/Dreams",
            r"r/lucidDreaming$": "r/LucidDreaming",
            r"r/luciddreaming$": "r/LucidDreaming",
            r"r/shruglifesyndicate$": "r/ShrugLifeSyndicate",
            r"r/shittyaskflying$": "r/Shittyaskflying",
            r"r/astralprojection$": "r/AstralProjection",
            r"r/dreaminterpretation$": "r/DreamInterpretation",
            # r"r/mylittleandysonic2": "r/mylittleandysonic",
            r"^LD4all.com : ": "LD4all ",  # To be consistent for later splitting
        },
    }
    df = (
        # Load the source data corpus file
        pd.read_excel(import_path, sheet_name="Sheet1")
        # Drop columns that are incomplete, unreliable, or redundant
        .drop(columns=DROP_COLUMNS)
        # Rename columns for consistency
        .rename(columns=RENAME_COLUMNS)
        # Replace values in specific columns
        .replace(to_replace=REPLACE_VALUES)
    )
    # Replace source substrings before applying more specific replacements (these are mostly typos or inconsistent formatting)
    # Replace source names with shorter ID-style names
    for column, to_replace in REPLACE_STRINGS.items():
        for pattern, replacement in to_replace.items():
            df[column] = df[column].str.replace(pattern, replacement, regex=True)
    # Add subsources from thread_keywords to sources, in preparation for splitting
    # sources_with_keywords = ["SDDb", "LucidityInstitute", "DreamBank", "IDreamOfCovid", "AlchemyForums", "IAoD"]
    # sources_idx = df["source"].isin(sources_with_keywords)
    # sources_with_keywords = [s for s in df["source"].unique() if s.count(" ") == 0]
    sources_idx = df["source"].str.count(" ").eq(0)
    df.loc[sources_idx, "source"] = (
        df.loc[sources_idx, "source"] + " " + df.loc[sources_idx, "thread"].fillna("")
    )
    df["source"] = df["source"].str.strip()
    # AlchemyForums:
    #  IMHO --> In dreams, is flying always like swimming?
    # I Dream of Covid (IDoC) --> None
    # AlchemyForums --> Mind matters: flying dreams
    # International Archive of Dreams --> Flying and Falling
    # LucidityInstitute

    # Split sources into source and subsource
    df[["source", "subsource"]] = df["source"].str.split(" ", n=1, expand=True)
    # df = df.drop(columns=["source"])

    # Convert thread keywords to randomized IDs
    # Fill empty thread keywords with arbitrary but unique names
    # (To make sure they have unique values when generating thread IDs)
    # (To make sure they have content when merging sources and subsources)
    df.loc[df["thread"].isna(), "thread"] = [str(i) for i in range(df["thread"].isna().sum())]

    df = (
        df
        # Filter out reports with text lengths outside the specified range
        .loc[lambda df: df["text"].str.len().between(*CHAR_LENGTH_BETWEEN)]
        # Drop duplicate reports
        .drop_duplicates(subset=DROP_DUPLICATES)
        # Sort columns and rows
        .reindex(columns=COLUMN_ORDER)
        .sort_values(by=COLUMN_SORT_ORDER, axis="rows", ascending=True, kind="stable")
        # Reset index
        .reset_index(drop=True)
    )
    # Convert existing author IDs to randomized codes
    author_mapping = generate_identifier_mapping(df["author"].unique(), nchars=4, seed=32)
    thread_mapping = generate_identifier_mapping(df["thread"].unique(), nchars=6, seed=323232)
    # report_mapping = generate_identifier_mapping(df["id"].unique(), nchars=8, seed=323232323232)
    df["author"] = df["author"].map(author_mapping)
    df["thread"] = df["thread"].map(thread_mapping)
    # df["id"] = df["id"].map(report_mapping)
    # for i, col in enumerate(ID_COLUMNS, 2):
    #     mapping = utils.generate_identifier_mapping(df[col].unique(), nchars=i * 2, seed=i * 32323232)
    #     df[col] = df[col].map(mapping)
    # def clean_dream_column(ser):
    #     return (ser
    #         .apply(unidecode.unidecode, errors="ignore", replace_str=None)
    #         .str.replace('"', "'")
    #         .str.strip()
    #     )
    # Validate
    assert df.drop(columns=["subsource"]).notna().all(axis=None), "Unexpected empty cells found"
    assert df["id"].is_unique, "Non-unique report IDs found"
    assert df["text"].is_unique, "Non-unique report texts found"
    assert df["type"].isin(["narrative", "observation"]).all(), "Invalid report types found"
    assert df["source"].str.count(" ").eq(0).all(), "Invalid source names found"
    assert df["text"].str.len().between(*CHAR_LENGTH_BETWEEN).all(), "Invalid report text lengths found"
    # assert df["subsource"].str.count(" ").eq(0).all(), "Invalid subsource names found"
    return df


def process_spans():
    annotator_fnames = config.source_filenames["spans"]
    sourcedata_directory = config.directories["sourcedata"]
    # Load annotators into dataframe
    dataframes = {}
    for i, fn in enumerate(sorted(annotator_fnames)):
        import_path = sourcedata_directory / fn
        # Load source data annotation file and validate quality
        df = pd.read_json(import_path, lines=True)
        df = df.drop(columns=["id"]).rename(columns={"report_id": "id", "label": "spans"}).set_index("id")["spans"]
        # assert df.notna().all(axis=None), f"{fn} has missing values."
        # assert df.index.notna().all(), f"{fn} has missing report IDs."
        # assert df.index.is_unique, f"{fn} has duplicate report IDs."
        # assert df["text"].is_unique, f"{fn} has duplicate text reports."
        # assert df["Comments"].str.len().isin([0, 1]).all(), f"{fn} has extra comments."
        # df["id"] = df["id"].str.split("-").str[1]
        # Clean a bit
        # df = df.rename(columns={"label": "spans"}).reindex(columns=["annotator", "id", "spans"])
        # Convert labels (Doccano format) to spans (generalizable, old spaCy format)
        # df["spans"] = df["spans"].apply(lambda labels: [{"start": start, "end": end, "label": label} for start, end, label in labels])
        # # Filter to verify all report IDs are in the corpus
        # df = df.query("id.isin(@corpus.report_id)")
        # Export as JSONL
        # df.to_json(export_path, lines=True, orient="records", force_ascii=False)
        annotator_id = "A" + string.ascii_uppercase[i]
        dataframes[annotator_id] = df
    # # suffixes = ("_" + fname.split(".")[0] for fname in annotator_fnames)
    # # anns = pd.merge(*annotator_dfs, on="id", suffixes=suffixes, how="outer", validate="1:1")
    # d = annotator_dfs[0]
    # d["id"] = d["id"].str.split("-").str[1]
    # x = df.merge(d, on="id", how="left", validate="1:1")
    # x = x.dropna(subset=["text_y"])
    # assert x["text_x"].eq(x["text_y"]).all(), "Texts do not match"
    # x = x.drop(columns=["text_y"]).rename(columns={"spans": "annotator1"})
    # df = pd.concat(annotator_dfs, axis=1)
    df = (
        pd.concat(dataframes, names=["annotator"])
        .explode()
        .apply(pd.Series)
        .rename(columns={0: "start", 1: "end", 2: "label"})
        .swaplevel()
        .reset_index(drop=False)
        # .fillna("N/A")
    )
    return df


def process_labels():
    sourcedata_directory = config.directories["sourcedata"]
    labels_fpath = sourcedata_directory / config.source_filenames["labels"]
    mapping_fpath = sourcedata_directory / config.source_filenames["mapping"]
    labels = pd.read_excel(labels_fpath)
    mapping = pd.read_excel(mapping_fpath, index_col="gpt_id").squeeze().to_dict()
    # Map GPT IDs to report IDs
    labels["id"] = labels["GPT_ID_500"].map(mapping)
    # # Print number of missing IDs
    # n_missing_report_ids = labels["report_id"].isna().sum()
    # print("Number of missing report IDs:", n_missing_report_ids)
    # Drop missing IDs
    df = (
        labels
        .dropna(subset=["id"])
        .set_index("id")
        .filter(regex=r"Lucidity_(T|C)[A-Z]{1,3}")
        .sort_index(axis="columns")
        .pipe(lambda df: df.rename(columns=lambda x: "A" + string.ascii_uppercase[df.columns.get_loc(x)]))
        .melt(ignore_index=False, var_name="annotator", value_name="label")
        .replace({"label": {0: "non-lucid", 1: "lucid"}})
        .reset_index(drop=False)
    )
    # df["annotator"] = df["annotator"].str.split("_").str[1]
    # for r in raters:
    #     # assert df["label"].notna().all(), "Invalid lucidity values."
    #     codings[r] = y[f"Lucidity_{rater}"].map({0: "non-lucid", 1: "lucid"})
    # # Ensure all report IDs are present in final corpus
    # assert lucidity.index.isin(corpus.index).all(), "Missing report IDs in final corpus."
    # Drop to only report IDs in final corpus
    # y = df.merge(codings, on="id", how="inner", validate="1:1")
    # codings = codings.loc[codings.index.intersection(corpus.index)]
    # for i, rater in enumerate(raters, 1):
    #     # assert df["label"].notna().all(), "Invalid lucidity values."
    #     y[f"lucidity_{i}"] = y[f"Lucidity_{rater}"].map({0: "non-lucid", 1: "lucid"})
    #     # df = codings[[f"Lucidity_{rater}", "dream"]].rename(columns={"dream": "text", f"Lucidity_{rater}": "label"})
    #     # df["label"] = df["label"].map({0: "non-lucid", 1: "lucid"})
    #     # export_path = utils.archive_directory / "annotations" / "labels" / f"annotator{i + 1}.jsonl"
    #     # export_path.parent.mkdir(parents=True, exist_ok=True)
    #     # df.reset_index().to_json(export_path, orient="records", lines=True, force_ascii=False)
    return df


def process_datapackage():
    """
    Fill in the blanks in the datapackage template and export it to the derivatives directory.
    Get current date and time in ISO 8601 format
    """
    # Load template
    import_path = config.directories["sourcedata"] / config.source_filenames["datapackage"]
    with import_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Fill empty values that need to be generated from files that were just made
    data["version"] = config.archive_metadata["version"]
    data["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in data["resources"]:
        fp = config.directories["archive"] / r["path"]
        assert fp.exists(), f"Resource file not found: {fp}"
        assert (x := r["format"]) == (y := fp.suffix[1:]), f"File format mismatch: {x} != {y}"
        assert (x := r["mediatype"]) == (y := mimetypes.guess_type(fp)[0]), f"Media type mismatch: {x} != {y}"
        r["hash"] = get_file_hash(fp)
        r["bytes"] = fp.stat().st_size
    return data


def validate_archive():
    # Validate the exported archive.
    archive_dir = config.directories["archive"]
    with chdir(archive_dir):
        # This will catch with a FrictionlessException if the datapackage.json is not valid.
        package = Package("datapackage.json")
        # This will validate resources in package (and other things)
        report = package.validate()
    if not report.valid:
        print(report.to_summary())
    else:
        print("Archive is valid.")
    #     for task in report.tasks:
    #         if task.valid:
    #             name = task.name
    #             path = task.place
    #             stats = task.stats
    #             hash_ = stats["hash"]
    #             bytes_ = stats["bytes"]

    # version = report["version"]
    # from pandera.io import from_frictionless_schema
    # for resource in data["resources"]:
    #     frictionless_schema = resource["schema"]
    #     pandera_schema = from_frictionless_schema(frictionless_schema)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    overwrite = args.overwrite

    # Based on overwrite, check if to-be-exported files already exist
    check_for_source_files()
    check_for_archive_files(overwrite)

    archive_directory = config.directories["archive"]

    archive_directory.mkdir(exist_ok=True, parents=False)

    corpus = process_corpus()
    labels = process_labels()
    spans = process_spans()
    spans = spans.query("id.isin(@corpus.id)")
    labels = labels.query("id.isin(@corpus.id)")
    # corpus = corpus.merge(labels, on="id", how="left", validate="1:1")
    id_mapping = generate_identifier_mapping(corpus["id"].unique(), nchars=8, seed=323232323232)
    corpus["id"] = corpus["id"].map(id_mapping)
    labels["id"] = labels["id"].map(id_mapping)
    spans["id"] = spans["id"].map(id_mapping)
    # spans = spans.dropna(subset=["id"])

    labels = labels.astype("string")
    spans = spans.astype({"id": "string", "annotator": "string", "start": "Int64", "end": "Int64", "label": "string"})
    corpus = corpus.astype("string")

    export_paths = get_archive_paths()
    # Export tabular before processing the data package, because it uses the generated files
    export_tabular(corpus, export_paths["corpus"])
    export_tabular(labels, export_paths["labels"])
    export_tabular(spans, export_paths["spans"])
    datapackage = process_datapackage()
    export_json(datapackage, export_paths["datapackage"])

    validate_archive()
