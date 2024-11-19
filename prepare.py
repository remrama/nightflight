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
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import spacy
from frictionless import Package
from spacy.language import Language
from spacy.tokens import Span

import config



def export_json(data: dict, filepath: Union[str, Path], **kwargs: Optional[Any]) -> None:
    kwargs.setdefault("indent", 4)
    kwargs.setdefault("ensure_ascii", True)
    kwargs.setdefault("sort_keys", False)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)
    return

def export_tabular(data: pd.DataFrame, filepath: Union[str, Path], **kwargs: Optional[Any]) -> None:
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


def get_file_hash(filepath: Union[str, Path], alg: str = "md5") -> str:
    hash_func = hashlib.new(alg)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def generate_identifier(length: int = 4, seed: Optional[int] = None) -> str:
    assert length >= 2, "Length must be at least 2"
    odd_chars = string.ascii_uppercase
    even_chars = string.digits
    rd = random.Random(seed)
    id_ = "".join(rd.choice(odd_chars if i % 2 == 0 else even_chars) for i in range(length))
    assert id_.isidentifier(), f"Generated ID is not a valid Python identifier: {id_}"
    return id_


def generate_identifier_mapping(unique_values: list[Union[int, str]], nchars: int = 4, seed: int = 32) -> dict:
    assert len(unique_values) == len(set(unique_values)), "Non-unique values found"
    mapping = {}
    for value in unique_values:
        while (id_ := generate_identifier(nchars, seed)) in set(mapping.values()):
            seed += 1
        mapping[value] = id_
    return mapping


def get_archive_paths() -> dict[str, Path]:
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

def check_for_archive_files(overwrite: bool = False) -> None:
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


def check_for_source_files() -> None:
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


def _merge_overlapping_spans(spans: list[Span]) -> list[Span]:
    """
    Merge overlapping spans into a single span.

    Parameters
    ----------
    spans : List[Span]
        List of Span objects to merge.

    Example
    -------
    >>> spans = [Span(doc, 0, 5, label="flying"), Span(doc, 3, 8, label="flying")]
    >>> merged_spans = _merge_overlapping_spans(spans)
    >>> merged_spans
    [Span(doc, 0, 8, label="flying")]
    """
    assert all(isinstance(span, Span) for span in spans), "All spans must be Span objects"
    merged_spans = spacy.util.filter_spans(spans)
    return merged_spans


def _merge_touching_spans(spans: list[Span]) -> list[Span]:
    """
    Merge touching/connected spans into a single span.

    Parameters
    ----------
    spans : List[Span]
        List of Span objects to merge.

    Example
    -------
    >>> spans = [Span(doc, 0, 5, label="flying"), Span(doc, 5, 8, label="flying")]
    >>> merged_spans = _merge_touching_spans(spans)
    >>> merged_spans
    [Span(doc, 0, 8, label="flying")]
    """
    assert all(isinstance(span, Span) for span in spans), "All spans must be Span objects"
    merged_spans = []
    for span in spans:
        if merged_spans and merged_spans[-1].end == span.start:
            merged_spans[-1] = Span(
                span.doc,
                merged_spans[-1].start,
                span.end,
                label=span.label_,
                # span_id=span.id_ ,
            )
        else:
            merged_spans.append(span)
    return merged_spans


def merge_spans_on_groupby(df: pd.DataFrame, texts: dict[str, str], nlp: Language) -> pd.DataFrame:
    """
    This function is used to merge overlapping spans within a groupby object.
    The overall number of spans is reduced because it combines overlapping
    and touching spans within each annotator/label group (for each id).
    Also drops empty spans (where start is NaN).

    1. Overlapping spans (e.g., [0, 5] and [3, 8]) are merged into a single span [0, 8]
    2. Touching spans (e.g., [0, 5] and [5, 8]) are merged into a single span [0, 8]

    It only works when passed to groupby.apply as such:
        .groupby("id", as_index=True).apply(_merge_spans_on_groupby)
    and expects the groupby object to have the following columns:
        ["annotator", "label", "start", "end", "span_id"]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with index named "id" and columns ["annotator", "label", "start", "end", "span_id"]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["annotator", "label", "start", "end", "span_id"]
        where overlapping and touching spans have been merged.
        
    Example
    -------
    >>> df = pd.DataFrame({
    ...     "id": ["H5Y2K6U1", "H5Y2K6U1", "H5Y2K6U1", "H5Y2K6U1", "H5Y2K6U1", "H5Y2K6U1"],
    ...     "annotator": ["AA", "AA", "AA", "AB", "AB", "AB"],
    ...     "label": ["flying", "flying", "flying", "flying", "flying", "flying"],
    ...     "start": [0, 3, 5, 0, 3, 5],
    ...     "end": [5, 8, 8, 5, 8, 8],
    ...     "span_id": ["A0", "A1", "A2", "B0", "B1", "B2"],
    ... })
    >>> df.groupby("id", as_index=True).apply(_merge_spans_on_groupby, include_groups=False).reset_index(1, drop=True)
             annotator   label  start  end span_id
    id
    H5Y2K6U1        AA  flying      0   13      A0
    H5Y2K6U1        AB  flying      0   13      B0
    """
    id_ = df.name
    # id_ = "A0Z7A2J1"
    text = texts[id_]
    doc = nlp(text)
    def _row_to_token_span(row: pd.Series) -> Span:
        """Extract row data about character span and return spaCy token span."""
        return doc.char_span(row["start"], row["end"], label=row["label"], alignment_mode="expand")
        # return doc.char_span(row["start"], row["end"], label=row["label"], span_id=row["span_id"], alignment_mode="expand")

    spans_data = []
    for (annotator, label), df_ in df.groupby(["annotator", "label"]): #, as_index=True, sort=True):
        raw_spans = df_.apply(_row_to_token_span, axis=1).tolist()
        assert all(raw_spans), "Invalid span"
        merged_spans_no_overlap = _merge_overlapping_spans(raw_spans)
        merged_spans_no_touch = _merge_touching_spans(merged_spans_no_overlap)
        for span in merged_spans_no_touch:
            span_data = {
                "annotator": annotator,
                "start": span.start_char,
                "end": span.end_char,
                "label": span.label_,
                # "span_id": span.id_,
            }
            spans_data.append(span_data)
    # print("====================================================")
    # print(id_)
    # print("----------------------------------------------------")
    # print(spans_data)
    return (
        pd.DataFrame(spans_data)
        .sort_values(["annotator", "label", "start"], kind="stable")
        .reindex(columns=["annotator", "label", "start", "end"])
        .reset_index(drop=True)
    )
    # group = SpanGroup(doc, name=label, spans=merged_spans, attrs={"annotator": annotator})
    # doc.spans[ann] = group
    # span_groups.append(group)


def process_corpus() -> pd.DataFrame:
    import_path = config.directories["sourcedata"] / config.source_filenames["corpus"]
    CHAR_LENGTH_BETWEEN = (50, 5000)
    COLUMN_ORDER = [
        "id",
        "author",
        "thread",
        "source",
        "subsource",
        "age",
        "sex",
        "race",
        "date",
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
        # "age",  # unreliable
        # "sex",  # unreliable
        # "race",  # unreliable
        # "date",  # unreliable
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
            1: "male (he/him)",
            2: "female (she/her)",
            3: "they/them",
            4: "she/they",
            5: "they/he",
            6: "he/they",
            7: "they/she",
            "Female": "female",
            "unspecified": "unspecified",
            # pd.NA: "unspecified",  # or use .fillna({"sex": "unspecified"})
        },
        "race": {
            0: "other/mixed race",
            1: "white",
            2: "hispanic",
            3: "african american/black",
            4: "asian or pacific islander",
            5: "native american or alaskan native",
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
    # # Validate
    # assert df.drop(columns=["subsource"]).notna().all(axis=None), "Unexpected empty cells found"
    # assert df["id"].is_unique, "Non-unique report IDs found"
    # assert df["text"].is_unique, "Non-unique report texts found"
    # assert df["type"].isin(["narrative", "observation"]).all(), "Invalid report types found"
    # assert df["source"].str.count(" ").eq(0).all(), "Invalid source names found"
    # assert df["text"].str.len().between(*CHAR_LENGTH_BETWEEN).all(), "Invalid report text lengths found"
    # # assert df["subsource"].str.count(" ").eq(0).all(), "Invalid subsource names found"
    df = df.astype("string")
    return df


def process_spans(corpus_texts: dict[str, str], nlp: Language) -> pd.DataFrame:
    """
    Load and process the span annotations from the source files.
    Each annotator's Doccano annotation file is loaded and the spans are extracted.
    All annotations/annotators are combined into one dataframe.
    """
    def _insert_empty_spans(spans_list: list[list[Optional[str]]]) -> list[list[Optional[str]]]:
        # Mostly a correction step bc they annotated both at the same time so not really an "empty" output
        # Get labels from datapackage
        empty_code = pd.NA
        possible_labels = ["lucid", "flying"]
        annotated_labels = [x[2] for x in spans_list]
        for label in possible_labels:
            if label not in annotated_labels:
                spans_list.append([empty_code, empty_code, label])
        return spans_list

    annotator_fnames = config.source_filenames["spans"]
    sourcedata_directory = config.directories["sourcedata"]
    annotations = {}
    for i, fn in enumerate(sorted(annotator_fnames)):
        annotations["A" + string.ascii_uppercase[i]] = (
            pd.read_json(sourcedata_directory / fn, lines=True)
            .drop(columns=["id"])
            .rename(columns={"report_id": "id", "label": "spans"})
            .set_index("id")
            ["spans"]
        )
    raw_spans = (
        pd.concat(annotations, names=["annotator"])
        .explode()
        .apply(pd.Series)
        .rename(columns={0: "start", 1: "end", 2: "label"})
        .swaplevel()
        .reset_index(drop=False)
    )
    columns = ["id", "span_id", "annotator", "start", "end", "label"]
    sort_columns = ["annotator", "label", "start"]
    corpus_ids = list(corpus_texts.keys())
    merged_spans = (
        raw_spans
        .query("id.isin(@corpus_ids)")
        # Span merging
        .groupby("id", as_index=True)
        .apply(merge_spans_on_groupby, include_groups=False, texts=corpus_texts, nlp=nlp)
        .reset_index("id", drop=False)
        # # Fill empty spans
        # .apply(_insert_empty_spans)
    )
    # Add empty spans to id/annotator/label combos that don't exist
    empty_spans = (
        merged_spans
        .groupby(["id", "annotator"])["label"]
        .value_counts()
        .unstack(["annotator", "label"], fill_value=0)
        .eq(0)
        .melt(value_name="no_spans", ignore_index=False)
        .query("no_spans")
        .drop(columns=["no_spans"])
        .reset_index(drop=False)
        # .assign(start=pd.NA, end=pd.NA)
    )
    spans = pd.concat([merged_spans, empty_spans], axis=0, ignore_index=True)
    span_mapping = generate_identifier_mapping(spans.index, nchars=3, seed=23)
    spans = (
        spans
        .assign(span_id=spans.index.map(span_mapping))
        # Tidy up: sort rows and columns, reset index, set types
        .reindex(columns=columns)
        .sort_values(sort_columns, kind="stable")
        .reset_index(drop=True)
        .astype({"id": "string", "span_id": "string", "annotator": "string", "start": "Int64", "end": "Int64", "label": "string"})
    )
    return spans

def process_labels(corpus_ids: list[str]) -> pd.DataFrame:
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
        .query("id.isin(@corpus_ids)")
        .reset_index(drop=False)
    )
    label_mapping = generate_identifier_mapping(df.index, nchars=3, seed=232323)
    df = df.assign(label_id=df.index.map(label_mapping))
    df = df.reindex(columns=["id", "label_id", "annotator", "label"])
    df = df.sort_values(["id", "annotator", "label"])
    df = df.reset_index(drop=True)
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
    df = df.astype({"id": "string", "label_id": "string", "annotator": "string", "label": "string"})
    return df


def process_datapackage() -> dict:
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


def validate_archive() -> None:
    # Validate the exported archive.
    archive_dir = config.directories["archive"]
    with chdir(archive_dir):
        # This will catch with a FrictionlessException if the datapackage.json is not valid.
        package = Package("datapackage.json")
        # This will validate resources in package (and other things)
        report = package.validate()
    if not report.valid:
        print(report.to_summary())
        print("Archive was exported but is NOT valid. See above for details.")
    else:
        print("Archive was exported and validated. See above for details.")
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





def main(overwrite: bool) -> None:

    # Based on overwrite, check if to-be-exported files already exist
    check_for_source_files()
    check_for_archive_files(overwrite)

    archive_directory = config.directories["archive"]
    archive_directory.mkdir(exist_ok=True, parents=False)

    corpus = process_corpus()
    corpus_texts = corpus.set_index("id")["text"].to_dict()
    unique_corpus_ids = list(corpus_texts)
    labels = process_labels(corpus_ids=unique_corpus_ids)
    nlp = spacy.load("blank:en")
    spans = process_spans(corpus_texts=corpus_texts, nlp=nlp)

    # corpus = corpus.merge(labels, on="id", how="left", validate="1:1")
    id_mapping = generate_identifier_mapping(unique_corpus_ids, nchars=8, seed=323232323232)
    corpus["id"] = corpus["id"].map(id_mapping).astype("string")
    labels["id"] = labels["id"].map(id_mapping).astype("string")
    spans["id"] = spans["id"].map(id_mapping).astype("string")

    export_paths = get_archive_paths()
    # Export tabular before processing the data package, because it uses the generated files
    export_tabular(corpus, export_paths["corpus"])
    export_tabular(labels, export_paths["labels"])
    export_tabular(spans, export_paths["spans"])
    datapackage = process_datapackage()
    export_json(datapackage, export_paths["datapackage"])

    validate_archive()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    overwrite = args.overwrite
    main(overwrite=overwrite)
