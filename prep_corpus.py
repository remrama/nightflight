"""
Go from current raw sourcedata to what will be the first official version of the corpus
and what is uploaded to borealis.
"""

import calendar
import random
import uuid

import pandas as pd

import utils


# Define import and export paths
import_path = utils.sourcedata_directory / "fdd-v0.2.0.xlsx"
export_path = utils.archive_directory / "dreamfar-corpus.tsv"


def generate_unique_ids(n_ids: int, n_chars: int = 4, r_seed: int = 32) -> list[str]:
    """
    Generate a list of unique identifiers.

    This function generates a specified number of unique identifiers, each with a specified number of characters.
    The identifiers are generated using a random seed and are ensured to be valid Python identifiers, contain at least
    one numeric character, and do not contain any month abbreviations.

    Parameters
    ----------
    n_ids : int
        The number of unique identifiers to generate.
    n_chars : int, optional
        The number of characters in each identifier. The default is 4.
    r_seed : int, optional
        The random seed to use for generating identifiers. The default is 32.

    Returns
    -------
    list of str
        A list of unique identifiers.

    Notes
    -----
    - Identifiers are generated using UUIDs and random bits.
    - Identifiers are filtered to ensure they are valid Python identifiers, contain at least one numeric character,
      and do not contain any month abbreviations.
    """
    rd = random.Random()
    rd.seed(r_seed)
    months = [x.lower() for x in calendar.month_abbr[1:]]
    ids = []
    while len(ids) < n_ids:
        h = uuid.UUID(int=rd.getrandbits(128)).hex[:n_chars]
        if (
            h.isidentifier()
            and any(x.isnumeric() for x in h)
            and h not in ids
            and not any(x in h for x in months)
        ):
            ids.append(h)
    return ids


# Load latest corpus
df = pd.read_excel(import_path, index_col="report_id")


#######################################################################################
# Make minor adjustments to corpus file
#######################################################################################

# Drop columns that will not be included in final corpus
df = df.drop(
    columns=[
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
)

# Rename some columns
df = df.rename(columns={"report": "report_text"})

# Update report_type values
df["report_type"] = df["report_type"].replace({"dream": "narrative", "comment": "observation"})

# Replace fly- with report- for report IDS
df.index = df.index.str.replace("fly-", "report-")

# Convert author IDs to randomized codes
n_unique_authors = df["author_id"].nunique()
unique_authors = df["author_id"].unique()
new_author_ids = generate_unique_ids(n_unique_authors, n_chars=4, r_seed=23)
author_id_map = {k: v for k, v in zip(unique_authors, new_author_ids)}
df["author_id"] = df["author_id"].map(author_id_map).map("auth-{}".format)

# Drop rows with duplicated report texts
df = df.drop_duplicates(subset=["report_text"])

# sex_replacements = {
#     1: "male", 2: "female", 4: "they", "Female": "female", "unspecified": "unspecified", pd.NA: "unspecified",  # or use .fillna({"sex": "unspecified"})
# }
# # Remove extremely short and extremely long reports
# lengths = df["dream_text"].str.len()
# df = df.loc[lengths.ge(50) & lengths.le(5000)]
# def clean_dream_column(ser):
#     return (ser
#         .apply(unidecode.unidecode, errors="ignore", replace_str=None)
#         .str.replace('"', "'")
#         .str.strip()
#     )


#######################################################################################
# Clean up sources and break into source/subsource
#######################################################################################

# Replace source substrings before applying more specific replacements (these are mostly typos or inconsistent formatting)
substring_replacements = {
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
}
for typo, fix in substring_replacements.items():
    df["source"] = df["source"].str.replace(typo, fix, regex=True)

# Replace source names with shorter ID-style names
df["source"] = df["source"].replace(
    {
        "Alchemy forums": "AlchemyForums",
        "International Archive of Dreams": "IntArchivDreams",
        "I Dream of Covid (IDoC)": "IDreamOfCovid",
        "Reddit": "Reddit r/offbeat",  # correcting missing values (r/offbeat/comments/8089p/how_many_people_have_had_true_flying_dreams_where)
        "SDDB Flying Dreams - Export": "SDDb",
        "Straight Dope Message Board > Main > In My Humble Opinion (IMHO) >": "StraightDope IMHO",
        "The Lucidity Institute, Stephen Laberge": "LucidityInstitute",
        "Twitter : @CovidDreams 30mar2020_11jul2022 (33834) for Tobi v2": "Twitter @CovidDreams",
    }
)

# Replace some subsource/thread_keywords for consistency and corrections
df["thread_keywords"] = df["thread_keywords"].str.replace(
    r"Barb Sanders #\d+", "Barb Sanders", regex=True
)

# Add subsources from thread_keywords to sources, in preparation for splitting
# sources_with_keywords = ["SDDb", "LucidityInstitute", "DreamBank", "IDreamOfCovid", "AlchemyForums", "IAoD"]
# sources_idx = df["source"].isin(sources_with_keywords)
# sources_with_keywords = [s for s in df["source"].unique() if s.count(" ") == 0]
sources_idx = df["source"].str.count(" ").eq(0)
df.loc[sources_idx, "source"] = (
    df.loc[sources_idx, "source"] + " " + df.loc[sources_idx, "thread_keywords"].fillna("")
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
n_empty_thread_kws = df["thread_keywords"].isna().sum()
df.loc[df["thread_keywords"].isna(), "thread_keywords"] = [
    str(i) for i in range(n_empty_thread_kws)
]
n_unique_threads = df["thread_keywords"].nunique()
unique_threads = df["thread_keywords"].unique()
new_thread_ids = generate_unique_ids(n_unique_threads, n_chars=4, r_seed=3223)
thread_id_map = {k: v for k, v in zip(unique_threads, new_thread_ids)}
df["thread_id"] = df["thread_keywords"].map(thread_id_map).map("thread-{}".format)
df = df.drop(columns=["thread_keywords"])


#######################################################################################
# Clean up, validate, and export
#######################################################################################

# Reorder columns and sort in a sensible fashion
df = df[["author_id", "thread_id", "source", "subsource", "report_type", "report_text"]]
df = df.sort_values(["source", "subsource", "author_id", "thread_id", "report_type", "report_text"])

# Validate
df = df.reset_index(drop=False)
assert df.drop(columns=["subsource"]).notna().all(axis=None), "Unexpected empty cells found"
assert df["report_id"].is_unique, "Non-unique report IDs found"
assert df["report_text"].is_unique, "Non-unique report texts found"
assert df["report_type"].isin(["narrative", "observation"]).all(), "Invalid report types found"
assert df["report_id"].str.startswith("report-").all(), "Invalid report IDs found"
assert df["author_id"].str.startswith("auth-").all(), "Invalid author IDs found"
assert df["thread_id"].str.startswith("thread-").all(), "Invalid thread IDs found"
assert df["source"].str.count(" ").eq(0).all(), "Invalid source names found"
# assert df["subsource"].str.count(" ").eq(0).all(), "Invalid subsource names found"

# Save updated corpus to tab-delimited file
df.to_csv(export_path, index=False, sep="\t", encoding="utf-8")
