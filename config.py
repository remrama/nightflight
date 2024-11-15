from pathlib import Path

_sourcedata_dir = "./sourcedata"  # READ-ONLY!!!
_descriptives_dir = "./descriptives"  # WRITE
_archive_dir = "./archive"  # WRITE

directories = {
    "sourcedata": Path(_sourcedata_dir),
    "descriptives": Path(_descriptives_dir),
    "archive": Path(_archive_dir),
}

source_filenames = {
    "datapackage": "datapackage-template.json",
    "corpus": "fdd-v0.2.0.xlsx",
    "mapping": "report2gpt_id_mapping.xlsx",
    "labels": "Inter-rater_100dreams.xlsx",
    "spans": ["cpicard.jsonl", "tmatzek.jsonl"],
}

archive_metadata = {
    "full_name": "Dream Flight Archive",
    "version": "0.0.1-beta",
}
