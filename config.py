import logging
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

def get_logger(fstem=None, **kwargs):
    fstem = fstem or "misc"
    kwargs.setdefault("filename", f"{fstem}.log")
    kwargs.setdefault("level", logging.INFO)
    kwargs.setdefault("filemode", "a")
    kwargs.setdefault("format", "%(asctime)s - %(levelname)s - %(message)s")
    kwargs.setdefault("datefmt", "%Y-%m-%dT%H:%M:%S%z")
    logging.basicConfig(**kwargs)
    return logging.getLogger()
