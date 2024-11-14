"""
Utility functions.
"""
import logging
from pathlib import Path

import yaml

# Load configuration from `config.yml`
with open("./config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Convert all directory paths to `pathlib.Path` objects
for k, v in config["directories"].items():
    config["directories"][k] = Path(v)

def get_logger(fstem=None, **kwargs):
    fstem = fstem or "misc"
    kwargs.setdefault("filename", f"{fstem}.log")
    kwargs.setdefault("level", logging.INFO)
    kwargs.setdefault("filemode", "a")
    kwargs.setdefault("format", "%(asctime)s - %(levelname)s - %(message)s")
    kwargs.setdefault("datefmt", "%Y-%m-%dT%H:%M:%S%z")
    logging.basicConfig(**kwargs)
    return logging.getLogger()
