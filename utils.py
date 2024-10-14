"""
Utility functions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Define directory paths used in all files
archive_directory = Path("./archive")
derivatives_directory = Path("./derivatives")
sourcedata_directory = Path("./sourcedata")

# Create directories if they do not exist
archive_directory.mkdir(exist_ok=True)
derivatives_directory.mkdir(exist_ok=True)


def get_color_palette() -> dict:
    """
    Get a dictionary of hex-colors for each label.

    Returns
    -------
    dict
        A dictionary where keys are labels (str) and values are their corresponding hex-colors (str).
        The labels include:
        - "narrative": "#0c7bdc"
        - "observation": "#ffc20a"
        - "lucid": "#29c9e6"
        - "flying": "#fca094"
    """
    palette = {
        "narrative": "#0c7bdc",
        "observation": "#ffc20a",
        "lucid": "#29c9e6",  # "#29cae7",
        "flying": "#fca094",  # "#fda094",
    }
    return palette


def get_marker_palette() -> dict:
    """
    Get a dictionary of markers for each source.

    Returns
    -------
    dict
        A dictionary with source names as keys and marker styles as values.
        The marker styles are represented by single-character strings,
        corresponding to the markers available in `Line2D.filled_markers`.
    """
    markers = {
        "AlchemyForums": "h",
        "DreamBank": "o",
        "IDreamOfCovid": "<",
        "IntArchivDreams": "v",
        "LD4all": "s",
        "LucidityInstitute": "8",
        "Reddit": "^",
        "SDDb": "*",
        "StraightDope": "p",
        "Twitter": ">",
    }
    return markers


def load_corpus(**kwargs) -> pd.DataFrame:
    """
    Load the corpus into a pandas DataFrame.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments to pass to `pd.read_csv`.

    Returns
    -------
    pd.DataFrame
        The loaded corpus as a pandas DataFrame.
    """
    kwargs.setdefault("index_col", "report_id")
    kwargs.setdefault("sep", "\t")
    kwargs.setdefault("encoding", "utf-8")
    import_path = archive_directory / "dreamfar-corpus.tsv"
    corpus = pd.read_csv(import_path, **kwargs)
    return corpus


def load_codings(coder: str) -> pd.Series:
    """
    Load codings from a TSV file for a given coder.

    Parameters
    ----------
    coder : str
        The identifier for the coder whose codings are to be loaded.

    Returns
    -------
    pd.Series
        A pandas Series containing the codings, indexed by 'report_id' and renamed to the coder's identifier.
    """
    import_path = archive_directory / "annotations" / "labels" / f"{coder}.tsv"
    codings = pd.read_table(import_path, index_col="report_id").squeeze().rename(coder)
    return codings


def load_annotations(annotator: str, drop_text: bool = True) -> pd.Series | pd.DataFrame:
    """
    Load annotations from a JSONL file and return them as a DataFrame and Series.

    Parameters
    ----------
    annotator : str
        The name of the annotator whose annotations are to be loaded.
    drop_text : bool, optional
        If True, the 'report' column will be dropped from the DataFrame (default is True).

    Returns
    -------
    pd.Series | pd.DataFrame
        The annotations as a pandas Series if `drop_text` is True, otherwise as a pandas DataFrame.
    """
    import_path = archive_directory / "annotations" / "spans" / f"{annotator}.jsonl"
    annotations = pd.read_json(import_path, lines=True).set_index("report_id")
    if drop_text:
        annotations = annotations.drop(columns="report").squeeze()
    return annotations


def save_and_close_fig(path: Path, include_hires: bool = False, **kwargs) -> None:
    """
    Save a matplotlib figure to a specified path and close the figure.

    Parameters
    ----------
    path : Path
        The file path where the figure will be saved.
    include_hires : bool, optional
        If True, an additional high-resolution SVG file will be saved (default is False).
    **kwargs : dict
        Additional keyword arguments passed to `plt.savefig`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **kwargs)
    if include_hires:
        plt.savefig(path.with_suffix(".svg"), **kwargs)
    plt.close()


def set_global_matplotlib_settings(**kwargs) -> None:
    """
    Set global matplotlib settings with custom parameters.

    This function updates the global matplotlib settings with a predefined set of custom settings.
    Additional settings can be provided as keyword arguments, which will override the default settings.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional matplotlib settings to override the default custom settings.
    """
    custom_settings = {
        "interactive": False,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
    custom_settings.update(kwargs)
    plt.rcParams.update(custom_settings)
