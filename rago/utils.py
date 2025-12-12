"""Define utility function and constants for the rest of the repo."""

from pathlib import Path

PATH_PROJECT = str(Path(__file__).parent.parent.resolve())
PATH_DEFAULT_DATA_DIR = Path(PATH_PROJECT, "data")
