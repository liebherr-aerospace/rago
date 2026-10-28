"""Define utility function and constants for the rest of the repo."""

from hashlib import sha256
from pathlib import Path

PATH_PROJECT = str(Path(__file__).parent.parent.resolve())
PATH_DEFAULT_DATA_DIR = Path(PATH_PROJECT, "data")


def get_document_hash(text: str) -> str:
    """Convert a string into unique id."""
    return sha256(text.encode("utf-8")).hexdigest()
