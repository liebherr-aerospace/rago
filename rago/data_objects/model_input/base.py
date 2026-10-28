"""Model Input format for Tunable Models."""
from pydantic.dataclasses import dataclass


@dataclass
class ModelInput:
    """Model input format for Tunable Models."""

    query: str
