"""Define the index config."""

from __future__ import annotations

from typing import Optional

from pydantic.dataclasses import dataclass

from rago.model.configs.base import Config


@dataclass
class IndexConfig(Config):
    """Configuration parameters of an index."""

    type: str
    num_children: Optional[int] = None
