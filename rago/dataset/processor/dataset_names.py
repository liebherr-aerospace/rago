"""Define the Dataset Names in an enum."""

from enum import StrEnum


class DatasetNames(StrEnum):
    """Enum containing the different Dataset Name."""

    ASQA = "asqa"
    CRAG = "crag"
    HOTPOTQA = "hotpot_qa"
    SYNTH = "synth"
