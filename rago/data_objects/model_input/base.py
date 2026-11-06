"""Model Input format for Tunable Models."""
from pydantic.dataclasses import dataclass

from rago.data_objects.base import DataObject


@dataclass
class ModelInput(DataObject):
    """Model input format for Tunable Models.

    :param query: The input query.
    :type query: str
    """

    query: str
