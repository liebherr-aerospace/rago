"""Define the Base builder class."""

from abc import ABC, abstractmethod
from typing import Any


class Factory(ABC):
    """Base builder class to build object."""

    @staticmethod
    @abstractmethod
    def make(config: dict, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Build something.

        :param config: The config of the thing to build.
        :type config: dict
        :return: The built thing
        :rtype: Any
        """
        ...
