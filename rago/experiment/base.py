"""Defines an abstract experiment class to run experiments."""

from abc import ABC, abstractmethod


class Experiment(ABC):
    """Abstract Experiment class to carry out experiments."""

    @abstractmethod
    def run(self) -> None:
        """Run an experiment."""
