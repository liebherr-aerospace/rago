"""Define QA dataset dataclass holding QA eval samples."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

from pydantic.dataclasses import dataclass

from rago.data_objects import DataObject, Document, EvalSample
from rago.utils import PATH_DEFAULT_DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DEFAULT_CACHE_DIR = str(os.environ.get("DEFAULT_CACHE_DIR", PATH_DEFAULT_DATA_DIR))


DatasetType = TypeVar("DatasetType", bound="QADataset")
RawDataType = TypeVar("RawDataType")
PreProcessedDataType = TypeVar("PreProcessedDataType")


@dataclass(repr=False)
class QADataset(DataObject):
    """Base dataset class for Question-Answering tasks.

    :param samples: List of evaluation samples (query, context, reference answer).
    :type samples: list[EvalSample]
    """

    samples: list[EvalSample]

    @classmethod
    def load_dataset(
        cls,
        name: str | None = None,
        dataset_path: str | None = None,
        cache_dir: str | None = None,
    ) -> Self | dict[str, Self]:
        """Load a dataset, either from cache or by processing raw data.

        :param name: Dataset name, defaults to None.
        :type name: Optional[str], optional
        :param dataset_path: Path to dataset folder or file, defaults to None
        :type dataset_path: str, optional
        :param cache_dir: Directory to look for cached dataset, defaults to None.
        :type cache_dir: str, optional
        :return: Loaded dataset or a dict of split datasets.
        :rtype: DatasetType | dict[str, DatasetType]
        """
        from .dataloader import QADatasetLoader  # noqa: PLC0415

        return QADatasetLoader.load_dataset(cls, name, dataset_path, cache_dir)

    @classmethod
    def get_dataset_dict(
        cls,
        path_dataset: str,
    ) -> dict[str, Self]:
        """Load multiple dataset splits from a directory.

        :param path_dataset: Path containing split datasets.
        :type path_dataset: str
        :return: Mapping of split name to dataset.
        :rtype: dict[str, DatasetType]
        """
        dataset_dict: dict[str, Self] = {}
        for split_name in Path(path_dataset).iterdir():
            split_path = str(Path(path_dataset, split_name))
            dataset_dict[str(split_name).replace(".json", "")] = cls.load_from_json(split_path, is_list=False)
        return dataset_dict

    @classmethod
    def is_dataset_cached(cls, dataset_path: str) -> bool:
        """Check if dataset exists in cache.

        :param dataset_path: Path to dataset.
        :type dataset_path: str
        :return: True if cached, else False.
        :rtype: bool
        """
        return Path(dataset_path).exists() or Path(dataset_path + ".json").exists()

    @classmethod
    def save_to_disk(
        cls,
        processed_data: Self | dict[str, Self],
        path_dataset: str,
    ) -> None:
        """Save processed dataset to disk.

        :param processed_data: Dataset or dict of datasets.
        :type processed_data: DatasetType | dict[str, DatasetType]
        :param path_dataset: Path to save dataset.
        :type path_dataset: str
        """
        if isinstance(processed_data, QADataset):
            if not path_dataset.endswith(".json"):
                path_dataset += ".json"
            processed_data.save_to_json(path_dataset)
        elif isinstance(processed_data, dict):
            cls._save_dataset_dict_to_disk(processed_data, path_dataset)
        else:
            raise TypeError

    @classmethod
    def _save_dataset_dict_to_disk(
        cls,
        processed_data_dict: dict[str, Self],
        path_dataset: str,
    ) -> None:
        """Save dataset dictionary to disk.

        :param processed_data_dict: Mapping of split name to dataset.
        :type processed_data_dict: dict[str, DatasetType]
        :param path_dataset: Path where datasets will be saved.
        :type path_dataset: str
        """
        Path(path_dataset).mkdir(exist_ok=True)
        for split_name, dataset in processed_data_dict.items():
            path_split = str(Path(path_dataset, split_name))
            cls.save_to_disk(dataset, path_split)

    @property
    def query(self) -> list[str]:
        """Get list of queries in the dataset.

        :return: List of queries.
        :rtype: list[str]
        """
        return [sample.query for sample in self.samples]

    @property
    def reference_answer(self) -> list[str | None]:
        """Get list of reference answers.

        :return: List of answers or None.
        :rtype: list[str | None]
        """
        return [sample.reference_answer for sample in self.samples]

    @property
    def context(self) -> list[list[Document] | None]:
        """Get list of reference contexts.

        :return: List of contexts (list of Documents or None).
        :rtype: list[list[Document] | None]
        """
        return [sample.context for sample in self.samples]

    def add_eval_samples(self, samples: list[EvalSample]) -> None:
        """Add eval samples to the dataset.

        :param samples: The samples to add to the dataset.
        :type samples: list[EvalSample]
        """
        self.samples += samples

    def sample(self, size: float, seed: int | None = None, **kwargs: Any) -> QADataset:  # noqa: ARG002, ANN401
        """Return a sampled subset of the dataset.

        :param size: Fraction or number of samples to select.
        :type size: float
        :param kwargs: Extra arguments.
        :type kwargs: Any
        :return: Sampled dataset.
        :rtype: QADataset
        """
        if seed is not None:
            random.seed(seed)
        samples = self.get_num_samples(size)
        return self.__class__(samples=samples)

    def get_num_samples(self, size: float) -> list[EvalSample]:
        """Get the number of samples to sample base on the size float parameter.

        :param size: Fraction (float) or number of samples (int).
        :type size: float
        :return: Sampled list of evaluation samples.
        :rtype: list[EvalSample]
        """
        num_samples = int(len(self.samples) * size) if not isinstance(size, int) else size
        return self.shuffle_sample()[:num_samples]

    def shuffle_sample(
        self,
    ) -> list[EvalSample]:
        """Shuffle dataset samples randomly.

        :return: Shuffled list of evaluation samples.
        :rtype: list[EvalSample]
        """
        samples = copy.deepcopy(self.samples)
        random.shuffle(samples)
        return samples

    @staticmethod
    def get_default_split_names(splits: Sequence[float]) -> list[str]:
        """Return the default split names for the splits.

        :param splits: the split ratios.
        :type splits: Sequence[float]
        :return: The splits names.
        :rtype: list[str]
        """
        return [f"split_{i}" for i in range(len(splits))]

    def _split_samples(
        self,
        splits: Sequence[float],
        split_names: list[str] | None = None,
    ) -> dict[str, list[EvalSample]]:
        split_names = QADataset.get_default_split_names(splits) if split_names is None else split_names
        samples_splits: dict[str, list[EvalSample]] = {}
        shuffled_samples = self.shuffle_sample()
        id_start = 0
        for i, name in enumerate(split_names):
            id_end = len(shuffled_samples) if i == len(splits) else id_start + int(len(shuffled_samples) * splits[i])
            samples_splits[name] = shuffled_samples[id_start:id_end]
            id_start = id_end
        return samples_splits

    def split_dataset(
        self,
        splits: Sequence[float],
        split_names: list[str] | None = None,
        seed: int | None = None,
    ) -> dict[str, Self]:
        """Split dataset into multiple parts.

        :param splits: Proportions for each splits (sum <= 1).
        :type splits: Sequence[float]
        :param split_names: The name of the splits to create, defaults to None.
        :type split_names: Optional[list[str]], optional
        :param seed: The seed used to set the random generator, defaults to None.
        :type seed: Optional[int], optional
        :return: Mapping of split names to datasets.
        :rtype: dict[str, DatasetType]
        """
        if seed is not None:
            random.seed(seed)
        splitted_samples = self._split_samples(splits, split_names)
        dataset_splits = {
            split_name: self.__class__(samples=samples) for split_name, samples in splitted_samples.items()
        }
        return dataset_splits

    def map_samples(self, map_function: Callable[[EvalSample], EvalSample]) -> Self:
        """Apply `map_function` to all the the samples in the dataset and return the modified dataset.

        :param self: The source dataset.
        :type self: DatasetType
        :param map_function: The function to apply to every samples in the dataset.
        :type map_function: Callable[[EvalSample], EvalSample]
        :return: The new dataset with modified samples.
        :rtype: DatasetType
        """
        n_ds = copy.deepcopy(self)
        for idx, sample in enumerate(n_ds.samples):
            n_ds.samples[idx] = map_function(sample)
        return n_ds

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the dataset.

        :return: String representation with class name and key attributes.
        :rtype: str
        """
        return self.__str__()

    def __str__(self) -> str:
        """Return a human-readable summary of the dataset.

        :return: Summary string including number of samples and attributes.
        :rtype: str
        """
        return f"""
{self.__class__.__name__}:
Samples:
number: {len(self.samples)}
attr: {", ".join(list(self.samples[0].__annotations__.keys()))}
"""

    def _repr_html_(self) -> str:
        return self.__repr__()
