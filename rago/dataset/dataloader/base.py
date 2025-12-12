"""Define the qa dataset load that loads QA datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, overload

from rago.dataset import QADataset
from rago.dataset.processor import PROCESSOR_REGISTRY, DataProcessor
from rago.utils import PATH_DEFAULT_DATA_DIR

if TYPE_CHECKING:
    from rago.data_objects import EvalSample

DEFAULT_CACHE_DIR = str(os.environ.get("DEFAULT_CACHE_DIR", PATH_DEFAULT_DATA_DIR))


DatasetType = TypeVar("DatasetType", bound=QADataset)


class QADatasetLoader:
    """Loads QA datasets."""

    samples: list[EvalSample]

    @overload
    @classmethod
    def load_dataset(
        cls,
        dataset_class: type[DatasetType],
        name: str,
        dataset_path: Literal[None] = None,
        cache_dir: Optional[str] = None,
        processor: Optional[DataProcessor[DatasetType, Any]] = None,
    ) -> DatasetType | dict[str, DatasetType]: ...

    @overload
    @classmethod
    def load_dataset(
        cls,
        dataset_class: type[DatasetType],
        name: Optional[str] = None,
        dataset_path: str = ...,
        cache_dir: Literal[None] = None,
        processor: Optional[DataProcessor[DatasetType, Any]] = None,
    ) -> DatasetType | dict[str, DatasetType]: ...
    @overload
    @classmethod
    def load_dataset(
        cls,
        dataset_class: type[DatasetType],
        name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        processor: Optional[DataProcessor[DatasetType, Any]] = None,
    ) -> DatasetType | dict[str, DatasetType]: ...
    @classmethod
    def load_dataset(
        cls,
        dataset_class: type[DatasetType],
        name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        processor: Optional[DataProcessor[DatasetType, Any]] = None,
    ) -> DatasetType | dict[str, DatasetType]:
        """Load a dataset, either from cache or by processing raw data.

        :param dataset_class: Type of QA Dataset to create.
        :type dataset_class: type[DatasetType]
        :param name: Dataset name, defaults to None.
        :type name: Optional[str], optional
        :param dataset_path: Path to dataset folder or file, defaults to None
        :type dataset_path: str, optional
        :param cache_dir: Directory to look for cached dataset, defaults to None.
        :type cache_dir: str, optional
        :param processor: Custom data processor to use if cache not found, defaults to None.
        :type processor: DataProcessor, optional
        :return: Loaded dataset or a dict of split datasets.
        :rtype: DatasetType | dict[str, DatasetType]
        """
        dataset_path = QADatasetLoader.get_dataset_path(name, dataset_path, cache_dir)
        if not cls.is_dataset_cached(dataset_path):
            if name is None:
                raise ValueError
            return cls.get_processed_dataset(dataset_class, name, dataset_path, processor)
        return cls.load_from_disk(dataset_class, dataset_path)

    @staticmethod
    def get_dataset_path(
        name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Get the dataset cache path.

        :param dataset_class: Type of QA Dataset to create.
        :type dataset_class: type[DatasetType]
        :param name: Dataset name, defaults to None.
        :type name: Optional[str], optional
        :param dataset_path: Path to dataset folder or file, defaults to None.
        :type dataset_path: Optional[str], optional
        :param cache_dir: Directory to look for cached dataset, defaults to None.
        :type cache_dir: Optional[str], optional
        :return: path to the dataset.
        :rtype: str
        """
        cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else cache_dir
        Path(cache_dir).mkdir(exist_ok=True)

        if dataset_path is None:
            if name is None:
                raise ValueError
            return str(Path(cache_dir, name))
        return dataset_path

    @classmethod
    def list_available_datasets(cls) -> list[str]:
        """Get the available datasets.

        :return: The available datasets.
        :rtype: list[str]
        """
        return list(PROCESSOR_REGISTRY.keys())

    @classmethod
    def get_processed_dataset(
        cls,
        dataset_class: type[DatasetType],
        name: str,
        path_dataset: str,
        processor: Optional[DataProcessor[DatasetType, Any]] = None,
    ) -> DatasetType | dict[str, DatasetType]:
        """Process raw data and create a dataset.

        :param dataset_class: Type of QA Dataset to create.
        :type dataset_class: type[DatasetType]
        :param name: Dataset name.
        :type name: str
        :param path_dataset: Path where dataset will be stored.
        :type path_dataset: str
        :param processor: Processor instance or None (auto-detect).
        :type processor: DataProcessor, optional
        :return: Processed dataset or dict of split datasets.
        :rtype: DatasetType | dict[str, DatasetType]
        """
        processor = (
            DataProcessor[DatasetType, Any].get_data_processor(name, path_dataset) if processor is None else processor
        )
        processed_data = processor.get_processed_dataset()
        dataset_class.save_to_disk(processed_data, path_dataset)
        return processed_data

    @classmethod
    def load_from_disk(
        cls,
        dataset_class: type[DatasetType],
        path_dataset: str,
    ) -> DatasetType | dict[str, DatasetType]:
        """Load dataset from disk (either single file or directory of splits).

        :param dataset_class: Type of QA Dataset to create.
        :type dataset_class: type[DatasetType]
        :param path_dataset: Path to dataset folder or file.
        :type path_dataset: str
        :param split: Specific split to load (optional).
        :type split: str, optional
        :return: Dataset or dict of split datasets.
        :rtype: DatasetType | dict[str, DatasetType]
        """
        if Path(path_dataset).is_dir():
            return cls.get_dataset_dict(dataset_class, path_dataset)
        if not path_dataset.endswith(".json"):
            path_dataset += ".json"
        if Path(path_dataset).is_file():
            return dataset_class.load_from_json(path_dataset, is_list=False)
        raise ValueError

    @classmethod
    def get_dataset_dict(
        cls,
        dataset_class: type[DatasetType],
        path_dataset: str,
    ) -> dict[str, DatasetType]:
        """Load multiple dataset splits from a directory.

        :param dataset_class: Type of QA Dataset to create.
        :type dataset_class: type[DatasetType]
        :param path_dataset: Path containing split datasets.
        :type path_dataset: str
        :return: Mapping of split name to dataset.
        :rtype: dict[str, DatasetType]
        """
        dataset_dict: dict[str, DatasetType] = {}
        for split_name in os.listdir(path_dataset):
            split_path = str(Path(path_dataset, split_name))
            dataset_dict[split_name.replace(".json", "")] = dataset_class.load_from_json(split_path, is_list=False)
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
