"""Define the Test for qa dataset."""

import time
from pathlib import Path
from typing import cast

import pytest

from rago.dataset import QADataset, QADatasetLoader


@pytest.fixture
def expected_default_dataset_path(datasets_path: str, default_dataset_name: str) -> Path:
    return Path(datasets_path, default_dataset_name)


@pytest.fixture
def expected_sampled_dataset() -> QADataset:
    return QADataset.load_from_json("./tests/assets/expected_sampled_dataset.json", is_list=False)


def test_loading_uncached_dataset(
    default_dataset_name: str,
    datasets_path: Path,
    expected_default_dataset_path: Path,
) -> None:
    ds_dict = QADatasetLoader.load_dataset(QADataset, default_dataset_name, cache_dir=str(datasets_path))
    assert isinstance(ds_dict, dict)
    assert all(isinstance(key, str) and isinstance(value, QADataset) for key, value in ds_dict.items())
    assert expected_default_dataset_path.is_dir()
    for split_name in ds_dict:
        assert Path(expected_default_dataset_path, split_name + ".json").is_file()


def test_is_dataset_cached(expected_default_dataset_path: Path) -> None:
    assert QADataset.is_dataset_cached(str(expected_default_dataset_path))


def test_loading_cached_dataset(
    default_dataset_name: str,
    datasets_path: Path,
    expected_default_dataset_path: Path,
) -> None:
    assert expected_default_dataset_path.is_dir()
    start = time.time()
    ds_dict = QADatasetLoader.load_dataset(QADataset, default_dataset_name, cache_dir=str(datasets_path))
    end = time.time()
    max_load_time_from_cache_sec = 10
    assert (end - start) < max_load_time_from_cache_sec
    assert isinstance(ds_dict, dict)
    assert all(isinstance(key, str) and isinstance(value, QADataset) for key, value in ds_dict.items())

    for split_name in ds_dict:
        assert Path(expected_default_dataset_path, split_name + ".json").is_file()


def test_sample(default_dataset_name: str, datasets_path: Path, expected_sampled_dataset: QADataset) -> None:
    ds_dict = QADatasetLoader.load_dataset(QADataset, default_dataset_name, cache_dir=str(datasets_path))
    ds_dict = cast(dict[str, QADataset], ds_dict)
    ds = next(iter(ds_dict.values()))
    sampled_ds = ds.sample(size=10, seed=0)
    assert expected_sampled_dataset == sampled_ds
