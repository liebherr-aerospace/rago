"""Define the tests for the rag dataset class."""

from pathlib import Path
from typing import cast

import pytest

from rago.dataset import QADatasetLoader, RAGDataset


@pytest.fixture
def expected_sampled_rag_dataset() -> RAGDataset:
    return RAGDataset.load_from_json("./tests/assets/expected_sampled_rag_dataset.json", is_list=False)


def test_sample(default_dataset_name: str, datasets_path: Path, expected_sampled_rag_dataset: RAGDataset) -> None:
    """Carry multiple tests on the sampled dataset.

    Since I have already verified the expected_dataset is correct it checks
    that the generated dataset is identic (seed works)
    and indirectly that the proprieties of the generated dataset are correct.
    """
    ds_dict = QADatasetLoader.load_dataset(RAGDataset, default_dataset_name, cache_dir=str(datasets_path))
    ds_dict = cast(dict[str, RAGDataset], ds_dict)
    ds = next(iter(ds_dict.values()))
    sampled_ds = ds.sample(size=10, seed=0)
    assert expected_sampled_rag_dataset == sampled_ds
