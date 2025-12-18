"""Define the fixtures shared by multiple test on dataset related objects."""

from pathlib import Path

import pytest
from datasets import DatasetDict

from rago.data_objects import EvalSample
from rago.dataset import HotPotQAProcessor


@pytest.fixture
def default_dataset_name() -> str:
    """Name of the default dataset to use."""
    return "hotpot_qa"


@pytest.fixture(scope="session")
def datasets_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary path to save the datasets related object to during testing.

    :param tmp_path_factory: Factory used to generated the path
    :type tmp_path_factory: pytest.TempPathFactory
    :return: Temporary path to save the datasets related object to during testing.
    :rtype: Path
    """
    return tmp_path_factory.mktemp("datasets")


@pytest.fixture
def hotpot_qa_processor() -> HotPotQAProcessor:
    """Define the hotpot qa processor used to generate the dataset."""
    processor = HotPotQAProcessor("hotpot_qa", "assests/tmp")
    return processor


@pytest.fixture
def raw_downloaded_data(hotpot_qa_processor: HotPotQAProcessor) -> DatasetDict:
    """Generate the raw data of the hotpot qa dataset with the hotpot qa processor.

    :param hotpot_qa_processor: The HotPot QA processor
    :type hotpot_qa_processor: HotPotQAProcessor
    """
    ds_dict = hotpot_qa_processor.download_raw_data()
    return ds_dict


@pytest.fixture
def preprocessed_data(hotpot_qa_processor: HotPotQAProcessor, raw_downloaded_data: DatasetDict) -> DatasetDict:
    """Preprocess the raw data of the hotpot qa dataset.

    :param hotpot_qa_processor: The processor for the hotpot qa dataset.
    :type hotpot_qa_processor: HotPotQAProcessor
    :param raw_downloaded_data: The raw data of the hotpot qa dataset.
    :type raw_downloaded_data: DatasetDict
    """
    ds_dict = hotpot_qa_processor.preprocess_raw_data(raw_downloaded_data)
    return ds_dict


@pytest.fixture
def processed_samples(
    hotpot_qa_processor: HotPotQAProcessor,
    preprocessed_data: DatasetDict,
) -> dict[str, list[EvalSample]]:
    """Get the samples from HotPot QA preprocessed data.

    :param hotpot_qa_processor: The processor for HotPotQA
    :type hotpot_qa_processor: HotPotQAProcessor
    :param preprocessed_data: The HotPot QA preprocessed data.
    :type preprocessed_data: DatasetDict
    """
    samples = hotpot_qa_processor.process_samples(preprocessed_data)
    return samples


@pytest.fixture
def raw_data_expected_columns() -> list[str]:
    """Get the HotPotQA raw data's expected columns."""
    return [
        "id",
        "question",
        "answer",
        "type",
        "level",
        "supporting_facts",
        "context",
    ]


@pytest.fixture
def pre_processed_expected_columns() -> list[str]:
    """Get the HotPotQA preprocessed data's expected columns."""
    return [
        "query",
        "reference_answer",
        "type",
        "level",
        "supporting_facts",
        "context",
    ]


@pytest.fixture
def hotpot_qa_expected_split_sizes() -> list[float]:
    """Get the HotPotQA expected split sizes."""
    return [90447, 7405]
