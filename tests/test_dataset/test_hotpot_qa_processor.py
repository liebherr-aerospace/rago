"""Test the hotpot qa processor."""

from typing import Any, cast

import numpy as np
import pytest
from datasets import DatasetDict

from rago.data_objects import Document, EvalSample


def test_raw_data_correct_number_splits(
    raw_downloaded_data: DatasetDict,
) -> None:
    expected_number_splits = 2
    assert len(raw_downloaded_data) == expected_number_splits


def test_raw_data_correct_columns(
    raw_downloaded_data: DatasetDict,
    raw_data_expected_columns: list[str],
) -> None:
    raw_data_cols = raw_downloaded_data.column_names
    for cols in raw_data_cols.values():
        if cols != raw_data_expected_columns:
            pytest.fail(reason="The columns of the hotpot qa raw data are incorrect.")
    assert True


def test_raw_data_split_size(
    raw_downloaded_data: DatasetDict,
    hotpot_qa_expected_split_sizes: list[float],
) -> None:
    for ds_split, expected_size in zip(raw_downloaded_data.values(), hotpot_qa_expected_split_sizes):
        if len(ds_split) != expected_size:
            pytest.fail(reason="The split sizes of the hotpot qa raw data are incorrect.")
    assert True


def test_raw_data_supporting_facts_in_context(
    raw_downloaded_data: DatasetDict,
) -> None:
    """Test that downloaded data supported facts are present in the context of each samples ."""
    for split in raw_downloaded_data:
        for el in raw_downloaded_data[split]:
            el = cast(dict[str, Any], el)
            for title in el["supporting_facts"]["title"]:
                if title not in el["context"]["title"]:
                    pytest.fail(reason="The supporting facts are not in context of the hotpot qa raw data.")
    assert True


def test_pre_processed_data_split_size(
    preprocessed_data: DatasetDict,
    hotpot_qa_expected_split_sizes: list[float],
) -> None:
    for ds_split, expected_size in zip(preprocessed_data.values(), hotpot_qa_expected_split_sizes):
        if len(ds_split) != expected_size:
            pytest.fail(reason="The split sizes of the hotpot qa preprocessed data are incorrect.")
    assert True


def test_preprocessed_data_contains_correct_columns(
    preprocessed_data: DatasetDict,
    pre_processed_expected_columns: list[str],
) -> None:
    pre_processed_data_cols = preprocessed_data.column_names
    for cols in pre_processed_data_cols.values():
        if cols != pre_processed_expected_columns:
            pytest.fail(reason="The columns of the hotpot qa preprocessed data are incorrect.")
    assert True


def test_preprocessed_data_correct_ref_context(
    raw_downloaded_data: DatasetDict,
    preprocessed_data: DatasetDict,
) -> None:
    """Tests that after preprocessing, each element meets certain conditions.

    - the ref_context field of each element still contains all the docs of the corresponding element in the raw data.
    - each doc in the ref_context is a dict with the fields: "id", "text", "metadata".
    - that metadata is a dict and only contains the "title".
    """
    for data_split, raw_split in zip(preprocessed_data.values(), raw_downloaded_data.values()):
        data_split = cast(DatasetDict, data_split)
        raw_split = cast(DatasetDict, raw_split)
        for ref_contexts, raw_context in zip(data_split["context"], raw_split["context"]):
            ref_contexts = cast(list, ref_contexts)
            raw_context = cast(dict[str, Any], raw_context)
            if len(ref_contexts) != len(raw_context["title"]):
                pytest.fail(reason="different number of context element in preprocess in raw data.")
            for context in ref_contexts:
                assert "id" in context
                assert "text" in context
                assert "metadata" in context
                assert len(context["metadata"]) == 1
                assert "title" in context["metadata"]


def test_processed_data_correct_num_samples(
    processed_samples: dict[str, list[EvalSample]],
    hotpot_qa_expected_split_sizes: list[float],
) -> None:
    for samples, expected_size in zip(processed_samples.values(), hotpot_qa_expected_split_sizes):
        if len(samples) != expected_size:
            pytest.fail(reason="different number of context element in preprocess in raw data.")


def test_processed_data_samples_correct_ref_context(
    processed_samples: dict[str, list[EvalSample]],
    preprocessed_data: DatasetDict,
) -> None:
    """Test that each sample's ref_context contains only docs whose title where in the supporting_facts of raw_data.

    Interestingly, the supporting facts are on the sentence level.
    So one document might be mentioned multiple times in supporting facts title list.
    """
    for samples_split, preprocessed_split in zip(processed_samples.values(), preprocessed_data.values()):
        for sample, preprocessed in zip(samples_split, preprocessed_split):
            if sample.context is None:
                pytest.fail(reason="sample's context is None")
            sample.context = cast(list[Document], sample.context)
            titles_sample = [context.id for context in sample.context]
            assert len(np.unique(titles_sample)) == len(titles_sample)
            assert len(titles_sample) == len(np.unique(preprocessed["supporting_facts"]["title"]))
            assert all(title in preprocessed["supporting_facts"]["title"] for title in titles_sample)


def test_processed_data_samples_correct_metadata(processed_samples: dict[str, list[EvalSample]]) -> None:
    expected_number_of_keys_in_metadata = 2
    for samples_split in processed_samples.values():
        for sample in samples_split:
            assert len(sample.metadata) == expected_number_of_keys_in_metadata
            assert "type" in sample.metadata
            assert "level" in sample.metadata


def test__processed_data_samples_correct_ref_answer(processed_samples: dict[str, list[EvalSample]]) -> None:
    for samples_split in processed_samples.values():
        for sample in samples_split:
            assert sample.reference_answer is not None
