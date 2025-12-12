"""Define a Processor for processing CRAG dataset."""

import bz2
import json
from collections.abc import Iterator
from typing import Any, ClassVar, cast

import wget
from datasets import Dataset

from rago.data_objects import Document, EvalSample
from rago.dataset.processor.base import register_processor
from rago.dataset.processor.dataset_names import DatasetNames

from .base import RAGDataProcessor


@register_processor(DatasetNames.CRAG)
class CRAGProcessor(RAGDataProcessor[Dataset, Dataset]):
    """Define a Processor for processing the CRAG dataset."""

    download_url: str = (
        "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"
    )
    metadata_keys: ClassVar[list[str]] = [
        "interaction_id",
        "query_time",
        "domain",
        "question_type",
        "static_or_dynamic",
        "split",
    ]

    def download_raw_data(self) -> Dataset:
        """Download raw CRAG dataset from HuggingFace.

        :return: Raw dataset content as a `Dataset`.
        :rtype: Dataset
        """
        path_raw_data = self.path_dataset[::-1].split(".", maxsplit=1)[-1][::-1]
        path_raw_data += "_raw"
        wget.download(self.download_url, path_raw_data)
        ds = Dataset.from_generator(lambda: self.gen_rows(path_raw_data))
        ds = cast(Dataset, ds)
        ds.cleanup_cache_files()
        return ds

    def preprocess_raw_data(self, raw_data: Dataset) -> Dataset:
        """Preprocess raw CRAG dataset before building samples.

        :param raw_data: Raw dataset content.
        :type raw_data: Dataset
        :return: Preprocessed HotPotQA dataset.
        :rtype: Dataset
        """
        return raw_data.map(self.get_ref_context, remove_columns="search_results", load_from_cache_file=False)

    def process_samples(self, preprocessed_data: Dataset) -> list[EvalSample]:
        """Extract samples from preprocessed CRAG dataset.

        :param preprocessed_data: Preprocessed HotPotQA dataset.
        :type preprocessed_data: preprocessed_data
        :return: List of samples or dict of split samples.
        :rtype: list[EvalSample]
        """
        dataset = preprocessed_data.filter(lambda example: len(example["alt_ans"]) == 0, load_from_cache_file=False)
        dataset = dataset.remove_columns("alt_ans")
        dataset = dataset.filter(lambda example: "i don't know" not in example["answer"], load_from_cache_file=False)
        dataset = dataset.filter(lambda example: len(example["context"]) > 0, load_from_cache_file=False)
        dataset = dataset.rename_column("answer", "reference_answer")
        dataset = dataset.map(self.add_metadata, remove_columns=self.metadata_keys)
        samples: list[EvalSample] = []
        for sample in dataset:
            if not isinstance(sample, dict):
                raise TypeError
            samples.append(EvalSample.load_from_dict(sample))
        return samples

    def _process_corpus(self, preprocessed_data: Dataset, is_dict_dataset: bool) -> dict[str, Document]:  # noqa: FBT001, ARG002
        """Low-level method to implement corpus extraction from the raw CRAG dataset.

        :param preprocessed_data: Preprocessed dataset.
        :type preprocessed_data: PreProcessedDataType
        :param is_dict_dataset: Whether corpus is dict of splits.
        :type is_dict_dataset: bool
        :return: Corpus documents (single or split).
        :rtype: dict[str, Document]
        """
        corpus: dict[str, Document] = {}
        for sample in preprocessed_data:
            if not isinstance(sample, dict):
                raise TypeError
            for el in sample["context"]:
                if el["id"] not in corpus:
                    corpus[el["id"]] = Document(**el)
        return corpus

    def gen_rows(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Generate rows from a bz2-compressed JSONL file, converting answers to strings.

        :param file_path: Path to the bz2-compressed JSONL file.
        :type file_path: str
        :yield: Each row as a dictionary with "answer" and "alt_ans" converted to strings.
        :rtype: Iterator[dict[str, Any]]
        """
        with bz2.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                example["alt_ans"] = self.int2str(example["alt_ans"])
                example["answer"] = str(example["answer"])
                yield example

    def int2str(self, labels: list) -> list[str]:
        """Convert labels to str.

        :param labels: The labels to convert to string.
        :type labels: list
        :return: The labels converted to string.
        :rtype: list[str]
        """
        fixed_labels = []
        for label in labels:
            fixed_label = str(label) if not isinstance(label, str) else label
            fixed_labels.append(fixed_label)
        return fixed_labels

    def add_answers(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Make `reference_answer` in input sample dict by concatenating `answer` and `alt_ans`.

        :param sample: The sample to modify.
        :type sample: dict[str, Any]
        :return: The modified sample.
        :rtype: dict[str, Any]
        """
        return {"reference_answer": [sample["answer"]] + sample["alt_ans"]}

    def get_ref_context(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Add `reference_context` in input sample dict by processing `search_results`.

        :param sample: The sample to `reference_context` to.
        :type sample: dict[str, Any]
        :return: The sample with `reference_context`.
        :rtype: dict[str, Any]
        """
        reference_documents = []
        for page in sample["search_results"]:
            if len(page["page_snippet"]) > 0:
                new_doc = {
                    "text": page["page_snippet"],
                    "id": str((page["page_url"], page["page_last_modified"])),
                    "metadata": {"title": page["page_name"], "date": page["page_last_modified"]},
                }
                reference_documents.append(new_doc)
        return {"context": reference_documents}

    def add_metadata(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Add `metadata` in input sample dict by taking all the keys in `self.metadata_keys`.

        :param sample: The sample to `metadata` to.
        :type sample: dict[str, Any]
        :return: The sample with `metadata`.
        :rtype: dict[str, Any]
        """
        metadata = {}
        for key in self.metadata_keys:
            metadata[key] = str(sample[key])
        sample["metadata"] = metadata
        return sample
