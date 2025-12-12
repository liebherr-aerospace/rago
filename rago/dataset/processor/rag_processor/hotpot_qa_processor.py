"""Define a Processor for processing HotPotQA dataset."""

from typing import Any, ClassVar, cast

from datasets import DatasetDict, load_dataset

from rago.data_objects import Document, EvalSample
from rago.dataset.processor.base import register_processor
from rago.dataset.processor.dataset_names import DatasetNames

from .base import RAGDataProcessor


@register_processor(DatasetNames.HOTPOTQA)
class HotPotQAProcessor(RAGDataProcessor[DatasetDict, DatasetDict]):
    """Define a Processor for processing the HotPotQA dataset from the HuggingFace dataset."""

    huggingface_path: str = "hotpotqa/hotpot_qa"
    metadata_keys: ClassVar[list[str]] = [
        "interaction_id",
        "query_time",
        "domain",
        "question_type",
        "static_or_dynamic",
        "split",
    ]

    def download_raw_data(self) -> DatasetDict:
        """Download raw HotPotQA dataset from HuggingFace.

        :return: Raw dataset content as a `DatasetDict`.
        :rtype: DatasetDict
        """
        ds = load_dataset(self.huggingface_path, "distractor")
        return cast(DatasetDict, ds)

    def preprocess_raw_data(self, raw_data: DatasetDict) -> DatasetDict:
        """Preprocess raw HotPotQA dataset before building samples.

        :param raw_data: Raw dataset content.
        :type raw_data: DatasetDict
        :return: Preprocessed HotPotQA dataset.
        :rtype: DatasetDict
        """
        ds = raw_data.remove_columns("id")
        ds = ds.rename_column("question", "query")
        ds = ds.rename_column("answer", "reference_answer")
        ds = ds.map(self.get_ref_context, load_from_cache_file=False)
        return ds

    def process_samples(self, preprocessed_data: DatasetDict) -> dict[str, list[EvalSample]]:
        """Extract samples from preprocessed HotPotQA dataset.

        :param preprocessed_data: Preprocessed HotPotQA dataset.
        :type preprocessed_data: preprocessed_data
        :return: List of samples or dict of split samples.
        :rtype: dict[str, list[EvalSample]]
        """
        ds = preprocessed_data.map(
            self.filter_ref_context,
            remove_columns=["supporting_facts"],
            load_from_cache_file=False,
        )
        ds = ds.map(self.get_metadata, remove_columns=["type", "level"], load_from_cache_file=False)
        samples: dict[str, list[EvalSample]] = {}
        for split_name in ds:
            samples[split_name] = []
            for sample in ds[split_name]:
                if not isinstance(sample, dict):
                    raise TypeError
                samples[split_name].append(EvalSample.load_from_dict(sample))
        return samples

    def _process_corpus(
        self,
        preprocessed_data: DatasetDict,
        is_dict_dataset: bool,  # noqa: FBT001, ARG002
    ) -> dict[str, dict[str, Document]]:
        """Low-level method to implement corpus extraction from the HotPotQA dataset.

        :param preprocessed_data: Preprocessed dataset.
        :type preprocessed_data: PreProcessedDataType
        :param is_dict_dataset: Whether corpus is dict of splits.
        :type is_dict_dataset: bool
        :return: Corpus documents (single or split).
        :rtype: dict[str, dict[str, Document]]
        """
        corpus: dict[str, dict[str, Document]] = {}
        for split_name in preprocessed_data:
            corpus[split_name] = {}
            for sample in preprocessed_data[split_name]:
                if not isinstance(sample, dict):
                    raise TypeError
                for el in sample["context"]:
                    if el["metadata"]["title"] not in corpus:
                        corpus[split_name][el["metadata"]["title"]] = Document(**el)
        return {split_name: corpus[split_name] for split_name in corpus}

    def get_ref_context(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Add the context as dict from a sample of the HotPotQA dataset.

        :param examples: The sample we want to add the reference context to.
        :type examples: dict[str, Any]
        :return: The sample with the reference context to.
        :rtype: dict[str, Any]
        """
        ref_context = [
            {
                "id": examples["context"]["title"][idx],
                "text": "\n".join(examples["context"]["sentences"][idx]),
                "metadata": {"title": examples["context"]["title"][idx]},
            }
            for idx in range(len(examples["context"]["title"]))
        ]

        examples["context"] = ref_context
        return examples

    def filter_ref_context(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Remove context that are not reference.

        :param examples:  The sample we want to remove context from.
        :type examples: dict[str, Any]
        :return: The sample with the spurious context removed.
        :rtype: dict[str, Any]
        """
        titles_ref_context = examples["supporting_facts"]["title"]
        examples["context"] = [context for context in examples["context"] if (context["id"] in titles_ref_context)]
        return examples

    def get_metadata(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Add metadata to a sample from the HotPotQA dataset.

        :param examples: The examples we want to add metadata to.
        :type examples: dict[str, Any]
        :return: The sample with the metadata attribute.
        :rtype: dict[str, Any]
        """
        examples["metadata"] = {"type": examples["type"], "level": examples["level"]}
        return examples
