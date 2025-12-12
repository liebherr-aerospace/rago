"""Defines an Abstract Processor class that processor raw data to generate RAG QA datasets."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, overload

from rago.dataset.processor.base import DataProcessor
from rago.dataset.rag_dataset import RAGDataset

if TYPE_CHECKING:
    from rago.data_objects import Document, EvalSample


class RAGDataProcessor[RawDataType, PreProcessedDataType](
    DataProcessor[RAGDataset, RawDataType],
):
    """Abstract base class for processing RAG datasets."""

    def process_raw_data(
        self,
        raw_data: RawDataType,
    ) -> RAGDataset | dict[str, RAGDataset]:
        """Transform raw dataset into RAGDataset.

        :param raw_data: Raw dataset content.
        :type raw_data: RawDataType
        :return: Processed RAGDataset or dict of split datasets.
        :rtype: RAGDataset | dict[str, RAGDataset]
        """
        preprocessed_data = self.preprocess_raw_data(raw_data)
        samples = self.process_samples(preprocessed_data)
        match samples:
            case list():
                corpus = self.process_corpus(preprocessed_data, is_dict_dataset=False)
                return RAGDataset(samples, corpus)
            case dict():
                corpus_dict = self.process_corpus(preprocessed_data, is_dict_dataset=True)
                if samples.keys() != corpus_dict.keys():
                    raise ValueError
                return {split_name: RAGDataset(samples[split_name], corpus_dict[split_name]) for split_name in samples}
            case _:
                raise TypeError

    @abstractmethod
    def preprocess_raw_data(self, raw_data: RawDataType) -> PreProcessedDataType:
        """Preprocess raw dataset before building samples.

        :param raw_data: Raw dataset content.
        :type raw_data: RawDataType
        :return: Preprocessed dataset.
        :rtype: PreProcessedDataType
        """

    @abstractmethod
    def process_samples(
        self,
        preprocessed_data: PreProcessedDataType,
    ) -> list[EvalSample] | dict[str, list[EvalSample]]:
        """Extract samples from preprocessed dataset.

        :param preprocessed_data: Preprocessed data.
        :type preprocessed_data: PreProcessedDataType
        :return: List of samples or dict of split samples.
        :rtype: list[EvalSample] | dict[str, list[EvalSample]]
        """

    @overload
    def process_corpus(
        self,
        preprocessed_data: PreProcessedDataType,
        is_dict_dataset: Literal[False],
    ) -> dict[str, Document]: ...

    @overload
    def process_corpus(
        self,
        preprocessed_data: PreProcessedDataType,
        is_dict_dataset: Literal[True],
    ) -> dict[str, dict[str, Document]]: ...

    def process_corpus(
        self,
        preprocessed_data: PreProcessedDataType,
        is_dict_dataset: bool,  # noqa: FBT001
    ) -> dict[str, Document] | dict[str, dict[str, Document]]:
        """Process corpus from preprocessed data.

        :param preprocessed_data: Preprocessed dataset.
        :type preprocessed_data: PreProcessedDataType
        :param is_dict_dataset: Whether corpus is dict of splits.
        :type is_dict_dataset: bool
        :return: Corpus documents (single or split).
        :rtype: dict[str, Document] | dict[str, dict[str, Document]]
        """
        return self._process_corpus(preprocessed_data, is_dict_dataset)

    @abstractmethod
    def _process_corpus(
        self,
        preprocessed_data: PreProcessedDataType,
        is_dict_dataset: bool,  # noqa: FBT001
    ) -> dict[str, Document] | dict[str, dict[str, Document]]:
        """Low-level method to implement corpus extraction.

        :param preprocessed_data: Preprocessed dataset.
        :type preprocessed_data: PreProcessedDataType
        :param is_dict_dataset: Whether corpus is dict of splits.
        :type is_dict_dataset: bool
        :return: Corpus documents (single or split).
        :rtype: dict[str, Document] | dict[str, dict[str, Document]]
        """
