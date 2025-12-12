"""Define rag QA dataset dataclass holding QA eval samples and a corpus of documents to use to answer the queries."""

from __future__ import annotations

import copy
import os
import random
from typing import TYPE_CHECKING, Any, Optional, Self, TypeVar

from pydantic.dataclasses import dataclass

from rago.data_objects import Document, EvalSample  # noqa: TC001
from rago.utils import PATH_DEFAULT_DATA_DIR

from .base import QADataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


DEFAULT_CACHE_DIR = str(os.environ.get("DEFAULT_CACHE_DIR", PATH_DEFAULT_DATA_DIR))


RawDataType = TypeVar("RawDataType")
PreProcessedDataType = TypeVar("PreProcessedDataType")

DatasetType = TypeVar("DatasetType", bound="RAGDataset")


@dataclass(repr=False)
class RAGDataset(QADataset):
    """Dataset class for Retrieval-Augmented Generation. Extends QADataset with a document corpus.

    :param samples: List of evaluation samples.
    :type samples: list[EvalSample]
    :param corpus: Mapping of document IDs to Documents.
    :type corpus: dict[str, Document]
    """

    corpus: dict[str, Document]

    @property
    def corpus_docs(
        self,
    ) -> list[Document]:
        """Get list of documents in the corpus.

        :return: List of documents.
        :rtype: list[Document]
        """
        return list(self.corpus.values())

    def __str__(self) -> str:
        """Return Basic info about RAG dataset in a string."""
        return (
            super().__str__()
            + f"""Documents:
number: {len(self.corpus)}
attr: {", ".join(list(next(iter(self.corpus.values())).__annotations__.keys()))}
"""
        )

    def add_eval_samples(self, samples: list[EvalSample]) -> None:
        """Add eval samples to the dataset.

        :param samples: The samples to add to the dataset.
        :type samples: list[EvalSample]
        """
        super().__init__(samples)
        for sample in samples:
            if sample.context is not None:
                for context in sample.context:
                    if context.id not in self.corpus:
                        self.corpus[context.id] = context

    def sample(
        self,
        size: float,
        seed: Optional[int] = None,
        num_distractor_documents_per_sample: int = 10,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> RAGDataset:
        """Return a sampled subset of the dataset with corpus.

        :param size: Fraction or number of samples.
        :type size: float | int
        :param num_distractor_documents_per_sample: Number of distractor docs per sample.
        :type num_distractor_documents_per_sample: int
        :param kwargs: Extra arguments.
        :type kwargs: Any
        :return: Sampled RAG dataset.
        :rtype: RAGDataset
        """
        if seed is not None:
            random.seed(seed)
        samples = self.get_num_samples(size)
        corpus = self.sample_corpus(samples, num_distractor_documents_per_sample)
        return self.__class__(samples=samples, corpus=corpus)

    def sample_corpus(
        self,
        sampled_samples: Sequence[EvalSample],
        num_distractor_documents_per_sample: int,
    ) -> dict[str, Document]:
        """Build a corpus including references and distractors.

        :param sampled_samples: Subset of evaluation samples.
        :type sampled_samples: Sequence[EvalSample]
        :param num_distractor_documents_per_sample: Number of distractors per sample.
        :type num_distractor_documents_per_sample: int
        :return: Subset corpus.
        :rtype: dict[str, Document]
        """
        ref_corpus = self.get_ref_corpus_from_samples(sampled_samples)
        distractor_corpus = self.get_distractor_corpus(
            list(ref_corpus.keys()),
            num_distractor=num_distractor_documents_per_sample * len(sampled_samples),
        )

        return ref_corpus | distractor_corpus

    def get_ref_corpus_from_samples(
        self,
        sampled_samples: Sequence[EvalSample],
    ) -> dict[str, Document]:
        """Extract reference corpus from samples.

        :param sampled_samples: Subset of evaluation samples.
        :type sampled_samples: Sequence[EvalSample]
        :return: Reference corpus documents.
        :rtype: dict[str, Document]
        """
        ids_refs_corpus = [
            ([context.id for context in sample.context] if sample.context is not None else [])
            for sample in sampled_samples
        ]
        sampled_corpus_refs: dict[str, Document] = {}
        for ids_refs_sample in ids_refs_corpus:
            for id_ref in ids_refs_sample:
                if id_ref not in self.corpus:
                    raise KeyError
                if id_ref not in sampled_corpus_refs:
                    sampled_corpus_refs[id_ref] = self.corpus[id_ref]
        return sampled_corpus_refs

    def get_distractor_corpus(
        self,
        ids_ref_corpus: Sequence[str],
        num_distractor: int,
    ) -> dict[str, Document]:
        """Select distractor documents not in reference set.

        :param ids_ref_corpus: IDs of reference documents.
        :type ids_ref_corpus: Sequence[str]
        :param num_distractor: Number of distractors to select.
        :type num_distractor: int
        :return: Distractor documents.
        :rtype: dict[str, Document]
        """
        corpus_ids = list(self.corpus.keys())
        ids_distractor_corpus = [idx for idx in corpus_ids if idx not in ids_ref_corpus]
        random.shuffle(ids_distractor_corpus)
        sampled_ids_distractor = ids_distractor_corpus[:num_distractor]
        return {key: self.corpus[key] for key in sampled_ids_distractor}

    def map_corpus(self, map_function: Callable[[Document], Document]) -> RAGDataset:
        """Apply `map_function` to all the documents in the corpus and return the resulting new dataset.

        :param map_function: The function to apply to every documents in the corpus.
        :type map_function: Callable[[Document], Document]
        :return: The resulting new dataset with modified corpus.
        :rtype: RAGDataset
        """
        n_ds = copy.deepcopy(self)
        for id_el, element in self.corpus.items():
            n_ds.corpus[id_el] = map_function(element)
        return n_ds

    def _get_spitted_corpus(
        self,
        splits: Sequence[float],
        splitted_samples: dict[str, list[EvalSample]],
    ) -> dict[str, dict[str, Document]]:
        ref_corpus: dict[str, dict[str, Document]] = {
            name: self.get_ref_corpus_from_samples(samples) for name, samples in splitted_samples.items()
        }
        id_refs_corpus = [key for corpus in ref_corpus.values() for key in list(corpus.keys())]
        distractor_corpus: dict[str, Document] = {
            id_doc: doc for id_doc, doc in self.corpus.items() if id_doc not in id_refs_corpus
        }
        distractor_corpus_ids = list(distractor_corpus.keys())
        random.shuffle(distractor_corpus_ids)
        corpus_split: dict[str, dict[str, Document]] = {}
        id_start = 0
        for id_split, split_name in enumerate(splitted_samples):
            r_corpus = ref_corpus[split_name]
            id_end = (
                len(distractor_corpus_ids)
                if id_split == len(splits)
                else id_start + int(len(distractor_corpus_ids) * splits[id_split])
            )
            d_corpus = {key: distractor_corpus[key] for key in distractor_corpus_ids[id_start:id_end]}
            corpus_split[split_name] = r_corpus | d_corpus
            id_start = id_end
        return corpus_split

    def split_dataset(
        self,
        splits: Sequence[float],
        split_names: Optional[list[str]] = None,
        seed: Optional[int] = None,
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
        splitted_corpus = self._get_spitted_corpus(splits, splitted_samples)
        dataset_splits = {
            split_name: self.__class__(
                samples=splitted_samples[split_name],
                corpus=splitted_corpus[split_name],
            )
            for split_name in splitted_samples
        }
        return dataset_splits
