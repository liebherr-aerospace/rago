"""Defines a Reader builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    CompactAndRefine,
    Refine,
    SimpleSummarize,
    TreeSummarize,
)

from rago.model.configs.reader_config.llama_index import LLamaIndexReaderConfig  # noqa: TC001
from rago.model.constructors.llama_index.llm_factory import LLamaIndexLLMFactory

if TYPE_CHECKING:
    from llama_index.core.llms import LLM


class ReaderFactory:
    """A reader builder."""

    @staticmethod
    def make(
        config: LLamaIndexReaderConfig,
        text_qa_string_template: str,
        llm: Optional[LLM] = None,
    ) -> BaseSynthesizer:
        """Build a Reader.

        :param config: The configuration of the reader to build.
        :type config: dict
        :param llm: The LLM of the reader to build.
        :type llm: LLM
        :param text_qa_string_template: The prompt template used by the reader.
        :type text_qa_string_template: str
        :return: The built reader.
        :rtype: BaseSynthesizer
        """
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LLamaIndexLLMFactory.make(config.llm)
        match config.type:
            case "Refine":
                return Refine(
                    llm=llm,
                    text_qa_template=PromptTemplate(text_qa_string_template),
                )
            case "CompactAndRefine":
                return CompactAndRefine(
                    llm=llm,
                    text_qa_template=PromptTemplate(text_qa_string_template),
                )
            case "TreeSummarize":
                return TreeSummarize(
                    llm=llm,
                )
            case "SimpleSummarize":
                return SimpleSummarize(
                    llm=llm,
                    text_qa_template=PromptTemplate(text_qa_string_template),
                )
            case _:
                raise ValueError(config.type)
