"""Defines a llama-index post-processor wrapper around llama-index post-processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.model.constructors.llama_index.post_processor_factory import NodePostProcessorFactory
from rago.model.wrapper.context_post_processor.base import ContextPostProcessor

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

    from rago.model.configs.post_processor_config.llama_index import NodePostProcessorConfig

from rago.data_objects import RetrievedContext


class LLamaIndexContextPostProcessorWrapper(ContextPostProcessor):
    """A llama-index post-processor wrapper around llama-index post-processors.

    Can build build from its configuration with the build class method
    Can be used to filter and augment a set of nodes based on the query they were retrieved for.

    :param llama_index_post_processors: The llama-index post-processors to wrap around.
    :type llama_index_post_processors: list[BaseNodePostprocessor]
    """

    def __init__(self, llama_index_post_processors: list[BaseNodePostprocessor]) -> None:
        """Instantiate the wrapper from the llama-index post-processor.

        :param llama_index_post_processors: The llama-index post-processor to wrap.
        :type llama_index_post_processors: list[BaseNodePostprocessor]
        """
        self.llama_index_post_processors = llama_index_post_processors

    @classmethod
    def make(
        cls,
        nodes_post_processors_configs: list[NodePostProcessorConfig],
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> LLamaIndexContextPostProcessorWrapper:
        """Build the post-processor from its config.

        :param nodes_post_processors_dict_configs: Config of the post-processor to use.
        :type nodes_post_processors_dict_configs: dict
        :param encoder: encoder used by the post-processor for filtering.
        :type encoder: BaseEmbedding
        :param llm: LLM used by the post_processors
        :type llm: LLM
        :return: The built post-processor.
        :rtype: LLamaIndexContextPostProcessorWrapper
        """
        post_processors = [
            NodePostProcessorFactory.make(config, encoder, llm) for config in nodes_post_processors_configs
        ]
        return cls(post_processors)

    def post_process_context(self, query: str, context: list[RetrievedContext]) -> list[RetrievedContext]:
        """Post-process (i.e filter or augment) a list of retrieved context based on their query.

        :param context: The context to post-process.
        :type context: list[RetrievedContext]
        :param query: The query the retrieved context is for.
        :type query: str
        :return: A refined retrieved context for the query.
        :rtype: list[RetrievedContext]
        """
        llama_index_context = [RetrievedContext.get_llama_index_with_score(node) for node in context]
        # Taken from the apply_post_processor from RetrieverQueryEngine class in llamaIndex
        # This means the architecture is still not perfect
        for context_post_processor in self.llama_index_post_processors:
            llama_index_context = context_post_processor.postprocess_nodes(
                llama_index_context,
                query_str=query,
            )

        return [RetrievedContext.from_llama_index_node_with_score(node) for node in llama_index_context]
