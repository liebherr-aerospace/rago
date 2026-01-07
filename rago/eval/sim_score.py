"""Define a evaluator that evaluates the similarity between generated answer and reference answer."""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

from rago.data_objects import EvalSample, Metric, RAGOutput

from .base import BaseEvaluator, register_evaluator


@register_evaluator("similarity")
class SimilarityScore(BaseEvaluator[RAGOutput]):
    """Defines a evaluator that evaluates the similarity between generated answer and reference answer."""

    metrics: ClassVar[list[str]] = ["similarity"]

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """Create an instance of the SimilarityScore class.

        :param model_name: Name of the model used to compute the similarity.
        :type model_name: str
        """
        self.model = SentenceTransformer(model_name)

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the candidate RAG output using the eval_sample information.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: OutputType
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation result.
        :rtype: EvalOutputType
        """
        predictions = self.get_predictions(candidate_output)
        if eval_sample.reference_answer is None:
            raise ValueError
        references = self.get_targets(eval_sample)
        embeddings_1 = self.model.encode(predictions, normalize_embeddings=True)
        embeddings_2 = self.model.encode(references, normalize_embeddings=True)
        scores = (embeddings_1 * embeddings_2).sum(axis=1)

        mean_scores = np.mean(scores).item()
        return {
            "similarity": Metric(mean_scores),
        }

    def get_predictions(self, outputs: RAGOutput) -> list[str]:
        """Get the rag outputs answer.

        :param outputs: _description_
        :type outputs: RAGOutput | Sequence[RAGOutput]
        :return: _description_
        :rtype: list[str]
        """
        if outputs.answer is None:
            raise ValueError
        return [outputs.answer]

    def get_targets(self, eval_samples: EvalSample | Sequence[EvalSample]) -> list[str]:
        """Get the eval samples ref answer.

        :param eval_samples: _description_
        :type eval_samples: EvalSample | Sequence[EvalSample]
        :raises ValueError: _description_
        :return: _description_
        :rtype: list[str]
        """
        eval_samples = eval_samples if isinstance(eval_samples, Sequence) else [eval_samples]
        references: list[str] = []
        for sample in eval_samples:
            if sample.reference_answer is None:
                raise ValueError
            references.append(sample.reference_answer)
        return references
