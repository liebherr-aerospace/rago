"""Define a evaluator that evaluates the similarity between generated output and reference answer."""

from typing import ClassVar

from sentence_transformers import SentenceTransformer

from rago.data_objects import EvalSample, GenerationOutput, Metric
from rago.eval.base import BaseIndependentEvaluator, EvaluatorReference, register_evaluator


@register_evaluator("similarity")
class SimilarityScore(BaseIndependentEvaluator[GenerationOutput]):
    """Defines a evaluator that evaluates the similarity between generated answer and reference answer."""

    metrics: ClassVar[list[str]] = ["similarity"]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    ) -> None:
        """Create an instance of the SimilarityScore class.

        :param model_name: Name of the model used to compute the similarity.
        :type model_name: str
        """
        self.model = SentenceTransformer(model_name)

    def evaluate(self, candidate_output: GenerationOutput, reference: EvaluatorReference) -> Metric:
        """Evaluate the candidate generation output using the answer in reference's eval_sample information.

        :param candidate_output: The generation output to the query.
        :type candidate_output: GenerationOutput
        :param reference: The reference used by the evaluator to evaluate the candidate.
        :type reference: EvaluatorReference
        :return: The similarity metric.
        :rtype: EvalOutputType
        """
        references = self.get_targets(reference.eval_sample)
        embeddings_1 = self.model.encode(candidate_output.answer, normalize_embeddings=True)
        embeddings_2 = self.model.encode(references, normalize_embeddings=True)
        score = (embeddings_1 * embeddings_2).sum().item()
        return Metric(score)

    def get_targets(self, eval_samples: EvalSample) -> list[str]:
        """Get the eval samples ref answer.

        :param eval_samples: _description_
        :type eval_samples: EvalSample | Sequence[EvalSample]
        :raises ValueError: _description_
        :return: _description_
        :rtype: list[str]
        """
        if eval_samples.reference_answer is None:
            raise ValueError

        return [eval_samples.reference_answer]
