"""Define a rule based evaluator of retrieval outputs."""
from rago.data_objects import Document, RetrievalOutput, RetrievedContext
from rago.eval.base import BaseIndependentEvaluator


class RuleBasedRetrievalEvaluator(BaseIndependentEvaluator[RetrievalOutput]):
    """Rule Based Evaluator of the retrieval output's retrieved context wrt the reference eval sample's context."""

    def __init__(self, max_ratio: float =1.0) -> None:
            """Instantiate context precision evaluator with the max_ratio allowed.

            :param max_ratio: The maximum ratio between the ref and retrieved context, defaults to 1 (exact match).
            :type max_ratio: float
            """
            self.max_ratio = max_ratio

    def get_num_correct_retrieved_context(
            self, retrieved_context: list[RetrievedContext], reference_contexts: list[Document]| None) -> int:
        """Return the number of correct retrieved context wrt to the refrence contexts.

        :param retrieved_context: The retrieved context to evaluate.
        :type retrieved_context: list[RetrievedContext]
        :param reference_contexts: The reference context to use.
        :type reference_contexts: list[Document]| None
        """
        if reference_contexts is None:
            return 0
        retrieved_texts = [retrieved.text for retrieved in retrieved_context]
        reference_texts = [ref.text for ref in reference_contexts]
        return sum([self.is_context_correct(retrieved, reference_texts) for retrieved in retrieved_texts])

    def is_context_correct(
                self, retrieved_context: str, reference_contexts: list[str]) -> bool:
        """Check if the retrieved context match with any of the reference context.

        :param retrieved_context: The retrieved context to evaluate
        :type retrieved_context: str
        :param reference_contexts: The list of reference texts to check against
        :type reference_contexts: list[str]
        """
        return any(
             self.check_reference_and_retrieved_match(retrieved_context, expected) for expected in reference_contexts)

    def check_reference_and_retrieved_match(self, retrieved_context: str, reference_context: str) -> bool:
        """Check wether the reference text is inside the retrieved context and that the ratio is lower than the max.

        :param retrieved_context: The retrieved context to evaluate
        :type retrieved_context: str
        :param reference_contexts: The reference text to check against
        :type reference_contexts: str
        """
        return reference_context in retrieved_context and len(retrieved_context)/len(reference_context)<= self.max_ratio

