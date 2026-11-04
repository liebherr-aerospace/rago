"""Test the relevancy evaluator."""

from rago.data_objects import Document, EvalSample, RetrievalOutput, RetrievedContext
from rago.eval import EvaluatorReference, RetrievalRecallEvaluator


def test_relevancy_score_is_valid() -> None:
    """Test that the output relevancy is correct."""
    evaluator = RetrievalRecallEvaluator()
    eval_sample = EvalSample(
        "How old is thomas",
        [Document("Thomas is 12."), Document("Thomas was born 12 years ago.")],
    )
    eval_result = evaluator.evaluate(
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12.")]),
        EvaluatorReference(eval_sample),
    )
    assert eval_result.score == 1 / 2


def test_relevancy_score_is_valid_with_no_retrieved_context() -> None:
    """Test that the output relevancy is null when the retrieved context is empty and the source context is not."""
    evaluator = RetrievalRecallEvaluator()
    eval_sample = EvalSample(
        "How old is thomas",
        [Document("Thomas is 12."), Document("Thomas was born 12 years ago.")],
    )
    eval_result = evaluator.evaluate(
        RetrievalOutput([]),
        EvaluatorReference(eval_sample),
    )
    assert eval_result.score == 0


def test_relevancy_score_is_valid_with_no_source_context() -> None:
    """Test that the output relevancy is one when the source context is empty and the retrieved context is not."""
    evaluator = RetrievalRecallEvaluator()
    eval_sample = EvalSample("How old is thomas")
    eval_result = evaluator.evaluate(
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12.")]),
        EvaluatorReference(eval_sample),
    )
    assert eval_result.score == 1


def test_relevancy_score_is_valid_with_no_context_at_all() -> None:
    """Test that the output relevancy is one with no context at all (retrieved or source)."""
    evaluator = RetrievalRecallEvaluator()
    ref = EvaluatorReference(EvalSample("How old is thomas"))
    eval_result = evaluator.evaluate(
        RetrievalOutput([]),
        ref,
    )
    assert eval_result.score == 1


def test_relevancy_pairwise_score_equal_individual_score() -> None:
    """Test that using pairwise with relevancy is equal to apply independently two direct evaluations."""
    evaluator = RetrievalRecallEvaluator()
    ref = EvaluatorReference(EvalSample(
        "How old is thomas",
        [Document("Thomas is 12."), Document("Thomas was born 12 years ago.")],
    ))

    outputs = [
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12.")]),
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 13.")]),
    ]

    results = [evaluator.evaluate(o, ref) for o in outputs]

    n_wise_results = evaluator.evaluate_n_wise(outputs, ref)
    assert len(results) == len(n_wise_results)
    assert all(r.score == n_r.score for r, n_r in zip(results, n_wise_results, strict=False))
