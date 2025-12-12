# Evaluators

**Evaluators** are the core components that measure the quality of RAG system outputs by comparing generated answers against expected results or ground truth. RAGO provides multiple evaluation strategies—from simple metric-based evaluators (BERTScore, relevancy) to LLM-as-judge evaluators—allowing you to choose the most appropriate method for your use case and optimize toward specific quality criteria.


## Base Evaluators

Evaluators are used to evaluate any `DataObject`. For evaluation they take as input one RAG output `evaluation` or two `pairwise_evaluation` and output a result as a either a single `Metric` or  a `dict[str, Metric]` object. A single Evaluator might return one or many metric.
To get all the available evaluators simply run:

```python
from rago.eval.base import BaseEvaluator

BaseEvaluator.list_available_evaluators()
```
To load a specific evaluator simply call `load` from the BaseEvaluator with the evaluator name:
```python
from rago.eval.base import BaseEvaluator

evaluator = BaseEvaluator.load('bert_score')
```

## Relevancy Evaluator

The relevancy evaluator evaluates only the retrieved context (obtained by the RAG based on the query) compared to the source context (used by the generator to generate query).
More precisely it returns the proportion of the source context that is in the retrieved context.
Therefore using it requires an `EvalSample` with `context`:

The relevancy evaluator does not require any argument to be instantiated:
```python
from rago.data_objects import EvalSample
from rago.eval import RelevancyEvaluator

eval_sample = EvalSample("How old is thomas", context = ["Thomas is 12.", "Thomas was born 12 years ago."])

evaluator = RelevancyEvaluator()
```
Below is an example of the relevancy evaluator usage:
```python
from rago.utils import Document, RAGOutput

eval_result = evaluator.evaluate(
    RAGOutput(retrieved_documents=[Document("Thomas is 12.")]),
    eval_sample,
)
eval_result["relevancy"].score #Output: 0.5
```
The Output in this case is 0.5 because the output has retrieved 1 out of the 2 source contexts (1/2 = 0.5).

The relevancy evaluator is an `IndependentEvaluator`.
This means evaluating two outputs with `pairwise_evaluation` is the same as evaluating both outputs independently:
```python
from rago.utils import Document, RAGOutput

output_1 = RAGOutput(retrieved_documents=[Document("Thomas is 12.")])
output_2 = RAGOutput(retrieved_documents=[Document("Thomas is 13.")])

eval_result_1 = evaluator.evaluate(output_1, eval_sample)
eval_result_2 = evaluator.evaluate(output_2, eval_sample)

pairwise_eval_result_1, pairwise_eval_result_2 = evaluator.evaluate_pairwise(output_1, output_2, eval_sample)

assert eval_result_1["relevancy"].score == pairwise_eval_result_1["relevancy"].score
assert eval_result_2["relevancy"].score == pairwise_eval_result_2["relevancy"].score
```

More generally, any Evaluator in RAGO is either an independent evaluator or a dependent evaluator.
- Independent evaluators inherit from ``BaseIndependentEvaluator``. This implies that the pairwise evaluation is just two sequential direct evaluation. This is why the example above is true.
- Dependent evaluators inherit from ``BaseDependantEvaluator``. This means that the example above would not hold.


### LLM evaluator

LLM evaluators are evaluators that use a LLM to generate scores.

> [!CAUTION]
> In the following subsection we present the ``SimpleLLMEvaluator`` only. Several LLM Evaluator shall be implemented.

The ``SimpleLLMEvaluator`` is an evaluator that uses an LM to output a score between `min_score` and ``max_score``.
This evaluator only outputs a score without explanation.
The evaluators can be instantiated with the same logics as ``DatasetGenerator``:
Default Instantiation:
```python
from rago.eval import SimpleLLMEvaluator

evaluator = SimpleLLMEvaluator()
```
Instantiation with a LLMConfig:
```python
from rago.eval import SimpleLLMEvaluator
from rago.eval.llm_evaluator.base import LLMEvaluatorConfig
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig

llm_config = LangchainOllamaConfig(model_name="smollm2:1.7b", temperature= 0.0)
evaluator_config = LLMEvaluatorConfig(judge = llm_config)
evaluator = SimpleLLMEvaluator.make(evaluator_config)
```
Instantiation with an existing LLM:

```python
from rago.eval import SimpleLLMEvaluator
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgentFactory

llm_config = LangchainOllamaConfig(model_name="phi3:3.8b-mini-128k-instruct-q8_0")
llm_agent = LLMAgentFactory.make(llm_config)
evaluator = SimpleLLMEvaluator(llm_agent)
```

Then once it is instantiated we can use it to evaluate RAG outputs:

1. Individually:

    ```python
    from rago.data_objects import EvalSample, RAGOutput

    eval_sample = EvalSample(query="How old is thomas?")
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    eval_result["correctness"].score
    ```
2. Pairwise:

    ```python
    from rago.data_objects import EvalSample, RAGOutput

    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 13.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    print(eval_result_1["correctness"].score, eval_result_2["correctness"].score)
    ```

## 📚 Related Documentation

- 📊 [Metrics](metrics.md)
- ⚙️ [RAG Configurations](../rag/rag_configuration.md)
