# Metric

As its name hints a `Metric` corresponds to the result of an evaluation.
It contains a `score` and optionally an `explanation` of the score, see [Metrics Dataclass](../../../rago/data_objects/metric.py)

Below is an example usage of the EvaluationResult:

```python
from rago.data_objects import Metric

correctness = Metric(5, explanation="The answer is perfect and targets all the key points of the query correctly")

```

Metrics can be generated manually as in the example above or using an evaluator.
An Evaluator outputs a dictionary of Metrics as it can output multiple score, see for instance the [Sequential Evaluator](../../../rago/eval/sequential.py)

---

## 📚 Related Documentation

- 📓 [Evaluators](evaluator.md)
- ⚙️ [RAG Configurations](../rag/rag_configuration.md)
