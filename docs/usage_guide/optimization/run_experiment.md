# Run Optimization with Optuna

## 🛠️ Configure Your Optimization

RAGO uses **Optuna** as its optimization framework.

Just a reminder on the configuration:
> [!CAUTION]
> - `TEST_OLLAMA_HOST` needs to be an env variable, "https://ollama.myserver.i" for instance
> - If you use the BM25 retriever you need to specify 2 env variables to use `OPEN SEARCH`. In our case we use elasticsearch, therefore here are our current variables:
>   - `OPENSEARCH_URL` is set to "https://node.myserver.i/"
>   - `OPENSEARCH_INDEX_NAME` is set to "rago_xp" for instance

The main components of the optimization frameworks are:

### The Basic optimization parameters

```python
from rago.optimization.manager import OptimParams

params = OptimParams(
    experiment_name="my_experiment",    # Name for saving results
    n_startup_trials=50,                # Random trials before TPE starts
    n_iter=1000,                          # Total number of trials
)
```
**Parameters explained:**
- `experiment_name`: Unique identifier for your optimization run. Results are saved in `experiments/{experiment_name}/`
- `n_startup_trials`: Number of **random exploration** trials before TPE learning begins. A rule is around 5-10% of the total number of trials
  - Too few → TPE might focus on local optima
  - Too many → Wastes time on random search
  - **Recommended**: 10-20% of total trials
- `n_iter`: Total number of configurations to test

### The Optimization manager

The manager orchestrates the optimization process:

```python
from rago.model.wrapper.rag.base import RAG
from rago.optimization.manager import SimpleDirectOptunaManager

# Split dataset into train/test
datasets = {"train": train_ds, "test": test_ds}

optimizer = SimpleDirectOptunaManager(
    params=params,
    datasets=datasets,              # Dictionary with 'train' and 'test' splits
    optim_evaluator=evaluator,      # Evaluator for optimization
    optim_metric_name="bert_score_f1",  # Metric to optimize
    test_evaluators=[evaluator],    # Evaluators for final test
    model_cls=RAG,                  # Model built from each sampled config: model_cls.make(config, ctx)
    config_space=config_space,      # TunableModelConfigSpace whose configs model_cls accepts
    sampler=None,                   # Optional: custom sampler (default: TPE)
    pruner=None,                    # Optional: custom pruner (default: Optuna's MedianPruner)
)
```

> `optim_metric_name` must be one of the keys returned by the evaluator, e.g. `bert_score_f1` for `BertScore`, `correctness` for LLM judges, `relevancy` / `precision` / `context_f1` for `ContextRetrievalEvaluator`.

### Default Sampler and Pruner

**Sampler** (default: `optuna.samplers.TPESampler`):
- Controls **how** new configurations are suggested
- **TPE (Tree-structured Parzen Estimator)** is the default
- Balances exploration (random search) and exploitation (focusing on promising areas)
- Starts with random trials, then switches to TPE. With `sampler=None`, Optuna's default number of startup trials is used: pass `sampler=optuna.samplers.TPESampler(n_startup_trials=params.n_startup_trials)` to apply your `n_startup_trials`.

**Pruner** (default: Optuna's `MedianPruner` when `pruner=None`):
- Controls **when** to stop unpromising trials early
- **MedianPruner**: Stops trials performing worse than the median of previous trials
- RAGO only prunes a trial if the pruner says so **and** its current mean score is below the best trial
- Saves computation by abandoning bad configurations early
- Especially useful for multi-step evaluations (e.g., evaluating on multiple test samples)

### Custom Sampler/Pruner (Advanced)

You can use several pruners and override sampler defaults for advanced control:

```python
import optuna

# Custom TPE with specific parameters
custom_sampler = optuna.samplers.TPESampler(
    n_startup_trials=50,
    multivariate=True,      # Model parameter interactions
    seed=42,                # Reproducibility
)

# More aggressive pruning
custom_pruner = optuna.pruners.MedianPruner(
    n_startup_trials=50,
    n_warmup_steps=3,       # Don't prune until 3 evaluations
)

# Split dataset into train/test
datasets = {"train": train_ds, "test": test_ds}

optimizer = SimpleDirectOptunaManager(
    params=params,
    datasets=datasets,
    optim_evaluator=evaluator,
    optim_metric_name="bert_score_f1",
    test_evaluators=[evaluator],
    model_cls=RAG,
    config_space=config_space,
    sampler=custom_sampler,
    pruner=custom_pruner,
)
```

## 📔 Complete Example

### Optimization from a RAG dataset

```python
from rago.dataset import RAGDataset
from rago.eval import BertScore
from rago.model.wrapper.rag.base import RAG
from rago.optimization.manager import OptimParams, SimpleDirectOptunaManager
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

# 1. Configure optimization
params = OptimParams(
    experiment_name="my_rag_optimization",
    n_startup_trials=50,     # 50 random trials
    n_iter=1000,             # 1000 total trials
)

# 2. Load dataset and evaluator
full_ds = RAGDataset.load_dataset("crag").sample(10, 0, 50)
evaluator = BertScore()

# 3. Split dataset into train/test (e.g., 80/20 split)
datasets = full_ds.split_dataset([0.8], split_names=["train", "test"], seed=42)

# 4. Define search space
config_space = RAGConfigSpace()

# 5. Instantiate the optimizer (TPE sampler + Optuna's MedianPruner by default)
optimizer = SimpleDirectOptunaManager(
    params=params,
    datasets=datasets,
    optim_evaluator=evaluator,
    optim_metric_name="bert_score_f1",
    test_evaluators=[evaluator],
    model_cls=RAG,
    config_space=config_space,
)

# 6. Start optimization
optimizer.optimize()

# 7. Get best configuration
study = optimizer.manager  # Access the Optuna study
print(f"Best score: {study.best_value}")
print(f"Best config: {study.best_params}")
```

### Optimization from a list of Documents

To instantiate the manager from a seed set of documents and generate synthetic documents:

```python
from typing import cast

from rago.dataset import QADatasetLoader, RAGDataset
from rago.eval import SimpleLLMEvaluator
from rago.model.wrapper.rag.base import RAG
from rago.optimization.manager import OptimParams, SimpleDirectOptunaManager
from rago.optimization.search_space.llm_config_space import OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace

# 1. Configure optimization
params = OptimParams(
    n_iter = 10,
)

# 2. Get corpus data and evaluator
corpus = cast(RAGDataset, QADatasetLoader.load_dataset(RAGDataset, "crag")).corpus_docs[:200]
evaluator = SimpleLLMEvaluator.make()

# 3. Define search space
config_space = RAGConfigSpace(
    reader_space = LangchainReaderConfigSpace(
        OllamaLLMConfigSpace(model_name=CategoricalParamSpace(choices=["smollm:1.7b"])),
    ),
)

# 4. Instantiate the optimizer from seed data (generates synthetic Q&A pairs)
optimizer = SimpleDirectOptunaManager.from_seed_data(
    params=params,
    seed_data=corpus,
    splits=(0.8, 0.2),                  # Train/test split
    optim_evaluator=evaluator,
    optim_metric_name="correctness",
    test_evaluators=[evaluator],
    model_cls=RAG,
    config_space=config_space,
)

# 5. Start optimization
optimizer.optimize()
```

> **Thinking models** (e.g. `qwen3`, `qwen3.5`): `OllamaLLMConfigSpace` has `reasoning=False` by default, so they answer directly. With `reasoning=None` or `True`, the thinking tokens count in `max_new_tokens` and can leave the answer empty (score 0 on every trial). See [Thinking models](../rag/reader.md#thinking-models).

> You can also replace the `LangchainReaderConfigSpace` by `LlamaIndexReaderConfigSpace` to run the LlamaIndex reader. In addition to generation via the LLM, the `LlamaIndexReaderConfigSpace` provides more advanced capabilities, such as 'refine', 'compact', and 'summarize' methods, which process chunks and may involve multiple LLM calls.

> [!CAUTION]
> You can also replace the `SimpleDirectOptunaManager` by `SimplePairWiseOptunaManager` to run the pair-wise optimization.

### Optimize a single component

The manager builds each sampled config with `model_cls.make(config, ctx)`, so you can optimize the retriever or the reader alone by passing the matching pair:

| `config_space` | `model_cls` | Typical evaluator / metric |
|---|---|---|
| `RAGConfigSpace` | `RAG` (retriever + reader) | `BertScore` / `bert_score_f1`, LLM judge / `correctness` |
| `RetrieverConfigSpace` (vector or BM25) | `LangchainRetrieverWrapper` | `ContextRetrievalEvaluator` / `relevancy`, `precision`, `context_f1` |
| `RetrieverConfigSpace` (`HybridRetriever` only) | `HybridLangchainRetrieverWrapper` | `ContextRetrievalEvaluator` |
| `QdrantRetrieverConfigSpace` | `QdrantRetrieverWrapper` | `ContextRetrievalEvaluator` |
| `LangchainReaderConfigSpace` | `LangchainWrapperReader` (no retrieved context) | `BertScore`, LLM judge |
| `LlamaIndexReaderConfigSpace` | `LlamaIndexWrapperReader` (no retrieved context) | `BertScore`, LLM judge |

> A retriever-only study uses one `model_cls`, so it cannot mix `HybridRetriever` with vector/BM25 choices. Inside a `RAG`, all retriever types can be mixed.

Retriever-only optimization does not call any LLM, so it is fast:

```python
from rago.dataset import RAGDataset
from rago.eval import ContextRetrievalEvaluator
from rago.model.wrapper.retriever.langchain_retriever import LangchainRetrieverWrapper
from rago.optimization.manager import OptimParams, SimpleDirectOptunaManager
from rago.optimization.search_space.param_space import CategoricalParamSpace, IntParamSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

datasets = RAGDataset.load_dataset("crag").sample(10, 0, 50).split_dataset([0.8], split_names=["train", "test"], seed=42)
evaluator = ContextRetrievalEvaluator()  # returns relevancy, precision and context_f1

optimizer = SimpleDirectOptunaManager(
    params=OptimParams(experiment_name="retriever_only", n_iter=50),
    datasets=datasets,
    optim_evaluator=evaluator,
    optim_metric_name="context_f1",  # "relevancy" alone pushes towards a large top_k
    test_evaluators=[evaluator],
    model_cls=LangchainRetrieverWrapper,
    config_space=RetrieverConfigSpace(
        # BM25Retriever needs OpenSearch (see env variables above)
        retriever_type_name=CategoricalParamSpace(choices=["VectorIndexRetriever", "BM25Retriever"]),
        top_k=IntParamSpace(low=1, high=10),
    ),
)
optimizer.optimize()
result = optimizer.test()
print(result.mean_evals, result.config)
```

> Changing the categorical choices or the optimized metric of an existing study is not supported by Optuna: use a new `experiment_name`.

## ⛵ To go further

👉 [Pimp your RAG configuration space (retriever and reader)](../rag/rag_configuration.md#pimp-your-rag-configuration-space)

> When you run an experiment with an already existing experiment name and with the same configuration spaces, Optuna will keep computing the experiment by using the previous iterations and computes.

### 📄 Optimization Logs

A log file exists in each experiment directory that is created and contains all the important optimization steps, for instance in `experiments/my_experiment/my_experiment.log`.

### 👀 Visualization

During the optimization process, in order to visualize the optimization details and identify hyperparameter importance, `Optuna` also offers an interactive dashboard to visualize the results. Here is how to use it (in directory experiments/my_experiment/ if exist):

```bash
optuna-dashboard sqlite:///experiments/my_experiment/study.db
```

---

## 📚 Related Documentation

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Algorithm](tpe.md)
- [RAG Configurations](../rag/rag_configuration.md)

---
