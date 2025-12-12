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
from rago.optimization.manager import SimpleDirectOptunaManager

optimizer = SimpleDirectOptunaManager.from_dataset(
    params=params,
    dataset=ds,
    evaluator=evaluator,
    metric_name="bert_score_f1",
    config_space=config_space,
    sampler=None,      # Optional: custom sampler (default: TPE)
    pruner=None,       # Optional: custom pruner (default: MedianPruner)
)
```

### Default Sampler and Pruner

**Sampler** (default: `optuna.samplers.TPESampler`):
- Controls **how** new configurations are suggested
- **TPE (Tree-structured Parzen Estimator)** is the default
- Balances exploration (random search) and exploitation (focusing on promising areas)
- Uses `n_startup_trials` random trials, then switches to TPE

**Pruner**:
- Controls **when** to stop unpromising trials early
- **MedianPruner**: Stops trials performing worse than the median of previous trials
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

optimizer = SimpleDirectOptunaManager.from_dataset(
    params=params,
    dataset=ds,
    evaluator=evaluator,
    metric_name="bert_score_f1",
    config_space=config_space,
    sampler=custom_sampler,
    pruner=custom_pruner,
)
```

## 📔 Complete Example

### Optimization from a RAG dataset

```python
from rago.dataset import QADatasetLoader, RAGDataset
from rago.eval import BertScore
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
ds = QADatasetLoader.load_dataset(RAGDataset, "crag").sample(10, 0, 50)
evaluator = BertScore()

# 3. Define search space
config_space = RAGConfigSpace()

# 4. Instantiate the optimizer (uses TPE sampler + no pruner by default)
optimizer = SimpleDirectOptunaManager.from_dataset(
    params=params,
    dataset=ds,
    evaluator=evaluator,
    metric_name="bert_score_f1",
    config_space=config_space,
)

# 5. Start optimization
study = optimizer.optimize()

# 6. Get best configuration
print(f"Best score: {study.best_value}")
print(f"Best config: {study.best_params}")
```

### Optimization from a list of Documents

To instantiate the manager from a seed set of documents and generate synthetic documents:

```python
from typing import cast

from rago.dataset import QADatasetLoader, RAGDataset
from rago.eval import SimpleLLMEvaluator
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
evaluator = SimpleLLMEvaluator()

# 3. Define search space
config_space = RAGConfigSpace(
    reader_space = LangchainReaderConfigSpace(
        OllamaLLMConfigSpace(model_name=CategoricalParamSpace(choices=["smollm:1.7b"])),
    ),
)

# 4. Instantiate the optimizer (uses TPE sampler + no pruner by default)
optimizer = SimpleDirectOptunaManager.from_seed_data(
            params= params,
            seed_data= corpus,
            evaluator=evaluator,
            metric_name="correctness",
            config_space = config_space,
        )

# 5. Start optimization
study = optimizer.optimize()
```
> You can also replace the `LlamaIndexReaderConfigSpace` by `LangchainReaderConfigSpace` to run the langchain reader. However, in addition to generation via the LLM, the `LlamaIndexReaderConfigSpace` provides more advanced capabilities, such as 'refine', 'compact', and 'summarize' methods, which process chunks and may involve multiple LLM calls.

> [!CAUTION]
> You can also replace the `SimpleDirectOptunaManager` by `SimplePairWiseOptunaManager` to run the pair-wise optimization.

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
