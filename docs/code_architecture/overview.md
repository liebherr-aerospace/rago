# Code Architecture Overview

This document provides a high-level overview of RAGO's code architecture and design patterns.

---

## 📦 Project Structure

```
rago/
├── data_objects/           # Core data structures (RAGOutput, EvalSample, Metric)
├── dataset/                # Dataset handling and generation
│   └── generator/           # Automatic dataset generation from documents
├── eval/                   # Evaluation framework and metrics
├── model/                  # Model wrappers and configurations
│   ├── configs/             # Configuration dataclasses
│   └── wrapper/             # Unified interfaces for different frameworks
├── optimization/           # Optimization engine
│   ├── manager/             # Optimization managers (Direct, Pairwise)
│   └── search_space/        # Parameter search spaces
├── prompts/                # Prompt templates and configurations
└── utils/                  # Utility functions
```

---

## 💡 Design Principles

### Separation of Concerns
- **Config**: What you want
- **Wrapper**: How to build it
- **Space**: What to optimize
- **Manager**: How to optimize

### Framework Agnostic
- Core logic independent of LangChain/LlamaIndex
- Easy to add new frameworks via wrappers

### Type Safety
- Dataclasses for configs
- Type hints everywhere
- Pydantic validation

### Composability
- Small, focused components
- Easy to combine in different ways
- Clear interfaces

### Extensibility
- Abstract base classes for extension points
- Plugin-like architecture for new components

---

## 🎯 Core Design Patterns

### Config Pattern

All components use **pydantic dataclass configs** (subclasses of `Config`):

```python
from pydantic.dataclasses import dataclass

from rago.model.configs.retriever_config.base import RetrieverConfig

@dataclass
class LangchainRetrieverConfig(RetrieverConfig):
    """Configuration of the langchain retriever."""
    type: str                                       # "VectorIndexRetriever", "BM25Retriever", "HybridRetriever"
    similarity_function: Optional[str] = None       # e.g. "cosine"
    search_type: Optional[str] = None               # e.g. "similarity_score_threshold"
    search_kwargs: Optional[dict[str, Any]] = None  # e.g. {"k": 5, "score_threshold": 0.7}
    encoder: Optional[LangchainEncoderConfig] = None
    node_post_processor_config: Optional[list[NodePostProcessorConfig]] = None
    # Hybrid retriever fields
    vector_config: Optional[LangchainRetrieverConfig] = None
    bm25_config: Optional[LangchainRetrieverConfig] = None
    hybrid_weight: Optional[float] = None           # weight of the vector retriever (1 - weight for BM25)
```

**Benefits:**
- ✅ Type-safe configuration
- ✅ Validation at creation time
- ✅ Easy serialization/deserialization
- ✅ Clear documentation via type hints

---

### Wrapper Pattern

RAGO provides **unified interfaces** for different LLM/RAG frameworks. Every optimizable model is a `TunableModel`:

```python
class TunableModel(ABC):
    @abstractmethod
    def get_output(self, query: str) -> RAGOutput: ...

class Retriever(TunableModel):          # get_output -> RAGOutput(answer=None, retrieved_context=...)
    @abstractmethod
    def get_retriever_output(self, query: str) -> list[RetrievedContext]: ...

class Reader(TunableModel):             # get_output -> RAGOutput(answer=..., retrieved_context=None)
    @abstractmethod
    def get_reader_output(self, query: str, retrieved_context: Optional[list[RetrievedContext]] = None) -> str: ...

class RAG(TunableModel):                # retriever + reader
    def get_output(self, query: str) -> RAGOutput:
        retrieved_context = self.retriever.get_retriever_output(query)
        answer = self.reader.get_reader_output(query, retrieved_context)
        return RAGOutput(answer=answer, retrieved_context=retrieved_context)
```

Concrete wrappers: `LangchainRetrieverWrapper`, `HybridLangchainRetrieverWrapper`, `LlamaIndexRetrieverWrapper`, `QdrantRetrieverWrapper` (retrievers) and LangChain / LlamaIndex / HuggingFace readers.

**Benefits:**
- ✅ Framework-agnostic optimization
- ✅ Easy to add new frameworks
- ✅ Consistent interface across implementations
- ✅ Swap implementations without changing optimization code

---

### Space Pattern

Search spaces define **optimization parameters**. Every space a manager can optimize is a `TunableModelConfigSpace` whose `sample(trial)` returns a `TunableModelConfig`:

```python
@dataclass
class RetrieverConfigSpace(TunableModelConfigSpace):
    retriever_type_name: CategoricalParamSpace = Field(default=CategoricalParamSpace(choices=["VectorIndexRetriever"]))
    top_k: IntParamSpace = Field(default=IntParamSpace(low=1, high=5))
    score_threshold: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=0.9))
    ...

    def sample(self, trial: optuna.trial.BaseTrial) -> LangchainRetrieverConfig:
        ...  # dispatch on retriever_type_name: vector / BM25 / hybrid
```

**Benefits:**
- ✅ Declarative parameter definition
- ✅ Type-safe parameter ranges
- ✅ Integration with Optuna
- ✅ Easy to extend with new parameters

---

## 🏗️ Component Hierarchy

### Model Layer

```
Config → Factory → Wrapper (TunableModel)

LangchainRetrieverConfig / QdrantRetrieverConfig → RetrieverWrapperFactory → Retriever
ReaderConfig (LangChain / LlamaIndex / HF)        → ReaderWrapperFactory    → Reader
RAGConfig (retriever + reader)                    → RAG.make                → RAG

make_tunable_model(config) dispatches to the right factory based on the config type.
```

**Flow:**
1. **Config**: Defines what you want (pydantic dataclass)
2. **Factory / Wrapper**: Translates config to framework-specific code
3. **TunableModel**: Answers queries through `get_output(query) -> RAGOutput`

**Example:**
```python
from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig
from rago.model.wrapper.tunable_model_factory import make_tunable_model

# 1. Define config
config = LangchainRetrieverConfig(
    type="VectorIndexRetriever",
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},
)

# 2. Build the model (RAGConfig, RetrieverConfig or ReaderConfig)
retriever = make_tunable_model(config, inputs_chunks=["Thomas is 12.", "Paris is in France."])

# 3. Use it
output = retriever.get_output("How old is Thomas?")
output.retrieved_context  # list[RetrievedContext]
```

---

### Optimization Layer

```
TunableModelConfigSpace → Manager → Optuna Study

RAGConfigSpace / RetrieverConfigSpace / QdrantRetrieverConfigSpace / ReaderConfigSpace
        → SimpleDirectOptunaManager / SimplePairWiseOptunaManager → optuna.Study
                  ↓
              Evaluator → dict[str, Metric]  (optim_metric_name selects the optimized one)
```

**Flow:**
1. **SearchSpace**: Defines parameter ranges
2. **Manager**: Orchestrates optimization loop
3. **Optuna**: Suggests next configuration (TPE)
4. **Evaluator**: Scores RAG output
5. **Feedback**: Update Optuna model

---

## 🔄 Data Flow

### Complete Optimization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION LOOP                           │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ TunableModel     │  Define parameter ranges
    │ ConfigSpace      │  (RAG, retriever-only or reader-only)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Optuna Trial    │  Suggest next configuration
    │  (TPE sampler)   │  based on past results
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ TunableModel     │  Sampled configuration
    │ Config           │  {retriever: vector, k: 5, temp: 0.7...}
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │make_tunable_model│  Build RAG / Retriever / Reader
    │  (instantiate)   │  from config
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │   EvalSample     │────────>│ model.get_output │  Query the model
    │  (test question) │         │  (execute)       │  with test sample
    └──────────────────┘         └────────┬─────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │   RAGOutput      │  Generated answer
                                 │ (answer+context) │  + retrieved contexts
                                 └────────┬─────────┘
                                          │
    ┌──────────────────┐                  │
    │  Expected Answer │──────────────────┘
    │ (ground truth)   │         │
    └──────────────────┘         ▼
                        ┌──────────────────┐
                        │   Evaluator      │  Compare output vs expected
                        │ (BERTScore/LLM)  │  (compute quality score)
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ dict[str, Metric]│  Score of optim_metric_name
                        │   (score: 0.85)  │  (quality metric)
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Optuna Update   │  Learn from result
                        │  (update model)  │  Update probability model
                        └────────┬─────────┘
                                 │
                                 │ (repeat until convergence)
                                 │
                                 └──────────────────────────────┐
                                                                │
                                                                ▼
                                                    ┌──────────────────┐
                                                    │   Best Config    │
                                                    │  (optimal RAG)   │
                                                    └──────────────────┘
```

---

## 🎨 Key Abstractions

### Config Classes

**Location**: `rago/model/configs/`

```
Config (DataObject)
└── TunableModelConfig
    ├── RAGConfig(reader: ReaderConfig, retriever: RetrieverConfig)
    ├── RetrieverConfig
    │   ├── LangchainRetrieverConfig   (vector, BM25, hybrid)
    │   ├── LlamaIndexRetrieverConfig
    │   └── QdrantRetrieverConfig
    └── ReaderConfig
        ├── LangchainReaderConfig      (llm: LangchainLLMConfig)
        └── LLamaIndexReaderConfig     (llm: LlamaIndexOllamaConfig)

LLM configs: LangchainOllamaConfig, LangchainHuggingFaceConfig, LlamaIndexOllamaConfig
```

**Purpose**: Type-safe, serializable, validatable configurations

---

### Wrapper Classes

**Location**: `rago/model/wrapper/`

```
TunableModel.get_output(query) -> RAGOutput
├── RAG          (Retriever + Reader)
├── Retriever    .get_retriever_output(query) -> list[RetrievedContext]
│   ├── LangchainRetrieverWrapper, HybridLangchainRetrieverWrapper
│   ├── LlamaIndexRetrieverWrapper
│   └── QdrantRetrieverWrapper
└── Reader       .get_reader_output(query, retrieved_context) -> str

Factories: RetrieverWrapperFactory.make(config, input_chunks), ReaderWrapperFactory.make(config, prompt_config),
           RAG.make(rag_config, prompt_config, inputs_chunks), make_tunable_model(config, prompt_config, inputs_chunks)
```

**Purpose**: Abstract away framework-specific details

---

### Space Classes

**Location**: `rago/optimization/search_space/`

```
ConfigSpace.sample(trial) -> Config
├── TunableModelConfigSpace          (usable as a manager config_space)
│   ├── RAGConfigSpace(retriever_space, reader_space)
│   ├── RetrieverConfigSpace         (vector / BM25 / hybrid)
│   ├── QdrantRetrieverConfigSpace
│   └── ReaderConfigSpace            (LangChain / LlamaIndex / HuggingFace)
└── sub-spaces: HFEncoderConfigSpace, OllamaLLMConfigSpace, OllamaLlamaIndexLLMConfigSpace, ...

Parameters: CategoricalParamSpace, IntParamSpace, FloatParamSpace
```

**Purpose**: Declarative optimization parameter definition

---

### Manager Classes

**Location**: `rago/optimization/manager/`

```python
class BaseOptunaManager[EvaluatorType: BaseEvaluator[RAGOutput]](ABC):
    def __init__(
        self, *,
        params: Optional[OptimParams] = None,
        datasets: dict[str, RAGDataset],             # {"train": ..., "test": ...}
        optim_evaluator: EvaluatorType,
        optim_metric_name: str,                      # key of the evaluator result to optimize
        test_evaluators: list[BaseEvaluator],
        config_space: Optional[TunableModelConfigSpace] = None,  # defaults to RAGConfigSpace()
        prompt_config: Optional[PromptConfig] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> None: ...

    def sample_model(self, trial, dataset) -> TunableModel:
        config = self.config_space.sample(trial)
        inputs_chunks = [doc.text for doc in dataset.corpus.values()]
        return make_tunable_model(config, prompt_config=self.prompt_config, inputs_chunks=inputs_chunks)

    @abstractmethod
    def optimize(self) -> None: ...

    @abstractmethod
    def eval_trial(self, trial, dataset: RAGDataset) -> float: ...

    def test(self) -> RAGCandidateEval: ...  # evaluates the best trial on datasets["test"]

# Direct optimization
class SimpleDirectOptunaManager(BaseOptunaManager[BaseEvaluator]):
    def eval_trial(self, trial, dataset) -> float:
        model = self.sample_model(trial, dataset)
        for n, sample in enumerate(dataset.samples):
            output = model.get_output(sample.query)
            evaluation = self.optim_evaluator.evaluate(output, sample)  # dict[str, Metric]
            ...  # running mean of evaluation[self.optim_metric_name].score, report + prune
        return mean_score
```

`SimplePairWiseOptunaManager` compares candidates pairwise with an LLM judge (`BaseLLMEvaluator`).

**Purpose**: Orchestrate optimization loop

---

## 🔌 Extension Points

### Adding a New Retriever

```python
# 1. Define config
@dataclass
class MyRetrieverConfig(RetrieverConfig):
    param1: str
    param2: int

# 2. Create wrapper
class MyRetriever(Retriever):
    @classmethod
    def make(cls, config: MyRetrieverConfig, input_chunks: list[str]) -> MyRetriever: ...

    def get_retriever_output(self, query: str) -> list[RetrievedContext]:
        ...  # get_output() is inherited and wraps this in a RAGOutput

# 3. Register it in rago/model/wrapper/retriever/retriever_wrapper_factory.py
_REGISTRY[MyRetrieverConfig] = MyRetriever.make

# 4. Add a search space
@dataclass
class MyRetrieverConfigSpace(TunableModelConfigSpace):
    param2: IntParamSpace = Field(default=IntParamSpace(low=1, high=10))

    def sample(self, trial: optuna.trial.BaseTrial) -> MyRetrieverConfig:
        return MyRetrieverConfig(param1="fixed", param2=self.param2.sample(trial))
```

The space can be used alone (retriever-only optimization) or as `RAGConfigSpace(retriever_space=...)`.

---

### Adding a New Evaluator

```python
# 1. Inherit from BaseIndependentEvaluator (one score per sample)
from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval import BaseIndependentEvaluator

class MyEvaluator(BaseIndependentEvaluator[RAGOutput]):
    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        score = compute_score(candidate_output.answer, eval_sample.reference_answer)
        return {"my_metric": Metric(score=score)}

# 2. Use in optimization
datasets = {"train": train_ds, "test": test_ds}

optimizer = SimpleDirectOptunaManager(
    params=OptimParams(...),
    datasets=datasets,
    optim_evaluator=MyEvaluator(),
    optim_metric_name="my_metric",  # ← key returned by evaluate()
    test_evaluators=[MyEvaluator()],
    config_space=RAGConfigSpace(),
)
```

---

<!-- ## 🔗 Related Documentation

- [Configs](configs.md) - Detailed config reference
- [Wrappers](wrappers.md) - Wrapper implementation guide
- [Search Spaces](search_spaces.md) - Parameter space definition
- [Managers](managers.md) - Optimization manager details

--- -->
