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

All components use **dataclass configs** for configuration:

```python
from dataclasses import dataclass
from rago.model.configs import LangchainRetrieverConfig

@dataclass
class LangchainRetrieverConfig:
    """Configuration for a retriever."""
    type: str                          # e.g., "VectorIndexRetriever"
    similarity_function: str           # e.g., "cosine"
    search_type: str                   # e.g., "similarity_score_threshold"
    search_kwargs: dict                # e.g., {"k": 5, "score_threshold": 0.7}
    encoder: HFEncoderConfig          # Embedding model config
```

**Benefits:**
- ✅ Type-safe configuration
- ✅ Validation at creation time
- ✅ Easy serialization/deserialization
- ✅ Clear documentation via type hints

---

### Wrapper Pattern

RAGO provides **unified interfaces** for different LLM/RAG frameworks:

```python
# Wrapper interface
class RAG(ABC):
    """Abstract base class for RAG systems."""

    @abstractmethod
    def query(self, query: str) -> RAGOutput:
        """Query the RAG system."""
        pass

# Concrete implementations
class LangchainRAG(RAG):
    """LangChain-based RAG implementation."""
    pass

class LlamaIndexRAG(RAG):
    """LlamaIndex-based RAG implementation."""
    pass
```

**Benefits:**
- ✅ Framework-agnostic optimization
- ✅ Easy to add new frameworks
- ✅ Consistent interface across implementations
- ✅ Swap implementations without changing optimization code

---

### Space Pattern

Search spaces define **optimization parameters**:

```python
from rago.optimization.search_space import RAGConfigSpace, RetrieverConfigSpace
from rago.optimization.search_space.param_space import IntParamSpace, FloatParamSpace

@dataclass
class RetrieverConfigSpace:
    """Defines the search space for retriever parameters."""

    top_k: IntParamSpace = IntParamSpace(low=1, high=10)
    score_threshold: FloatParamSpace = FloatParamSpace(low=0.0, high=1.0)

    def sample(self, trial: optuna.Trial) -> RetrieverConfig:
        """Sample a configuration from this space."""
        return RetrieverConfig(
            top_k=self.top_k.sample(trial),
            score_threshold=self.score_threshold.sample(trial)
        )
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
Config → Wrapper → Model

LangchainRetrieverConfig → LangchainRetriever → Vector Store
LangchainLLMConfig → LangchainLLM → Ollama/OpenAI
RAGConfig → RAG (LangChain/LlamaIndex) → Complete RAG Pipeline
```

**Flow:**
1. **Config**: Defines what you want (dataclass)
2. **Wrapper**: Translates config to framework-specific code
3. **Model**: Actual LLM/retriever/RAG implementation

**Example:**
```python
# 1. Define config
config = LangchainRetrieverConfig(
    type="VectorIndexRetriever",
    search_kwargs={"k": 5}
)

# 2. Wrapper creates the actual retriever
retriever = LangchainRetriever.from_config(config)

# 3. Use the retriever
docs = retriever.retrieve("query")
```

---

### Optimization Layer

```
SearchSpace → Manager → Optuna Study

RAGConfigSpace → SimpleDirectOptunaManager → optuna.Study
                  ↓
              Evaluator → Metric
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
    │  RAGConfigSpace  │  Define parameter ranges
    │  (search space)  │  (retriever, LLM, reader params)
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
    │   RAGConfig      │  Sampled configuration
    │  (specific vals) │  {retriever: vector, k: 5, temp: 0.7...}
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   RAG Wrapper    │  Build RAG system from config
    │  (instantiate)   │  (retriever + LLM + reader)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │   EvalSample     │────────>│   RAG.query()    │  Query RAG system
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
                        │     Metric       │  Score: 0.85
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

```python
@dataclass
class BaseConfig:
    """Base configuration class."""
    pass

# Retriever configs
@dataclass
class LangchainRetrieverConfig(BaseConfig):
    type: str
    search_kwargs: dict
    encoder: HFEncoderConfig

# LLM configs
@dataclass
class LangchainLLMConfig(BaseConfig):
    model: str
    temperature: float
    top_p: float

# Complete RAG config
@dataclass
class RAGConfig(BaseConfig):
    retriever: LangchainRetrieverConfig
    llm: LangchainLLMConfig
    reader: ReaderConfig
```

**Purpose**: Type-safe, serializable, validatable configurations

---

### Wrapper Classes

**Location**: `rago/model/wrapper/`

```python
class BaseWrapper(ABC):
    """Base wrapper for models."""

    @abstractmethod
    def from_config(cls, config: BaseConfig):
        """Create instance from config."""
        pass

# Example: RAG wrapper
class RAG(BaseWrapper):
    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = self._build_retriever()
        self.llm = self._build_llm()

    def query(self, query: str) -> RAGOutput:
        """Execute RAG pipeline."""
        contexts = self.retriever.retrieve(query)
        answer = self.llm.generate(query, contexts)
        return RAGOutput(answer=answer, contexts=contexts, query=query)
```

**Purpose**: Abstract away framework-specific details

---

### Space Classes

**Location**: `rago/optimization/search_space/`

```python
@dataclass
class ConfigSpace(ABC):
    """Base class for configuration search spaces."""

    @abstractmethod
    def sample(self, trial: optuna.Trial) -> BaseConfig:
        """Sample a config from this space."""
        pass

# Example: Retriever search space
@dataclass
class RetrieverConfigSpace(ConfigSpace):
    retriever_type: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(choices=["VectorIndexRetriever"])
    )
    top_k: IntParamSpace = Field(default=IntParamSpace(low=1, high=10))
    score_threshold: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))

    def sample(self, trial: optuna.Trial) -> LangchainRetrieverConfig:
        return LangchainRetrieverConfig(
            type=self.retriever_type.sample(trial),
            search_kwargs={
                "k": self.top_k.sample(trial),
                "score_threshold": self.score_threshold.sample(trial)
            }
        )
```

**Purpose**: Declarative optimization parameter definition

---

### Manager Classes

**Location**: `rago/optimization/manager/`

```python
class BaseOptunaManager(ABC):
    """Base optimization manager."""

    def __init__(self, params: OptimParams, config_space: RAGConfigSpace):
        self.params = params
        self.config_space = config_space
        self.study = self._create_study()

    @abstractmethod
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        pass

    def optimize(self) -> optuna.Study:
        """Run optimization."""
        self.study.optimize(self.objective, n_trials=self.params.n_iter)
        return self.study

# Direct optimization
class SimpleDirectOptunaManager(BaseOptunaManager):
    def objective(self, trial: optuna.Trial) -> float:
        # Sample config from space
        rag_config = self.config_space.sample(trial)

        # Build RAG system
        rag = RAG.from_config(rag_config)

        # Evaluate on dataset
        scores = []
        for sample in self.dataset:
            output = rag.query(sample.query)
            metric = self.evaluator.evaluate(output, sample.expected_answer)
            scores.append(metric.value)

        return sum(scores) / len(scores)
```

**Purpose**: Orchestrate optimization loop

---

## 🔌 Extension Points

### Adding a New Retriever

```python
# 1. Define config
@dataclass
class MyRetrieverConfig(BaseConfig):
    param1: str
    param2: int

# 2. Create wrapper
class MyRetriever(BaseRetriever):
    def __init__(self, config: MyRetrieverConfig):
        self.config = config

    def retrieve(self, query: str) -> List[str]:
        # Implementation
        pass

# 3. Add to search space
@dataclass
class RetrieverConfigSpace:
    retriever_type: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            choices=["VectorIndexRetriever", "BM25Retriever", "MyRetriever"]  # ← Add here
        )
    )

    def sample(self, trial: optuna.Trial):
        retriever_type = self.retriever_type.sample(trial)

        if retriever_type == "MyRetriever":
            return MyRetrieverConfig(
                param1=...,
                param2=...
            )
        # ...
```

---

### Adding a New Evaluator

```python
# 1. Inherit from BaseEvaluator
from rago.eval import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def evaluate(self, output: RAGOutput, expected: str) -> Metric:
        # Your evaluation logic
        score = compute_score(output.answer, expected)

        return Metric(
            name="my_metric",
            value=score,
            metadata={"details": "..."}
        )

# 2. Use in optimization
optimizer = SimpleDirectOptunaManager.from_dataset(
    evaluator=MyEvaluator(),  # ← Your evaluator
    metric_name="my_metric",
    # ...
)
```

---

<!-- ## 🔗 Related Documentation

- [Configs](configs.md) - Detailed config reference
- [Wrappers](wrappers.md) - Wrapper implementation guide
- [Search Spaces](search_spaces.md) - Parameter space definition
- [Managers](managers.md) - Optimization manager details

--- -->
