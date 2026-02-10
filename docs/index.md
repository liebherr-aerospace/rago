# RAGO Documentation

Welcome to the RAGO (Retrieval Augmented Generation Optimizer) documentation!

[:fontawesome-brands-github: View on GitHub](https://github.com/liebherr-aerospace/rago){ .md-button }

## 📚 Documentation Structure

```
    📁 docs/
    ├── 📁 code_architecture                 # Code Architecture
    │   ├── 📄 overview.md
    ├── 📁 installation                      # Installation guide
    │   ├── 📄 elasticsearch.md
    │   └── 📄 ollama.md
    └── 📁 usage_guide                       # Usage guide
        ├── 📁 dataset                       # Generate and load datasets
        │   ├── 📄 data_loader.md
        │   └── 📄 generator.md
        ├── 📁 evaluation                    # Evaluation and metrics
        │   └── 📄 metrics.md
        ├── 📁 optimization                  # Optimization methods and strategies
        │   ├── 📄 run_experiment.md
        │   └── 📄 tpe.md
        └── 📁 rag                           # RAG concepts, configurations and components
            ├── 📄 rag_concepts.md
            ├── 📄 rag_configuration.md
            ├── 📄 reader.md
            └── 📄 retriever.md
```

---

## 🎯 Quick Navigation

### 🚀 Getting Started

- **[Installation](installation/ollama.md)** - Setup & ollama configuration
- **[Quick Start](usage_guide/optimization/run_experiment.md)** - Your first optimization

### 📖 Core Documentation

- **[RAG Concepts](usage_guide/rag/rag_concepts.md)** - Understanding RAG
- **[RAG Configuration](usage_guide/rag/rag_configuration.md)** - Parameters & search space
- **[Retriever](usage_guide/rag/retriever.md)** - Retrieval methods
- **[Reader](usage_guide/rag/reader.md)** - Generation strategies

### ⚙️ Optimization

- **[Run Optimization](usage_guide/optimization/run_experiment.md)** - Optimization parameters and strategies
- **[TPE Algorithm](usage_guide/optimization/tpe.md)** - Bayesian optimization theory

### 🔧 Evaluation & Datasets

- **[Evaluators](usage_guide/evaluation/evaluator.md)** - Evaluators overview (BertScore, SimilarityScore, LLM-as-Judge)
- **[Metrics](usage_guide/evaluation/metrics.md)** - Evaluation metrics
- **[Dataset Loader](usage_guide/dataset/data_loader.md)** - Dataset loading and format
- **[Dataset Generator](usage_guide/dataset/generator.md)** - Dataset generators

---

## 🔬 Core Concepts

**RAG (Retrieval Augmented Generation)** combines:
1. **Retrieve** relevant documents from knowledge base
2. **Augment** LLM prompt with context
3. **Generate** informed answers

**RAG Optimization** automatically finds the best configuration (retriever, embeddings, LLM params) for your use case using **Bayesian Optimization**.

→ **Learn more**: [RAG Concepts](usage_guide/rag/rag_concepts.md) | [Config Space](usage_guide/rag/rag_configuration.md)

---


## 📖 External Resources

### Research Papers
- [Tree-structured Parzen Estimator](https://arxiv.org/html/2304.11127v4) - TPE optimization algorithm
- [BERTScore](https://arxiv.org/abs/1904.09675) - Semantic evaluation metrics
- [LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) - Using LLMs for evaluation

### Related Projects
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [LangChain](https://www.langchain.com/) - LLM application framework
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLMs
- [Ollama](https://ollama.ai/) - Run LLMs locally

---

## 💡 Need Help?

- 💬 Ask in [GitHub Discussions](https://github.com/liebherr-aerospace/rago/discussions)
- 🐛 Report bugs in [Issues](https://github.com/liebherr-aerospace/rago/issues)

---
