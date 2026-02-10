<a id="top"></a>
<div align="center">
  <img src="docs/assets/img/rago_img_3.png" alt="Retrieval Augmented Generation Optimizer" width="300" style="margin-right: 10px;">

  <h1>Retrieval Augmented Generation Optimizer</h1>

  <img src="https://www-assets.liebherr.com/media/global/global-media/liebherr_logos/logo_ci_liebherr.svg" alt="Liebherr Logo" width="180"/>

  <!-- Badges -->
  <p>
    <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-%3E%3D3.12-blue.svg" alt="Python ≥ 3.12"/>
    </a>
    <a href="">
      <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"/>
    </a>
    <a href="https://liebherr-aerospace.github.io/rago/">
      <img src="https://img.shields.io/badge/Documentation-online-brightgreen.svg" alt="Documentation"/>
    </a>
  </p>

  <p><b>Automatically optimize your RAG system using intelligent Bayesian search</b></p>
</div>

---

## 🎯 What is RAGO?

**RAGO (Retrieval Augmented Generation Optimizer)** is an intelligent toolkit that automatically discovers the best configuration for your Retrieval Augmented Generation (RAG) system through smart experimentation.

<table>
<tr>
<td width="50%" valign="top">

### 🔴 Problem

Building an effective RAG system requires tuning **Lot's of parameters**:

- ❓ Which **retrieval method**? Vector search, BM25, hybrid, other?
- 📚 How many **documents** to retrieve? 3, 5, 10, with what threshold?
- 🤖 Which **embedding** model? BGE, E5, Qwen, others?
- 🌡️ Which **LLM parameters**? Temperature, top-k, top-p, context length?
- 📖 Which **reader strategy**? Simple, refine, tree-summarize?

</td>
<td width="55%" valign="top">

### 🟢 Solution

RAGO uses **Optimization Methods** (Bayesian methods for instance) to automatically:

- 📝 **Generate** evaluation datasets from your documents if you do not have manual annotations
- 🔬 **Test** different RAG configurations exploring the parameter space
- 📊 **Evaluate** answer quality using BERTScore, LLM-as-Judge, Relevancy metrics
- 🧠 **Learn** from each trial and focus on promising configurations
- 🏆 **Find** the optimal setup and the best performance for your documents

</td>
</tr>
</table>

---

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🧩 Flexible RAG Components
- **Retrievers**: Vector search (semantic), BM25 (keyword), and more
- **Readers**: Simple generation, refine, compact, tree-summarize (LangChain & LlamaIndex)
- **LLM Support**: Ollama integration

### 📝 Dataset Generation
- **Automatically generate** evaluation datasets from your documents if no dataset with manual annotations
- **Synthetic question-answer** pairs with context created using LLMs


</td>
<td width="55%" valign="top">

### 📊 Evaluation Metrics
- **BERTScore**: Semantic similarity evaluation
- **LLM-as-a-Judge**: AI-powered answer quality assessment
- **Relevancy**: Retrieved context documents compared to the source context evaluation

### ⚙️ Smart Optimization
- **Optimization Methods** from Optuna frameworks. Bayesian optimization is used with Tree-structured Parzen Estimator (TPE) by default.
- **Direct** and **Pairwise** optimization modes
- Built on the powerful **Optuna** framework
- **Interactive visualization** dashboard and **feature importance** computation

</td>
</tr>
</table>

---

## 📦 Installation

**Prerequisites**
- Python ≥ 3.12
- [Ollama](https://liebherr-aerospace.github.io/rago/installation/ollama/) for local LLM inference

```bash
# 1. Clone the repository
git clone https://github.com/liebherr-aerospace/rago.git
cd rago

# 2. Install dependencies using UV
pipx install uv
uv sync

# 3. Activate virtual environment
source .venv/bin/activate

# 4. (Optional) Install pre-commit hooks for development
uv pip install pre-commit
pre-commit install
```

**Environment Variables**

```bash
# For Ollama LLM inference
export TEST_OLLAMA_HOST="<your-ollama-endpoint>"

# (Optional) For BM25 retriever with OpenSearch/Elasticsearch
export OPENSEARCH_URL="<your-opensearch-endpoint>"
export OPENSEARCH_INDEX_NAME="<your-index-name>"
```

📖 See [Ollama Installation Guide](https://liebherr-aerospace.github.io/rago/installation/ollama/) for detailed instructions on Ollama set-up or [Elastic Installation Guide](https://liebherr-aerospace.github.io/rago/installation/elasticsearch/) on Elastic installation.

> 📘 **Full Documentation**: Visit our comprehensive documentation at [https://liebherr-aerospace.github.io/rago/](https://liebherr-aerospace.github.io/rago/)

---

## 🚀 Quick Start

Optimize a RAG system with default reader and retriever methods in just a few lines:

```python
from rago.dataset import QADatasetLoader, RAGDataset
from rago.eval import BertScore
from rago.optimization.manager import OptimParams, SimpleDirectOptunaManager
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

# Configure optimization
params = OptimParams(
    experiment_name="my_experiment",
    n_startup_trials=50,
    n_iter=1000,
)

# Load dataset and evaluator
full_ds = QADatasetLoader.load_dataset(RAGDataset, "crag").sample(5, 0, 10)
evaluator = BertScore()

# Split dataset into train/test
datasets = full_ds.split_dataset([0.8], split_names=["train", "test"], seed=42)

# Load RAG configuration space
config_space = RAGConfigSpace()

# Run optimization
optimizer = SimpleDirectOptunaManager(
    params=params,
    datasets=datasets,
    optim_evaluator=evaluator,
    optim_metric_name="bert_score_f1",
    test_evaluators=[evaluator],
    config_space=config_space,
)
optimizer.optimize()
```
By default, the RAGConfig space uses :
- default encoder models choices, vector index retriever methods and parameters $top_k$ and the similarity score threshold. See more here 👉 [Retriever Methods](https://liebherr-aerospace.github.io/rago/usage_guide/rag/retriever/)
- default LLM choices and LLM parameters $temperature$, $top_k$, $top_p$, max_new_tokens, num_ctx, repeat_last_n, mirostat parameters. See more here 👉 [Reader Methods](https://liebherr-aerospace.github.io/rago/usage_guide/rag/reader/)

📖 If you want to customize your RAG configuration space, read here [RAG Configurations](https://liebherr-aerospace.github.io/rago/usage_guide/rag/rag_configuration/)

**Visualize Results:**

```bash
optuna-dashboard sqlite:///experiments/my_experiment/study.db
```

> � **For detailed guides and API reference**, visit the full documentation at [https://liebherr-aerospace.github.io/rago/](https://liebherr-aerospace.github.io/rago/)

---

## 📚 Documentation Overview

<table>
<tr>
<td width="50%" valign="top">

### Getting Started
- 📖 [**Documentation Index**](https://liebherr-aerospace.github.io/rago/) - Start here!
- 🎓 [**Core Concepts**](https://liebherr-aerospace.github.io/rago/usage_guide/rag/rag_concepts/) - Understand RAG concepts
- ⚙️ [**Configuration Space**](https://liebherr-aerospace.github.io/rago/usage_guide/rag/rag_configuration/) - RAG configuration space details
- 🔍 [**Retriever Methods**](https://liebherr-aerospace.github.io/rago/usage_guide/rag/retriever/) - Retrieval strategies
- 🤖 [**Reader Methods**](https://liebherr-aerospace.github.io/rago/usage_guide/rag/reader/) - Generation strategies
- 🛠️ [**Run your Optimization**](https://liebherr-aerospace.github.io/rago/usage_guide/optimization/run_experiment/) - Detailed usage examples

</td>
<td width="55%" valign="top">

### Advanced Topics
- 🛠️ [**Optimization Methods**](https://liebherr-aerospace.github.io/rago/usage_guide/optimization/tpe/) - TPE algorithm details
- 🎯 [**Optimization Strategies**](https://liebherr-aerospace.github.io/rago/usage_guide/optimization/direct_pairwise/) - Direct vs Pairwise
- 📔 [**Dataset Generation**](https://liebherr-aerospace.github.io/rago/usage_guide/dataset/generator/) - (Query, Answer, Context) dataset Generation
- 👍 [**Evaluation Methods**](https://liebherr-aerospace.github.io/rago/usage_guide/evaluation/evaluator/) - LLM Evaluators
- 📊 [**Metrics**](https://liebherr-aerospace.github.io/rago/usage_guide/evaluation/metrics/) - Evaluation metrics
- 🏗️ [**Architecture**](https://liebherr-aerospace.github.io/rago/code_architecture/overview/) - System design and modules

</td>
</tr>
</table>

---

## 🧭 Roadmap

<table>
<tr>
<td width="50%" valign="top">

### 🚀 Coming Soon
- **Similarity based metrics**: metric based on similarity score
- **Configuration Files**: YAML/JSON for reproducible experiments
- **Advanced Retrievers**: Hybrid search, cluster-based retrieval, rerankers model
- **Document Splitters**: Semantic chunking, recursive splitting, and more


</td>
<td width="55%" valign="top">

### 🔮 Future Plans
- **Advanced HPO Strategies**: Iterative optimization with train/test/validation splits to reduce TPE bias
- **Multi-objective Optimization**: Pareto frontiers, weighted aggregation metrics, balancing performance vs. model size and energy consumption
- **Agentic Systems**: Agentic system optimization


</td>
</tr>
</table>

> **Note:** This roadmap reflects our current vision and priorities. It will evolve based on community feedback and open-source contributions. We welcome your input to shape RAGO's future!


💡 **Ideas?** [Open an issue](https://github.com/liebherr-aerospace/rago/issues) or contribute!

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. 🐛 **Report Bugs**: [Open an issue](https://github.com/liebherr-aerospace/rago/issues)
2. 💡 **Suggest Features**: [Share your ideas](https://github.com/liebherr-aerospace/rago/issues)
3. 📝 **Improve Docs**: Fix typos, add examples
4. 🔧 **Submit Code**: Fork, code, test, PR!


---

## 👥 Team

Developed by the **Datalab team at Liebherr Aerospace Toulouse** [(Liebherr Aerospace Website)](https://www.liebherr.com/en-int/aerospace-and-transportation-systems/liebherr-aerospace-transportation-7174865) in collaboration with **Institut de Recherche en Informatique de Toulouse** French Lab [(IRIT Website)](https://www.irit.fr/).

---

## ⚖️ License

Apache 2.0 License - See [LICENSE](LICENSE) for details.

Copyright © 2025 Liebherr Aerospace Toulouse SAS

---

## 📖 Citation

If you use RAGO in your research, please cite:

```bibtex
@software{rago2025,
  title={RAGO: Retrieval Augmented Generation Optimizer},
  author={Rehel, Briag and Bonneu, Adrien},
  year={2025},
  organization={Liebherr Aerospace Toulouse}
}
```

---

<div align="center">
  <p><i>Built with ❤️ by the Liebherr Aerospace Datalab Team</i></p>
  <p>
    <a href="#top">Back to Top ⬆️</a>
  </p>
</div>
