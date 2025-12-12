# Optimization with Tree-structured Parzen Estimator (TPE)



## 🎯 How Does Optimization Work?

### The Challenge

A RAG system has **hundreds of configurable parameters**:

```python
# Just a few examples:
retriever_type = ["vector", "bm25", "hybrid"]
embedding_model = ["bge-m3", "e5-large", "qwen-embedding", ...]
top_k = [1, 2, 3, 4, 5, ...]
temperature = [0.0, 0.1, 0.2, ..., 2.0]
top_p = [0.1, 0.2, ..., 1.0]
# ... and many more!
```
**Total possible combinations**: Millions! 😱

### A Solution: Bayesian Optimization

Instead of trying random configurations, RAGO uses **intelligent search**:

```
1. Try a few random configurations
2. Evaluate their performance
3. Build a probabilistic model of "what works"
4. Use the model to suggest the next configuration to try
5. Repeat until convergence
```

---

## 🧠 Bayesian Optimization Framework

### The Mathematical Problem

**Goal**: Find configuration $x^*$ that maximizes objective function $f(x)$

$$x^* = \arg\max_{x} f(x)$$

Where:
- $x$ = RAG configuration (retriever type, embeddings, LLM params, etc.)
- $f(x)$ = performance score (BERTScore, LLM-as-Judge, etc.)

**Challenge**: $f(x)$ is **expensive to evaluate** (requires running RAG + evaluation on full dataset)

### The Bayesian Approach

Instead of blindly testing configurations, Bayesian optimization:

1. **Builds a surrogate model** $\hat{f}(x)$ that approximates $f(x)$
2. **Uses the model** to decide where to evaluate next (acquisition function)
3. **Updates the model** with new results
4. **Repeats** until convergence

```
┌─────────────────────────────────────────────────────┐
│  Bayesian Optimization Loop                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Surrogate Model: P(score | config)              │
│                                                     │
│  2. Acquisition Function: Which config to try next? │
│                                                     │
│  3. Evaluate: Run RAG with suggested config         │
│                                                     │
│  4. Update Model: Incorporate new result            │
│                                                     │
│  5. Repeat until convergence                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Example iteration**:

```python
Iteration 1:  Try random config x₁ → score = 0.65 → update model
Iteration 2:  Model suggests x₂  → score = 0.72 → update model
Iteration 3:  Model suggests x₃  → score = 0.68 → update model
Iteration 4:  Model suggests x₄  → score = 0.81 → update model
...
```

### Acquisition Functions

**Question**: How do we decide where to evaluate next?

**Expected Improvement (EI)**: A common acquisition function

$$\text{EI}(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$$

Where $f(x^+)$ is the **best value found so far**.

**Interpretation**: "How much improvement do we expect at this point?"

**Trade-off**:
- **Exploitation**: Try configs similar to current best (high expected score)
- **Exploration**: Try different configs to avoid local optima (high uncertainty)

---

## 🌲 Tree-structured Parzen Estimator (TPE)

RAGO uses **TPE** by default, a powerful Bayesian optimization algorithm designed for hyperparameter tuning.

### How TPE Works

**Key Insight**: Instead of modeling $P(\text{score}|\text{config})$ directly, TPE models two distributions:

1. $l(x)$ = $P(\text{config} | \text{score} \geq y^*)$ → "good" configurations
2. $g(x)$ = $P(\text{config} | \text{score} < y^*)$ → "bad" configurations

Where $y^*$ is a threshold (e.g., top 25% of scores).

### TPE Algorithm Steps

```
1. Separate trials into "good" and "bad" based on threshold y*

2. Model parameter distributions:
   l(x) = distribution of params in GOOD trials
   g(x) = distribution of params in BAD trials

3. Sample next configuration by maximizing:

   x_next = argmax [ l(x) / g(x) ]

   Meaning: "Params that appear often in good trials
            but rarely in bad trials"

4. Evaluate f(x_next) and add to trial history

5. Repeat
```

### Intuitive Example

Imagine searching for treasure in a field:

| Strategy | Approach |
|----------|----------|
| **Random Search** | Dig holes randomly everywhere |
| **Grid Search** | Dig in a systematic grid pattern |
| **TPE** | Notice treasures found near water → focus there, but occasionally explore dry areas (in case of hidden patterns) |

### Mathematical Formulation

TPE maximizes the **Expected Improvement** using:

$$\text{EI}(x) \propto \frac{l(x)}{g(x)}$$

**Why this works**:
- High $l(x)$: Config $x$ appears often in successful trials
- Low $g(x)$: Config $x$ appears rarely in failed trials
- High ratio → high probability of improvement

### Exploration vs. Exploitation

Good optimization balances two strategies:

- **Exploitation**: Focus on areas known to work well, _i.e._ high $l(x)$
- **Exploration**: Try new areas to avoid missing better solutions, _i.e._ high uncertainty in $g(x)$

TPE naturally balances both through:
1. **Startup trials** ($n_{\text{startup}}$): Pure random exploration
2. **Probabilistic modeling**: Uncertainty in $l(x)$ and $g(x)$ encourages exploration
3. **Threshold $y^*$**: Controls exploration-exploitation trade-off

### TPE vs. Other Methods

| Method | Exploration | Categorical Params | Conditional Params | Speed |
|--------|-------------|-------------------|-------------------|-------|
| **Random Search** | ✅ Excellent | ✅ Yes | ✅ Yes | ⚡ Fast |
| **Grid Search** | ❌ Systematic only | ✅ Yes | ❌ Difficult | 🐌 Very Slow |
| **Gaussian Process** | ✅ Good | ❌ Difficult | ❌ Difficult | 🐌 Slow (high-dim) |
| **TPE** | ✅ Good | ✅ Yes | ✅ Yes | ⚡ Fast |

**TPE vs Random Search**: While Random Search explores uniformly across the entire parameter space, TPE **learns from previous trials** to focus exploration on promising regions. Random Search needs many more trials to find good configurations by chance, whereas TPE achieves better results with fewer trials by modeling which parameter combinations lead to success. For RAG optimization with expensive evaluations (each trial requires running inference on the full dataset), TPE's sample efficiency—typically finding good configurations in 30-50 trials vs 100+ for Random Search—translates to significant time savings.

**Why TPE for RAG optimization?**
- Handles **categorical** parameters (retriever type, embedding model)
- Handles **conditional** parameters (BM25 params only when using BM25, LLM params $\eta$ and $\tau$ if mirostat not None)
- **Fast** even with many parameters
- **Proven** on hyperparameter tuning tasks

### Convergence

Optimization stops when:
1. **Max iterations** reached (`n_iter`)
2. **No improvement** for N consecutive trials
3. **Target metric** achieved (optional early stopping)

```
Trials:  1   2   3   4   5   6   7   8   9   10
Score:  .30 .45 .52 .68 .79 .80 .85 .86 .87 .87
        ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   -

        Large improvements → Small improvements → Convergence
```

---

## 📚 Further Reading

### Research Papers

- **TPE Algorithm**: Bergstra et al. 2011 - [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- **TPE Deep Dive**: Watanabe 2023 - [Tree-Structured Parzen Estimator: Understanding Its Algorithm Components](https://arxiv.org/abs/2304.11127)
- **Bayesian Optimization**: Shahriari et al. 2016 - [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://ieeexplore.ieee.org/document/7352306)

### Related Documentation

- [Run your optimization](run_experiment.md)
- [RAG Configurations](../rag/rag_configuration.md)

---
