# RAG - Retrieval Augmented Generation Concepts

---

## 🤖 What is RAG (Retrieval Augmented Generation)?

### The Basic Idea

**RAG** enhances language models by giving them access to external knowledge.

#### Without RAG:
```
User: "What are the safety procedures for Model X-500?"
LLM: "I don't have specific information about Model X-500..."
```

#### With RAG:
```
User: "What are the safety procedures for Model X-500?"

[System retrieves relevant documentation]

LLM: "According to the safety manual, Model X-500 requires:
1. Lock-out tag-out procedures...
2. Personal protective equipment including...
3. ..."
```

## 🏯 RAG Pipeline

```
┌──────────────────┐
│      Query       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   RETRIEVAL      │ ← Search through docs
│  - Vector Search │
│  - BM25, ...     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Retrieved Docs  │ ← Docs for context
│  [Doc1, Doc2...] │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   GENERATION     │ ← LLM uses docs to answer
│   (LLM + Docs)   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│      Answer      │
└──────────────────┘
```

###  Why RAG?

1. **Up-to-date Information**: Access current documents without retraining
2. **Domain Specificity**: Use your own proprietary knowledge
3. **Transparency**: See which documents informed the answer
4. **Reduced Hallucinations**: Grounded in actual documents

## ✅ RAG vs. Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Data Changes** | Update documents immediately | Requires retraining |
| **Cost** | Low (storage and some GPU) | High (much GPUs, time) |
| **Transparency** | Can cite sources | Black box |
| **Use Case** | Dynamic knowledge | Specialized behavior |

---

## ⚙️ RAG Configurations

The **RAG configuration** consists of:
- A **Splitter / Node Parser**: during pre-processing phase, the splitter splits the document into chunks (Not implemented at this stage).
- A **Retriever**: At inference, the retriever select chunks, create a context to answer a query.
- A **Reader / Synthesizer**: Still in inference, the reader use the selected chunks to answer the query.

> ⚠️ An infinite number of methods exist, only methods implemented at this stage are described in the following subsections.

Here you have more information about RAG Configuration Space in RAGO 👉 [RAG Configuration](rag_configuration.md)

### Splitter (not implemented yet)

> ⚠️ `Sentence Splitter`, `Recursive Character Text Splitter`, `Semantic Splitter` shall be implemented in next versions.

### Retriever

Once we have a set of chunks we can then use a retriever to select chunks useful to answer a given query.
More information on RAGO usage for the Retriever Configuration Spaces here 👉 [Retriever Methods](retriever.md)

### Reader
Classical reader method uses the retrieved chunks, compact it in a single context block and use the LLM to generate the query directly. More complex reader methods proposed by `llamaIndex` library such as `Refine`, `Compact & Refine`, `Tree Summarize` are implemented in the RAGO library.
More information here 👉 [Reader Methods](reader.md)

## 📄 RAG Output in RAGO

A `RAGOutput` as its name hints is the output of RAG to a query. It contains:

- The `answer` of the RAG to the query (`None` for a retriever alone).
- The `retrieved_context` by the RAG to answer the query (`None` for a reader alone).
Below is an example usage of the `RAGOutput` class:

```python
from rago.data_objects import RAGOutput, RetrievedContext

RAGOutput(answer="Thomas is 12.", retrieved_context=[RetrievedContext("Thomas is 12.")])
```
RAG outputs can be generated manually or with `get_output(query)` on any `TunableModel` (`RAG`, `Retriever` or `Reader`).

---

## 📚 Related Documentation

- ⚙️ [RAG Configurations](rag_configuration.md)
- 🔍 [Retriever Methods](retriever.md)
- 🤖 [Reader Methods](reader.md)


---
