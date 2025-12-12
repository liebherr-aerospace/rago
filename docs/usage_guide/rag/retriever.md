# Retriever methods

Once we have a set of chunks we can then use a retriever to select chunks useful to answer a given query.

## 🔢 VectorIndexRetriever
The `Vector Index Retriever` is the basic retriever. The idea is to use an embedding model to compute a similarity score using cosine similarity between each chunks and the query.
> The retrieved chunks are the k chunks with the highest similarity score. Hence, the parameters are:
> - **top k** is number of retrieved chunks
> - **similarity score threshold**: the threshold between 0 and 1 in which we keep the retrieved document.
> - **embedding**: encoder language model for embedding the documents and queries

To find relevant documents, we measure **cosine similarity**:

$$\text{similarity}(A, B) = \frac{A \cdot B}{|A| \times |B|}$$

Where:
- $A \cdot B$ = dot product, _i.e._ how aligned the vectors are, 1 means "identical meaning" therefore 0 means "unrelated".
- $|A|$ = length of vector A
- $|B|$ = length of vector B

## 📊 BM25 Retriever
`BM25` stands for Best Match 25. The BM25 retriever scores each document based on the terms (words) it shares with the query using the following formula (2):

$$BM25(D, Q) = \sum_{i=1}^{n}IDF(q_i)·\frac{TF(q_i, D)·(k_1+ 1)}{TF(q_i, D) +k_1·(1−b+b·\frac{|D|}{avgdl})}$$

where:
- $q_i$: The $i$-th term in the query $Q$.
- $n$: The total number of terms in the query $Q$.
- $TF(q_i, D)$: Term Frequency, i.e., the number of times term $q_i$ appears in document $D$.
- $|D|$: The length of the document $D$ (_e.g_., the number of words in $D$).
- $avgdl$: The average document length across all documents in the corpus.
- $k1$: A tuning parameter that controls the saturation of term frequency, typically set between $1.2$ and $2.0$.
- $b$: A tuning parameter that controls the degree of length normalization, typically set around $0.75$.
- $IDF(q_i)$: Inverse Document Frequency, calculated as:

$$IDF(q_i) = log\left( \frac{N−DF(q_i) + 0.5}{DF(q_i) + 0.5} + 1 \right)$$

where
- $N$ is the total number of documents
- $DF(q_i)$ is the number of documents containing the term $q_i$.

### When to Use BM25

**Good for:**
- Exact term matching (product codes, model numbers)
- Queries with rare, specific keywords
- Multilingual scenarios (no pre-trained embeddings needed)
- Explainable retrieval (can see why doc matched)

**Less good for:**
- Synonyms ("car" vs "automobile")
- Paraphrasing
- Semantic similarity

## ✅ Vector vs BM25

| Aspect | Vector Search | BM25 |
|--------|---------------|------|
| **Type** | Semantic | Lexical |
| **Synonyms** | ✅ Understands | ❌ Misses |
| **Exact Match** | Sometimes misses | ✅ Perfect |
| **Speed** | Fast (with indexing) | Very fast |
| **Setup** | Needs embedding model | No training needed |
| **Multilingual** | Model-dependent | Works naturally |

**Best practice**: Use **hybrid retrieval** (combine both)!

---

> ⚠️ `Hybrid Retriever`, `Cluster Based Retriever` shall be implemented in next versions.

## 📄 Retrieved Context

The retrieved context is a list of `RetrievedContext`. A retrieved context contains:
- a text: chunk of information (e.g a document, a sentence or paragraph.)
- with optionally its relevancy (the score) and its embedding (vector representations of the text)

Below is an example usage of the ContextNode class:

```python
from rago.data_objects import RetrievedContext

RetrievedContext(text ="Thomas will be twelve in 2012.", embedding= [0, 0, 0, 0, 0, 0, 0], score = 0.8)
```
> see [Retrieved Context Dataclass](../../../rago/data_objects/retrieved_context.py) for more details

## 🔍 Retriever Wrappers

The retriever wrapper is a key component of the RAG pipeline it takes in the query and returns a context it deems relevant to answer the query.
The retriever wrapper can be instantiated as follow:

```python
from rago.model.wrapper.retriever.langchain_retriever import LangchainRetrieverWrapper
from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig
from rago.model.configs.encoder_config.langchain import HuggingFaceLangchainEncoderConfig

encoder_config = HuggingFaceLangchainEncoderConfig(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
config = LangchainRetrieverConfig(
    type = "VectorIndexRetriever",
    similarity_function="cosine",
    search_type="similarity_score_threshold",
    search_kwargs={"k":3, "score_threshold" : 0.0},
    encoder = encoder_config,
    )
retriever = LangchainRetrieverWrapper.make(config, inputs_chunks=["Thomas is 12."])
print(retriever.get_retriever_output("Hello"))
```

## 🧭 Retrievers Configuration Space in RAGO

In order to have a deeper look on the retriever configuration spaces implemented in RAGO, see the following code 👉 🐍 [Retriever Configuration Space python methods](../../../rago/optimization/search_space/retriever_config_space.py).
All the following parameter values can be adapted as you need, and you can combine or choose one of the methods.

### Default Retriever Configuration Space

By default the Retriever Configuration Space used in RAGO is based on the `VectorIndexRetriever` method with these default parameter values:
```python
similarity_function: CategoricalParamSpace = Field(default=CategoricalParamSpace(choices=["cosine"]))
search_type: CategoricalParamSpace = Field(default=CategoricalParamSpace(choices=["similarity_score_threshold"]))
top_k: IntParamSpace = Field(default=IntParamSpace(low=1, high=5))
score_threshold: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=0.9))
encoder: HFEncoderConfigSpace = Field(default=HFEncoderConfigSpace())
```
By default the encoder configuration space used with this retriever methods is defined by:
```python
    model_name: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            name="encoder_model_name",
            choices=[
                "BAAI/bge-m3",
                "intfloat/e5-large-v2",
                "Qwen/Qwen3-Embedding-0.6B",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ],
        ),
    )
```
See the following code 👉 🐍 [Encoder Configuration Space python methods](../../../rago/optimization/search_space/encoder_config_space.py) for more details.

### BM25 Configuration Space

Here is the default configuration space fo the BM25 retrievers:
```python
bm25_k1: FloatParamSpace = Field(default=FloatParamSpace(low=1.2, high=2.0))
bm25_b: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
bm25_similarity: CategoricalParamSpace = Field(default=CategoricalParamSpace(choices=["custom_bm25"]))
```
You can find the values $k1$ and $b$ defined in the formula (2).

---

## 📚 Related Documentation

- ⚙️ [RAG Configurations](rag_configuration.md)
- 🤖 [Reader Methods](reader.md)

---
