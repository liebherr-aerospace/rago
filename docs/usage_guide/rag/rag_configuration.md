# RAG Configuration

## 🧩 RAG Key Components

The RAG configuration consists of:
- A Splitter/ Node Parser: during pre-processing phase, the splitter splits the document into chunks (Not implemented at this stage).
- A Retriever: At inference, the retriever select chunks, create a context to answer a query.
- A Reader/ Synthesizer: Still in inference, the reader use the selected chunks to answer the query.

> ⚠️ An infinite number of methods exist, only methods implemented at this stage are described in the following subsections.

### Splitter (not implemented yet)

> ⚠️ `Sentence Splitter`, `Recursive Character Text Splitter`, `Semantic Splitter` shall be implemented in next versions.

### Retriever

Once we have a set of chunks we can then use a retriever to select chunks useful to answer a given query. Most retriever methods use embedding models in order to compute similarity between the query and the chunks.
More information on RAGO usage for the Retriever Configuration Spaces here 👉 [Retriever Methods](retriever.md)

### Reader

Classical reader method uses the retrieved chunks, compact it in a single context block and use the `LLM` to generate the query directly. More complex reader methods proposed by `llamaIndex` library such as `Refine`, `Compact & Refine`, `Tree Summarize` are implemented in the RAGO library.
More information about LLM parameters and reader strategies here 👉 [Reader Methods](reader.md).

## 🔭 RAGO Configuration Space

### Default RAG Configuration Space

```python
from rago.optimization.search_space.rag_config_space import RAGConfigSpace

config_space = RAGConfigSpace()
```
When the `RAGConfigSpace` is instantiated without any argument, the retriever and reader default configuration spaces are respectively `RetrieverConfigSpace` and `LangchainReaderConfigSpace`:

```python
RetrieverConfigSpace(
    retriever_type_name=CategoricalParamSpace(name='RetrieverConfigSpace_retriever_type_name', choices=['VectorIndexRetriever']),
    similarity_function=CategoricalParamSpace(name='RetrieverConfigSpace_similarity_function', choices=['cosine']),
    search_type=CategoricalParamSpace(name='RetrieverConfigSpace_search_type', choices=['similarity_score_threshold']),
    top_k=IntParamSpace(name='RetrieverConfigSpace_top_k', low=1, high=5, step=1, log=False),
    score_threshold=FloatParamSpace(name='RetrieverConfigSpace_score_threshold', low=0.0, high=0.9, step=None, log=False),
    encoder=HFEncoderConfigSpace(model_name=CategoricalParamSpace(name='encoder_model_name', choices=['BAAI/bge-m3', 'intfloat/e5-large-v2', 'Qwen/Qwen3-Embedding-0.6B', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2']))
)
```
```python
LangchainReaderConfigSpace(
    llm_config=OllamaLLMConfigSpace(
        model_name=CategoricalParamSpace(name='OllamaLLMConfigSpace_model_name', choices=['smollm2:1.7b', 'qwen3:4b', 'gemma3:4b', 'llama3.2:3b']),
        temperature=FloatParamSpace(name='OllamaLLMConfigSpace_temperature', low=0.0, high=1.0, step=None, log=False),
        top_k=IntParamSpace(name='OllamaLLMConfigSpace_top_k', low=0, high=10000, step=1, log=False),
        top_p=FloatParamSpace(name='OllamaLLMConfigSpace_top_p', low=0.0, high=1.0, step=None, log=False),
        max_new_tokens=IntParamSpace(name='OllamaLLMConfigSpace_max_new_tokens', low=64, high=1024, step=1, log=False),
        mirostat=CategoricalParamSpace(name='OllamaLLMConfigSpace_mirostat', choices=[0, 1, 2]),
        mirostat_eta=FloatParamSpace(name='OllamaLLMConfigSpace_mirostat_eta', low=0.0, high=1.0, step=None, log=False),
        mirostat_tau=FloatParamSpace(name='OllamaLLMConfigSpace_mirostat_tau', low=0.0, high=1.0, step=None, log=False),
        num_ctx=IntParamSpace(name='OllamaLLMConfigSpace_num_ctx', low=64, high=12800, step=64, log=False),
        repeat_last_n=IntParamSpace(name='OllamaLLMConfigSpace_repeat_last_n', low=-1, high=256, step=1, log=False),
        base_url='TEST_OLLAMA_HOST', client_kwargs={'verify': False})
)
```

### Pimp your RAG Configuration Space

#### Retriever methods

```python
from rago.optimization.search_space.param_space import CategoricalParamSpace
from rago.optimization.search_space.reader_config_space import ReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

# By default - Vector Index retriever only
config_space = RAGConfigSpace(retriever_space=RetrieverConfigSpace())

# BM25Retriever only
config_space = RAGConfigSpace(RetrieverConfigSpace(
    retriever_type_name=CategoricalParamSpace(choices=["BM25Retriever"])
    )
)
# Vector Index and BM25 retrievers in configuration space
config_space = RAGConfigSpace(RetrieverConfigSpace(
    retriever_type_name=CategoricalParamSpace(choices=["VectorIndexRetriever", "BM25Retriever"])
    )
)
```
#### Reader methods

You can specify and choose the reader method from `llama Index` or `Langchain` as described here 👉 [Reader Methods](reader.md#reader-methods-and-strategies)

> Regarding the retriever and reader methods it is also possible to modify the default values of the configuration spaces by specifying the class arguments directly.

## 📚 Related Documentation

- ⚙️ [RAG Concepts](rag_concepts.md)
- 🔍 [Retriever Methods](retriever.md)
- 🤖 [Reader Methods](reader.md)
