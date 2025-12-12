# Reader Methods and strategies


## 🎲 Language Model Parameters

### Definition

The LLM computes the probability distribution over the vocabulary for the next token, given the previously generated tokens. During inference, we sample the next token $x_t$ according to this distribution.

For a sequence $x_1, \ldots, x_{t-1}$ and a candidate token $w_i$ from vocabulary $V$:

$$P(x_t = w_i \mid x_1, \ldots, x_{t-1}) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

where:
- $x_1, \ldots, x_{t-1}$: Previously generated tokens (context)
- $x_t$: Next token to generate
- $w_i$: The $i$-th token in vocabulary $V$ (candidate)
- $z_i = f_\theta(x_1, \ldots, x_{t-1})_i$: Logit (raw score) for token $w_i$ from the model
- $T$: Temperature parameter controlling randomness
- $|V|$: Vocabulary size

The parameter of the models are described in the following subsections.

### Temperature

- The idea of the temperature parameter $T$ is to modify the next token distribution probability describe in (1).

- As $T$ increases $∀ x ∈ R^{d_{model}}$, $\frac{x}{T}$ converge to $0$. This means the distribution converge to the uniform distribution as $T$ goes to infinity.

- Therefore, higher is the Temperature parameter the more noise it adds; When $T$ goes to $0$ the distribution becomes deterministic and gives all the density to the word with the highest probability. In this case we will always choose,

$$j^* = argmax_{j=1}^{|V|} z_j$$

### Top-k Sampling

$top_k$ allow to sample only from the $k$ most likely next tokens.

> ⚠️ $k$ is fixed
> - When model is confident: $k$ might be too large
> - When model is uncertain: $k$ might be too small

### Top-p Sampling (Nucleus Sampling)

- $top_p$ allow to sample from the smallest set of tokens whose cumulative probability exceeds $p$.
- Indeed the next token is also sampled from a restricted set of token. This sets corresponds to the $top_k$ token such that $\sum_{i=1}^k p_i=top_p$.
- The advantage of this method compared to $top_k$ is that the size of the set $k$ the next token is sampled from depends on the distribution. Therefore it leverages more information learned by the LLM.


### Mirostat

Advanced sampling algorithm that maintains target **perplexity** (uncertainty).

**Parameters**:
- $\tau$: Target perplexity
- $\eta$: Learning rate for adjustment

**Benefit**: More consistent output quality across different prompts

### Repetition Penalty

One issue with LLM is that tend to repeat themselves. A method to reduce repetition is to penalize token that have already been used.
The parameters of this method are:

**Parameters**:
- **repeat_last_n**: How many recent tokens to consider (e.g., 64)
- **repeat_penalty**: Penalty multiplier (e.g., 1.1 = 10% penalty)

**Effect**:
```
Without penalty:
"The system is important. The system is crucial. The system is..."

With penalty:
"The system is important. It plays a crucial role. This component..."
```

### Context Window (num_ctx)

Maximum number of tokens the model can process at once.
For instance:
```
num_ctx = 2048
num_ctx = 4096
num_ctx = 8192
```

**Trade-off**:
- Larger: More context, better understanding, slower, more memory
- Smaller: Faster, less memory, might miss important context


## 🎯 Reader strategies

The Reader mode as explained below influences how the chunks passed to the LLM are consumed by the LLM:
- **Compact**: All the chunks are consumed at the same time by the LLM.
- **Refine**: The chunks are consumed one at a time by the LLM, each time an answer is generated. The preceding answer is passed with current chunk as input to the LLM.
- **Tree summarize**: The Chunks are grouped together in groups of same size. Each groups are then summarize. The process is then repeated with the summaries iteratively until were each a certain number of summaries. The final set of summaries is used to generated the final answer.
- **Simple summarize**: The Chunks are grouped together then summarized. The summary is used as a context to generated the final answer.

More information on the reader factory of Llama Index here 👉 [Reader Factory](../../../rago/model/constructors/llama_index/reader_factory.py).

## 🤖 LLM Usage

### Instantiation

#### Langchain

To instantiate an ollama langchain model you need the only required parameter is the name of the model `model_name`:

```python
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgentFactory

llm_config = LangchainOllamaConfig(model_name="phi3:3.8b-mini-128k-instruct-q8_0")
llm_agent = LLMAgentFactory.make(llm_config)
llm_agent.query("Who is the most famous president of the united states?")
```
#### LlamaIndex

Similarly, to instantiate an ollama llama_index model you need to provide the name of the model `model_name`. The only difference is the config class to use: `LlamaIndexOllamaConfig` instead of `LangchainOllamaConfig`.

```python
from rago.model.configs.llm_config.llama_index import LlamaIndexOllamaConfig
from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgentFactory

llm_config = LlamaIndexOllamaConfig(model_name="phi3:3.8b-mini-128k-instruct-q8_0")
llm_agent = LLMAgentFactory.make(llm_config)
llm_agent.query("Who is the most famous president of the united states?")
```

### Usage

Any Language model can be:
- queried with a `string query`:
    ```python
    query = "What's the weather in Toulouse?"
    llm_agent.query(query)
    ```
- chatted to with a `message_sequence`:
    ```python
    from rago.model.wrapper.llm_agent.message import Message, Role

    message_sequence = [
        Message("What's the weather like in toulouse?", Role.USER),
        Message("The weather is great", Role.BOT),
        Message("Can you tell me more?", Role.USER),
        ]
    llm_agent.chat(message_sequence).text
    ```
## 📗 Reader

### Simple Reader

The simple reader uses a language model to answer the query. The retrieved context and the query are added to the reader's prompt and passed to the language model to generate an answer.
Below is an example usage of the simple reader using langchain:

```python
from rago.model.configs.reader_config.langchain import LangchainReaderConfig
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.wrapper.reader.langchain_reader import LangchainWrapperReader
from rago.data_objects import RetrievedContext

retrieved_context = [RetrievedContext("The date is 2012"), RetrievedContext("Thomas is going to be 12 in 2013")]
llm_config = LangchainOllamaConfig(model_name="phi3:3.8b-mini-128k-instruct-q8_0")
reader_config = LangchainReaderConfig(llm=llm_config)
reader = LangchainWrapperReader.make(config = reader_config)
print(reader.get_reader_output("hello", retrieved_context))
```

### Other Reader Strategies

Using llama_index allow us to choose more complex reader strategies such as `CompactAndRefine`.
To do so it is possible to add to the `LLamaIndexReaderConfig`.
The section [Llama Index Reader Configuration Space](#llama-index-reader-configuration-space) explains that if you use the llama Index configuration Space by default it will include the reader strategies instead of using langchain.


## 🔭 Reader Configuration Space

### Default Reader Configuration Space

By default we use Langchain LLM config space:

```python
  model_name: CategoricalParamSpace = Field(
      default=CategoricalParamSpace(
          choices=["smollm2:1.7b", "qwen3:4b", "gemma3:4b", "llama3.2:3b"],
      ),
  )
  mirostat: CategoricalParamSpace = Field(
      default=CategoricalParamSpace(
          choices=[0, 1, 2],
      ),
  )
  mirostat_eta: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
  mirostat_tau: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
  num_ctx: IntParamSpace = Field(default=IntParamSpace(low=64, high=12800, step=64))
  repeat_last_n: IntParamSpace = Field(default=IntParamSpace(low=-1, high=256))
  base_url: str = Field(default=os.environ.get("TEST_OLLAMA_HOST", ""))
```
See 👉 [Default LLM Configuration Space](../../../rago/optimization/search_space/llm_config_space.py)

### Llama Index Reader Configuration Space

In order to use the Llama index reader, use this reader config space:
```python
from rago.optimization.search_space.reader_config_space import LlamaIndexReaderConfigSpace
reader_config_space = LlamaIndexReaderConfigSpace()
```

The default configuration is the same for the LLM parameters but reader modes are added:

```python
  type_config: CategoricalParamSpace = Field(
      default=CategoricalParamSpace(
          choices=["Refine", "CompactAndRefine", "TreeSummarize", "SimpleSummarize"],
      ),
  )
```

---

## 📚 Related Documentation

- ⚙️ [RAG Configurations](rag_configuration.md)
- 🤖 [Retriever Methods](retriever.md)

---
