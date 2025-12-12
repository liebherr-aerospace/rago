# Dataset Generator

## 📦 Overview

The **Dataset Generator** automatically creates question-answer pairs from your documents using an LLM. This is useful in two scenarios:

1. **No existing dataset**: You only have documents raw Documents but no annotated question-answer pairs for evaluation.
2. **Dataset augmentation**: You have an existing QA dataset and a corpus but want to generate additional samples from the same or new documents to increase coverage.

Instead of manually creating evaluation samples, the generator leverages an LLM to:
- Analyze your documents
- Generate relevant questions
- Extract accurate answers from the text
- Create structured `EvalSample` objects ready for optimization

**Input**: Raw documents or existing dataset
**Output**: `RAGDataset` with generated question-answer pairs

## ⚙️ Simple Generator

The simplest way to instantiate a `SimpleDatasetGenerator` is the following:

```python
from rago.dataset.generator import SimpleDatasetGenerator

dataset_generator = SimpleDatasetGenerator()
```

It will use an Ollama Langchain LLM agent with default parameters.

Additionally, we can either:

- Pass an LLM config to the make method of `SimpleDatasetGenerator` to make a custom LLM specifically for the Dataset Generator:
    ```python
    from rago.dataset.generator import DatasetGeneratorConfig, SimpleDatasetGenerator
    from rago.model.configs.llm_config.langchain import LangchainOllamaConfig

    llm_config = LangchainOllamaConfig(model_name="smollm2:1.7b", temperature= 0.0)
    dataset_generator_config = DatasetGeneratorConfig(llm_config)

    dataset_generator = SimpleDatasetGenerator.make(dataset_generator_config)
    ```
- Or pass an existing LLM Agent to the `SimpleDatasetGenerator` (e.g to share a single LLM between multiple module):
    ```python
    from rago.dataset.generator import SimpleDatasetGenerator
    from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
    from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgentFactory

    llm_config = LangchainOllamaConfig(model_name="phi3:3.8b-mini-128k-instruct-q8_0")
    llm_agent = LLMAgentFactory.make(llm_config)
    dataset_generator = SimpleDatasetGenerator(llm_agent)
    ```
> You can add and customize your own dataset generator that inherits from the class [BaseDatasetGenerator](../../../rago/dataset/generator/base.py)

## 💭 Generate from Dataset

To generate a dataset from `seed_data`:
```python
from rago.data_objects import Document

generated_dataset = dataset_generator.generate_dataset(sampled_rag_ds)
```
To generate a dataset from a `seed_data` and directly save the generated_dataset:

```python
dataset_generator.generate_and_save_dataset("./synth_rag_dataset.json", sampled_rag_ds)
```

## 📄 Generate from Documents

```python
from rago.dataset.generator import SimpleDatasetGenerator
from rago.data_objects import Document

# Raw documents
documents = [
    Document(text="Safety procedures for Model X-500: Always wear protective equipment..."),
    Document(text="Maintenance schedule: Weekly inspection of hydraulic systems..."),
    Document(text="Emergency shutdown: Press red button and follow evacuation protocol...")
]

# Generator creation
dataset_generator = SimpleDatasetGenerator(
    number_questions_per_document=2
)

# Dataset Generation
generated_dataset = dataset_generator.generate_dataset(seed_data=documents)

print(f"Samples {generated_dataset.samples}")
```
The output of the generated dataset is a `RAGDataset` object and the samples are `EvalSample` type (see [Eval Sample Class](../../../rago/data_objects/eval_sample.py))

---

## 📚 Related Documentation

- [Data Loader](data_loader.md)
- [Run your first experiment](../optimization/run_experiment.md)

---
