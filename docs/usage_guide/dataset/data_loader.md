# Datasets

## 📔 Load Datasets

To get all the available dataset:

```python
from rago.dataset import QADatasetLoader

QADatasetLoader.list_available_datasets()
```

Then, to load a particular dataset simply call the load_dataset of `QADatasetLoader`.

```python
from rago.dataset import QADataset

ds = QADataset.load_dataset("hotpot_qa")
```
The first argument is the class of the dataset you want to create `QAdataset` (i.e. only query and answer) or `RAGdataset` if you want the Retriever corpus as well:

```python
from rago.dataset import RAGDataset

rag_ds = RAGDataset.load_dataset("hotpot_qa")
```

When the dataset is first loaded it is saved to `cache_dir` (default to "./data" at root dir) or `save_path` if specified. Then, when loading the dataset again with same caching arguments it will be simply loaded from the cached version.

## 📐 Dataset Format

A `QAdataset` has samples of type `list[EvalSample]`. This is because one element taken from a dataset is called a sample. So an element of the evaluation dataset is an eval sample. An `EvalSample` contains contains both the information necessary to obtain the RAG output and the information necessary to evaluate it:

- the `query` is what we want the RAG output to answer. It used both to generate the rag output and to evaluate the RAG output (e.g to evaluate the relevance of the RAG output).
- the `context` (Optional) used by the generator to generate the answer. It is used to evaluate the RAG output (e.g to evaluate the correctness of the answer).
- the `explanations` (Optional) eventually given by the generator, it gives further guidance to the judge .
- the `reference_answer` (Optional) the correct answer to the query or an answer to compare to.
- the `reference_score` (Optional) the score of the reference.

Here an example usage of the EvalSample Object:

```python
from rago.data_objects import EvalSample

eval_sample = EvalSample("How old is thomas", context = ["Thomas is 12.", "Thomas was born 12 years ago."])
```

To quickly access one argument of all samples from a dataset one can directly access it from the dataset:

```python
ds["train"].query
```

This returns a list containing the query of each element in the dataset.

Finally sometimes we only want to use a subset of the dataset (for instance if the dataset is to big and apply the algorithm on the full dataset is prohibitively expansive). Then we can sample a smaller version of the dataset from the original dataset:

```python
sampled_ds = ds["train"].sample(size = 10, seed = 0)
```

For a `RAGDataset` we need to specify how the corpus should be constructed. It must contains all the reference documents otherwise some questions are unanswerable. But we can specify how many distractor documents (by opposition to ref documents) we want per sample with the param `max_num_distractor_documents_per_sample`.

```python
sampled_rag_ds = rag_ds["train"].sample(size = 10, seed = 0, max_num_distractor_documents_per_sample = 2)
```


## 🔧 Custom Datasets with Processors

RAGO provides built-in loaders for popular datasets (HotpotQA and CRAG) with predefined formats. However, you can load **any custom dataset** by defining a **processor** that converts your data format into RAGO's internal structure. Processors are defined in [Processor Codes](../../../rago/dataset/processor/) and handle the transformation from raw dataset formats (JSON, CSV, HuggingFace datasets) to `EvalSample` objects. To add support for a new dataset, simply create a processor class that inherits from `BaseProcessor` and implements the required conversion logic.

---

## 📚 Related Documentation

- [Dataset Generator](generator.md)
- [Run your first experiment](../optimization/run_experiment.md)

---
