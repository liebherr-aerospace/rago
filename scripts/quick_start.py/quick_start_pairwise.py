from rago.dataset import RAGDataset
from rago.eval import BaseEvaluator
from rago.optimization.manager import OptimParams, SimplePairWiseOptunaManager
from rago.optimization.search_space.param_space import CategoricalParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

# Configure optimization
params = OptimParams(
    experiment_name="bm25_optim",
    n_startup_trials=1,
    n_iter=10,
)

# Load dataset and evaluator
full_ds = RAGDataset.load_dataset("crag").sample(5, 0, 10)
evaluator = BaseEvaluator.load("simple_llm_judge")

# Split dataset into train/test
datasets = full_ds.split_dataset([0.8], split_names=["train", "test"], seed=42)

# Load RAG configuration space
config_space = RAGConfigSpace(
    RetrieverConfigSpace(retriever_type_name=CategoricalParamSpace(
                choices=["BM25Retriever", "HybridRetriever"],
            )),
)

optimizer = SimplePairWiseOptunaManager(
        params=params,
        datasets=datasets,
        optim_evaluator=evaluator,
        optim_metric_name="correctness",
        test_evaluators=[evaluator],
        config_space=config_space,
        num_initial_rags=2,
    )
optimizer.optimize()
