"""RAGAS evaluation utilities for the RAG pipeline."""

import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.documents import Document
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from langchain_ollama import ChatOllama, OllamaEmbeddings


# --- Test Set Generation ---

def build_knowledge_graph(
    documents: list[Document],
    llm: ChatOllama,
    embeddings: OllamaEmbeddings,
) -> KnowledgeGraph:
    """Build a RAGAS KnowledgeGraph from documents.

    Args:
        documents: Source documents to build the graph from.
        llm: LLM for graph transformations.
        embeddings: Embedding model for graph transformations.

    Returns:
        Enriched KnowledgeGraph ready for test set generation.
    """
    kg = KnowledgeGraph()

    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    transforms = default_transforms(
        documents=documents,
        llm=evaluator_llm,
        embedding_model=evaluator_embeddings,
    )
    apply_transforms(kg, transforms)

    return kg


def generate_testset(
    documents: list[Document],
    llm: ChatOllama,
    embeddings: OllamaEmbeddings,
    testset_size: int = 10,
    knowledge_graph: KnowledgeGraph | None = None,
):
    """Generate a test set using RAGAS TestsetGenerator.

    Args:
        documents: Source documents for test generation.
        llm: LLM for test generation.
        embeddings: Embedding model.
        testset_size: Number of test samples to generate.
        knowledge_graph: Pre-built KnowledgeGraph (optional, will be created if None).

    Returns:
        RAGAS Testset object.
    """
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    generator = TestsetGenerator(
        llm=evaluator_llm,
        embedding_model=evaluator_embeddings,
        knowledge_graph=knowledge_graph,
    )

    if knowledge_graph is not None:
        testset = generator.generate(testset_size=testset_size)
    else:
        testset = generator.generate_with_langchain_docs(
            documents, testset_size=testset_size
        )

    return testset


# --- Dataset Creation from Static QA File ---

def create_dataset(
    graph,
    qa_path: str = "qa_dataset.xlsx",
) -> EvaluationDataset:
    """Run the RAG graph on a QA dataset and build a RAGAS EvaluationDataset.

    Args:
        graph: Compiled LangGraph to evaluate.
        qa_path: Path to the Excel file with 'question' and 'ground_truth' columns.

    Returns:
        RAGAS EvaluationDataset ready for evaluation.
    """
    df = pd.read_excel(qa_path)
    samples = []

    for _, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]

        result = graph.invoke({"question": question})

        samples.append({
            "user_input": question,
            "retrieved_contexts": [result.get("context", "")],
            "response": result.get("answer", ""),
            "reference": ground_truth,
        })
        print(f"  Evaluated: {question[:60]}...")

    return EvaluationDataset.from_list(samples)


def create_dataset_from_testset(graph, testset) -> EvaluationDataset:
    """Run the RAG graph on a RAGAS-generated testset and build an EvaluationDataset.

    Args:
        graph: Compiled LangGraph to evaluate.
        testset: RAGAS Testset from generate_testset().

    Returns:
        RAGAS EvaluationDataset ready for evaluation.
    """
    df = testset.to_pandas()
    samples = []

    for _, row in df.iterrows():
        question = row["user_input"]
        reference = row.get("reference", "")

        result = graph.invoke({"question": question})

        samples.append({
            "user_input": question,
            "retrieved_contexts": [result.get("context", "")],
            "response": result.get("answer", ""),
            "reference": reference,
        })
        print(f"  Evaluated: {question[:60]}...")

    return EvaluationDataset.from_list(samples)


# --- Evaluation ---

def evaluate_model(
    graph,
    name: str,
    llm: ChatOllama,
    embeddings: OllamaEmbeddings,
    qa_path: str | None = "qa_dataset.xlsx",
    testset=None,
) -> dict:
    """Evaluate a RAG graph and return named results.

    Args:
        graph: Compiled LangGraph to evaluate.
        name: Name for this configuration (e.g., "basic", "guarded").
        llm: LLM for RAGAS judge.
        embeddings: Embedding model for RAGAS.
        qa_path: Path to the QA dataset (used if testset is None).
        testset: RAGAS Testset (takes priority over qa_path).

    Returns:
        Dict with 'name', 'scores' (dict of metric scores), and 'dataframe'.
    """
    if testset is not None:
        dataset = create_dataset_from_testset(graph, testset)
    else:
        dataset = create_dataset(graph, qa_path)

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return {
        "name": name,
        "scores": {k: v for k, v in result.items() if isinstance(v, (int, float))},
        "dataframe": result.to_pandas(),
    }


def compare_results(results: list[dict]) -> pd.DataFrame:
    """Compare evaluation results from multiple configurations with a bar chart.

    Args:
        results: List of dicts from evaluate_model().

    Returns:
        Summary DataFrame with metrics per configuration.
    """
    rows = []
    for r in results:
        row = {"Configuration": r["name"]}
        row.update(r["scores"])
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Configuration")

    # Plot comparison
    ax = summary.plot(kind="bar", figsize=(10, 5), rot=0, colormap="viridis")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("RAG Configuration Comparison (RAGAS Metrics)")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    plt.tight_layout()
    plt.show()

    return summary
