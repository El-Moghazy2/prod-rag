"""RAG in Production - Reusable modules for the workshop."""

from rag.models import get_llm, get_embeddings, get_guard_llm
from rag.data_loader import load_pdf, get_documents
from rag.vectorstore import create_vector_store, get_splitter, get_doc_chunks, create_retriever
from rag.prompts import get_prompt
from rag.guardrails import check_llama_guard
from rag.pipeline import build_basic_graph, build_guarded_graph
from rag.evaluation import (
    create_dataset, create_dataset_from_testset, evaluate_model, compare_results,
    build_knowledge_graph, generate_testset,
)
