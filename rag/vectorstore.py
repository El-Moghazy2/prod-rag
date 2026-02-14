"""FAISS vector store creation and chunking utilities."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings


def get_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter with the given configuration.

    Args:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def get_doc_chunks(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into chunks.

    Args:
        documents: Raw documents to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = get_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)


def create_vector_store(
    documents: list[Document],
    embeddings: OllamaEmbeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> FAISS:
    """Create a FAISS vector store from documents.

    Args:
        documents: Raw documents to chunk and index.
        embeddings: Embedding model for vectorization.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        FAISS vector store ready for similarity search.
    """
    chunks = get_doc_chunks(documents, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return FAISS.from_documents(chunks, embeddings)


def create_retriever(
    strategy: str,
    chunks: list[Document],
    vector_store: FAISS,
    k: int = 3,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> BaseRetriever:
    """Create a retriever using the specified strategy.

    Args:
        strategy: One of "vector", "bm25", or "hybrid".
        chunks: Document chunks (needed for BM25).
        vector_store: FAISS vector store (needed for vector/hybrid).
        k: Number of documents to retrieve.
        bm25_weight: Weight for BM25 in hybrid mode.
        vector_weight: Weight for vector search in hybrid mode.

    Returns:
        A LangChain BaseRetriever instance.
    """
    if strategy == "vector":
        return vector_store.as_retriever(search_kwargs={"k": k})
    elif strategy == "bm25":
        return BM25Retriever.from_documents(chunks, k=k)
    elif strategy == "hybrid":
        bm25 = BM25Retriever.from_documents(chunks, k=k)
        faiss_ret = vector_store.as_retriever(search_kwargs={"k": k})
        return EnsembleRetriever(
            retrievers=[bm25, faiss_ret],
            weights=[bm25_weight, vector_weight],
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: vector, bm25, hybrid")
