"""LLM and embedding model initialization using Ollama."""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(model: str = "llama3.2", temperature: float = 0) -> ChatOllama:
    """Initialize the Ollama LLM.

    Args:
        model: Model name available in Ollama.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        ChatOllama instance.
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=OLLAMA_BASE_URL,
    )


def get_embeddings(model: str = "nomic-embed-text") -> OllamaEmbeddings:
    """Initialize the Ollama embedding model.

    Args:
        model: Embedding model name available in Ollama.

    Returns:
        OllamaEmbeddings instance.
    """
    return OllamaEmbeddings(
        model=model,
        base_url=OLLAMA_BASE_URL,
    )


def get_guard_llm(model: str = "llama-guard3", temperature: float = 0) -> ChatOllama:
    """Initialize the Llama Guard 3 content safety model.

    Args:
        model: Guard model name available in Ollama.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        ChatOllama instance for content safety classification.
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=OLLAMA_BASE_URL,
    )
