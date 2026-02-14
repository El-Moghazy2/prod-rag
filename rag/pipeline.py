"""LangGraph pipeline for basic and guarded RAG."""

from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from rag.guardrails import check_llama_guard
from rag.prompts import get_prompt


# --- State Definitions ---

class State(TypedDict):
    question: str
    context: str
    answer: str


class GuardedState(TypedDict):
    question: str
    context: str
    answer: str
    is_safe: bool
    guardrail_message: str


# --- Basic RAG Graph ---

def build_basic_graph(
    llm: ChatOllama,
    vector_store: FAISS,
    prompt_template: ChatPromptTemplate | None = None,
    k: int = 3,
    retriever: BaseRetriever | None = None,
) -> StateGraph:
    """Build and compile a basic RAG graph (no guardrails).

    Args:
        llm: The language model.
        vector_store: FAISS vector store for retrieval.
        prompt_template: Prompt template (defaults to restrictive).
        k: Number of documents to retrieve.
        retriever: Optional retriever to use instead of vector_store.similarity_search().

    Returns:
        Compiled LangGraph StateGraph.
    """
    if prompt_template is None:
        prompt_template = get_prompt("restrictive")

    def retrieve(state: State) -> dict:
        if retriever is not None:
            docs = retriever.invoke(state["question"])
        else:
            docs = vector_store.similarity_search(state["question"], k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    def generate(state: State) -> dict:
        messages = prompt_template.format_messages(
            context=state["context"],
            question=state["question"],
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# --- Guarded RAG Graph ---

def build_guarded_graph(
    llm: ChatOllama,
    vector_store: FAISS,
    prompt_template: ChatPromptTemplate | None = None,
    k: int = 3,
    retriever: BaseRetriever | None = None,
    guard_llm: ChatOllama | None = None,
) -> StateGraph:
    """Build and compile a guarded RAG graph with Llama Guard 3 safety checks.

    Uses Llama Guard for both input and output safety screening.

    Args:
        llm: The language model.
        vector_store: FAISS vector store for retrieval.
        prompt_template: Prompt template (defaults to restrictive).
        k: Number of documents to retrieve.
        retriever: Optional retriever to use instead of vector_store.similarity_search().
        guard_llm: Llama Guard 3 model for content safety. Required.

    Returns:
        Compiled LangGraph StateGraph with Llama Guard guardrails.
    """
    if prompt_template is None:
        prompt_template = get_prompt("restrictive")

    def input_guard(state: GuardedState) -> dict:
        question = state["question"]
        result = check_llama_guard(question, guard_llm)
        if not result["is_safe"]:
            category_info = ""
            if result["categories"]:
                category_info = " Categories: " + ", ".join(result["categories"]) + "."
            return {
                "is_safe": False,
                "guardrail_message": f"Your message was flagged as unsafe by content safety screening.{category_info} Please rephrase your question.",
            }
        return {"is_safe": True, "guardrail_message": ""}

    def route_after_guard(state: GuardedState) -> str:
        if state.get("is_safe"):
            return "retrieve"
        return "end_early"

    def end_early(state: GuardedState) -> dict:
        return {"answer": state.get("guardrail_message", "Request blocked by guardrails.")}

    def retrieve(state: GuardedState) -> dict:
        if retriever is not None:
            docs = retriever.invoke(state["question"])
        else:
            docs = vector_store.similarity_search(state["question"], k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    def generate(state: GuardedState) -> dict:
        messages = prompt_template.format_messages(
            context=state["context"],
            question=state["question"],
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    def output_guard(state: GuardedState) -> dict:
        answer = state["answer"]
        result = check_llama_guard(answer, guard_llm)
        if not result["is_safe"]:
            category_info = ""
            if result["categories"]:
                category_info = " Categories: " + ", ".join(result["categories"]) + "."
            return {
                "answer": f"The generated response was flagged as unsafe by content safety screening.{category_info} Please try a different question.",
            }
        return {}

    graph = StateGraph(GuardedState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("end_early", end_early)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("output_guard", output_guard)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", route_after_guard, {"retrieve": "retrieve", "end_early": "end_early"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("end_early", END)

    return graph.compile()
