"""Chainlit chat application for the RAG workshop.

Run with: chainlit run app.py -w
"""

from typing import cast

import chainlit as cl
from langchain_core.runnables import Runnable, RunnableConfig

from rag.models import get_llm, get_embeddings, get_guard_llm
from rag.data_loader import get_documents
from rag.vectorstore import create_vector_store, get_doc_chunks, create_retriever
from rag.pipeline import build_guarded_graph

# --- Module-level initialization (runs once at startup) ---

print("Initializing RAG pipeline...")

llm = get_llm()
embeddings = get_embeddings()
guard_llm = get_guard_llm()

print("Loading documents...")
documents = get_documents()

print("Building vector store...")
chunks = get_doc_chunks(documents, chunk_size=1000, chunk_overlap=200)
vector_store = create_vector_store(documents, embeddings, chunk_size=1000, chunk_overlap=200)

print("Creating hybrid retriever...")
hybrid_retriever = create_retriever("hybrid", chunks, vector_store, k=3)

graph = build_guarded_graph(llm, vector_store, k=3, retriever=hybrid_retriever, guard_llm=guard_llm)

print("RAG pipeline ready!")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("graph", graph)


@cl.on_message
async def on_message(message: cl.Message):
    graph = cast(Runnable, cl.user_session.get("graph"))

    msg = cl.Message(content="")

    result = await graph.ainvoke(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )

    msg.content = result["answer"]
    await msg.send()
