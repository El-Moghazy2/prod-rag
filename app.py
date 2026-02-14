"""Chainlit chat application for the RAG workshop.

Run with: chainlit run app.py -w
"""

import chainlit as cl

from rag.models import get_llm, get_embeddings, get_guard_llm
from rag.data_loader import get_documents
from rag.vectorstore import create_vector_store, get_doc_chunks, create_retriever
from rag.pipeline import build_guarded_graph

# --- Module-level initialization (runs once at startup) ---

print("Initializing RAG pipeline...")

# 1. Initialize models
llm = get_llm()
embeddings = get_embeddings()
guard_llm = get_guard_llm()

# 2. Load documents
print("Loading documents...")
documents = get_documents()

# 3. Create vector store and chunks
print("Building vector store...")
chunks = get_doc_chunks(documents, chunk_size=1000, chunk_overlap=200)
vector_store = create_vector_store(documents, embeddings, chunk_size=1000, chunk_overlap=200)

# 4. Create hybrid retriever (BM25 + vector)
print("Creating hybrid retriever...")
hybrid_retriever = create_retriever("hybrid", chunks, vector_store, k=3)

# 5. Build the guarded graph with hybrid retriever
graph = build_guarded_graph(llm, vector_store, k=3, retriever=hybrid_retriever, guard_llm=guard_llm)

print("RAG pipeline ready!")


# --- Chainlit Handlers ---

@cl.on_chat_start
async def on_chat_start():
    """Welcome the user and initialize session state."""
    cl.user_session.set("history", [])

    await cl.Message(
        content=(
            "Welcome to the **RAG in Production** workshop chatbot!\n\n"
            "I can answer questions about **Covestro Safety Data Sheets (SDS)** "
            "covering chemical safety, hazard identification, and handling procedures.\n\n"
            "Try asking:\n"
            "- *What PPE is required for DESMOPHEN XP 2680?*\n"
            "- *What are the hazardous decomposition products of BAYBLEND M750?*\n"
            "- *What first aid measures apply for eye contact with BAYBOND PU 407?*\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process a user message through the guarded RAG graph."""
    query = message.content

    # Show a thinking indicator
    msg = cl.Message(content="")
    await msg.send()

    # Run the guarded graph
    result = graph.invoke({"question": query})
    answer = result["answer"]

    # Retrieve source docs for citation (only if the graph produced a substantive answer)
    guardrail_prefixes = (
        "Input too short", "Input too long", "Your query was flagged",
        "Your message was flagged", "Your question doesn't appear",
        "I'm not confident",
    )
    if not answer.startswith(guardrail_prefixes):
        source_docs = hybrid_retriever.invoke(query)
        sources = set()
        for doc in source_docs:
            product = doc.metadata.get("product_name", "?")
            sources.add(product)

        if sources:
            source_text = "\n".join(f"- {s}" for s in sorted(sources))
            answer += f"\n\n---\n**Sources:**\n{source_text}"

    # Update message with the answer
    msg.content = answer
    await msg.update()

    # Store in session history
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)
