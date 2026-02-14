"""Content safety guardrails using Llama Guard 3."""

from langchain_ollama import ChatOllama


CATEGORY_MAP = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
}


def check_llama_guard(text: str, guard_llm: ChatOllama) -> dict:
    """Check text safety using Llama Guard 3.

    Calls the guard LLM and parses the safe/unsafe verdict plus
    any violated category codes.

    Args:
        text: The text to classify (user input or LLM output).
        guard_llm: Llama Guard 3 model instance.

    Returns:
        Dict with 'is_safe' (bool) and 'categories' (list[str]).
    """
    response = guard_llm.invoke(text)
    result = response.content.strip().lower()

    if result.startswith("safe"):
        return {"is_safe": True, "categories": []}

    categories = []
    for code, name in CATEGORY_MAP.items():
        if code.lower() in result:
            categories.append(f"{code}: {name}")

    return {"is_safe": False, "categories": categories}
