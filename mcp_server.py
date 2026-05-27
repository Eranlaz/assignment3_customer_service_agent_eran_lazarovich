from typing import Any

from mcp.server.fastmcp import FastMCP

from src.dataset_functions import (
    count_records,
    get_categories,
    get_examples,
    get_intent_distribution,
    get_intents,
    search_records,
)


mcp = FastMCP("Customer Service Dataset Analyst")


@mcp.tool()
def list_categories() -> list[str]:
    """
    Return all high-level categories in the Bitext customer service dataset.
    """
    return get_categories()


@mcp.tool()
def list_intents(category: str | None = None) -> list[str]:
    """
    Return all intents in the dataset.

    If category is provided, return only intents inside that category.
    """
    return get_intents(category=category)


@mcp.tool()
def count_dataset_records(
    category: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    """
    Count dataset records, optionally filtered by category and/or intent.
    """
    return {
        "category": category,
        "intent": intent,
        "count": count_records(category=category, intent=intent),
    }


@mcp.tool()
def show_dataset_examples(
    category: str | None = None,
    intent: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Return example customer-service records from the dataset.

    Supports category, intent, number of examples, and offset for pagination.
    """
    return {
        "category": category,
        "intent": intent,
        "n": n,
        "offset": offset,
        "examples": get_examples(
            category=category,
            intent=intent,
            n=n,
            offset=offset,
        ),
    }


@mcp.tool()
def get_intent_distribution_for_category(category: str) -> dict[str, Any]:
    """
    Return the distribution of intents inside a dataset category.
    """
    return {
        "category": category,
        "distribution": get_intent_distribution(category=category),
    }


@mcp.tool()
def search_dataset_records(query: str, n: int = 5) -> dict[str, Any]:
    """
    Search the dataset by natural-language text.
    """
    return {
        "query": query,
        "n": n,
        "results": search_records(query=query, n=n),
    }


if __name__ == "__main__":
    mcp.run()
