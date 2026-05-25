from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.dataset_functions import (
    count_records,
    get_categories,
    get_examples,
    get_intent_distribution,
    get_intents,
    search_records,
)


def _clean_optional_text(value: Any) -> str | None:
    """
    Convert optional tool inputs into clean strings.

    LLMs sometimes pass the literal strings "null", "None", or "" instead of
    omitting optional arguments. This helper converts those values to None.
    """
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in {"", "none", "null", "nil", "n/a", "na"}:
        return None

    return text


def _clean_n(value: Any, default: int = 3) -> int:
    """
    Convert n into a safe integer between 1 and 20.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default

    return max(1, min(n, 20))


class ListCategoriesInput(BaseModel):
    """
    Input schema for listing dataset categories.
    """


class ListIntentsInput(BaseModel):
    """
    Input schema for listing intents.
    """

    category: str | None = Field(
        default=None,
        description=(
            "Optional dataset category. If provided, only intents inside this "
            "category are returned. Omit this field when not needed; do not pass "
            "the literal string 'null'."
        ),
    )


class CountRecordsInput(BaseModel):
    """
    Input schema for counting records.
    """

    category: str | None = Field(
        default=None,
        description=(
            "Optional category filter, such as ACCOUNT, ORDER, REFUND, SHIPPING, "
            "or FEEDBACK. Omit when not needed; do not pass the literal string 'null'."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Optional intent filter, such as cancel_order, track_order, get_refund, "
            "or check_refund_policy. Omit when not needed; do not pass the literal "
            "string 'null'."
        ),
    )


class ShowExamplesInput(BaseModel):
    """
    Input schema for showing dataset examples.
    """

    category: str | None = Field(
        default=None,
        description=(
            "Optional category filter, such as ACCOUNT, ORDER, REFUND, SHIPPING, "
            "or FEEDBACK. Omit when not needed; do not pass the literal string 'null'."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Optional intent filter, such as cancel_order, track_order, get_refund, "
            "or check_refund_policy. Omit when not needed; do not pass the literal "
            "string 'null'."
        ),
    )
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of examples to return. Must be between 1 and 20.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of matching examples to skip before returning results. Use this for follow-up requests like 'show me 3 more'.",
    )


class IntentDistributionInput(BaseModel):
    """
    Input schema for intent distribution analysis.
    """

    category: str = Field(
        description="Dataset category to analyze, such as ACCOUNT, ORDER, REFUND, SHIPPING, or FEEDBACK.",
    )


class SearchRecordsInput(BaseModel):
    """
    Input schema for searching records.
    """

    query: str = Field(
        description="Search text, such as refund, money back, cancellation, complaint, account, or shipping.",
    )
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of search results to return. Must be between 1 and 20.",
    )


def list_categories_tool() -> dict[str, Any]:
    """
    Return all categories in the customer service dataset.
    """
    return {"categories": get_categories()}


def list_intents_tool(category: str | None = None) -> dict[str, Any]:
    """
    Return intents in the dataset, optionally restricted to one category.
    """
    clean_category = _clean_optional_text(category)

    return {
        "category": clean_category,
        "intents": get_intents(category=clean_category),
    }


def count_records_tool(
    category: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    """
    Count records filtered by category and/or intent.
    """
    clean_category = _clean_optional_text(category)
    clean_intent = _clean_optional_text(intent)

    return {
        "category": clean_category,
        "intent": clean_intent,
        "count": count_records(category=clean_category, intent=clean_intent),
    }


def show_examples_tool(
    category: str | None = None,
    intent: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Return examples from the dataset filtered by category and/or intent.
    """
    clean_category = _clean_optional_text(category)
    clean_intent = _clean_optional_text(intent)
    clean_n = _clean_n(n)

    try:
        clean_offset = int(offset)
    except (TypeError, ValueError):
        clean_offset = 0

    clean_offset = max(0, clean_offset)

    return {
        "category": clean_category,
        "intent": clean_intent,
        "n": clean_n,
        "offset": clean_offset,
        "examples": get_examples(
            category=clean_category,
            intent=clean_intent,
            n=clean_n,
            offset=clean_offset,
        ),
    }


def intent_distribution_tool(category: str) -> dict[str, Any]:
    """
    Return the distribution of intents inside a category.
    """
    clean_category = _clean_optional_text(category)

    if not clean_category:
        return {
            "category": None,
            "distribution": [],
            "error": "category is required for intent_distribution.",
        }

    return {
        "category": clean_category,
        "distribution": get_intent_distribution(category=clean_category),
    }


def search_records_tool(query: str, n: int = 5) -> dict[str, Any]:
    """
    Search records by text across category, intent, customer instruction, and agent response.
    """
    clean_query = str(query).strip()
    clean_n = _clean_n(n, default=5)

    return {
        "query": clean_query,
        "n": clean_n,
        "results": search_records(query=clean_query, n=clean_n),
    }


DATASET_TOOLS = [
    StructuredTool.from_function(
        func=list_categories_tool,
        name="list_categories",
        description=(
            "Use this tool when the user asks what categories exist in the "
            "Bitext customer service dataset. It returns the unique category names."
        ),
        args_schema=ListCategoriesInput,
    ),
    StructuredTool.from_function(
        func=list_intents_tool,
        name="list_intents",
        description=(
            "Use this tool when the user asks what intents exist in the dataset, "
            "or what intents exist inside a specific category."
        ),
        args_schema=ListIntentsInput,
    ),
    StructuredTool.from_function(
        func=count_records_tool,
        name="count_records",
        description=(
            "Use this tool when the user asks how many records, requests, queries, "
            "or examples exist for a category and/or intent. Provide category, intent, "
            "or both if known. Omit unknown optional fields instead of passing 'null'."
        ),
        args_schema=CountRecordsInput,
    ),
    StructuredTool.from_function(
        func=show_examples_tool,
        name="show_examples",
        description=(
            "Use this tool when the user asks to see example customer queries or "
            "agent responses from a specific category or intent. Provide category "
            "or intent and the number of examples. Omit unknown optional fields "
            "instead of passing 'null'."
        ),
        args_schema=ShowExamplesInput,
    ),
    StructuredTool.from_function(
        func=intent_distribution_tool,
        name="intent_distribution",
        description=(
            "Use this tool when the user asks for the distribution, breakdown, or "
            "counts of intents inside a specific category."
        ),
        args_schema=IntentDistributionInput,
    ),
    StructuredTool.from_function(
        func=search_records_tool,
        name="search_records",
        description=(
            "Use this tool when the user describes a concept in natural language "
            "instead of giving an exact category or intent. For example, use it for "
            "'people wanting their money back', 'complaints', or 'cancellation'."
        ),
        args_schema=SearchRecordsInput,
    ),
]
