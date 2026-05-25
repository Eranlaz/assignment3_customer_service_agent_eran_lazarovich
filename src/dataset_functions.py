from typing import Any

import pandas as pd

from src.config import (
    AGENT_RESPONSE_COLUMN,
    CATEGORY_COLUMN,
    CUSTOMER_QUERY_COLUMN,
    INTENT_COLUMN,
)
from src.data_loader import load_dataset


def _normalize_text(value: str) -> str:
    """
    Normalize text for case-insensitive comparisons.
    """
    return value.strip().lower()


def _filter_dataset(
    df: pd.DataFrame,
    category: str | None = None,
    intent: str | None = None,
) -> pd.DataFrame:
    """
    Filter the dataset by optional category and intent.
    Comparisons are case-insensitive.
    """
    filtered = df.copy()

    if category:
        normalized_category = _normalize_text(category)
        filtered = filtered[
            filtered[CATEGORY_COLUMN].astype(str).str.lower() == normalized_category
        ]

    if intent:
        normalized_intent = _normalize_text(intent)
        filtered = filtered[
            filtered[INTENT_COLUMN].astype(str).str.lower() == normalized_intent
        ]

    return filtered


def get_categories() -> list[str]:
    """
    Return all unique dataset categories sorted alphabetically.
    """
    df = load_dataset()
    categories = df[CATEGORY_COLUMN].dropna().astype(str).unique().tolist()
    return sorted(categories)


def get_intents(category: str | None = None) -> list[str]:
    """
    Return all unique intents.

    Args:
        category: Optional category used to restrict the intents.
    """
    df = load_dataset()
    filtered = _filter_dataset(df, category=category)

    intents = filtered[INTENT_COLUMN].dropna().astype(str).unique().tolist()
    return sorted(intents)


def count_records(
    category: str | None = None,
    intent: str | None = None,
) -> int:
    """
    Count records in the dataset, optionally filtered by category and/or intent.
    """
    df = load_dataset()
    filtered = _filter_dataset(df, category=category, intent=intent)
    return int(len(filtered))


def get_examples(
    category: str | None = None,
    intent: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Return example customer-service records.

    Args:
        category: Optional category filter.
        intent: Optional intent filter.
        n: Number of examples to return.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    if n > 20:
        raise ValueError("n must not be greater than 20")

    if offset < 0:
        raise ValueError("offset must not be negative")

    df = load_dataset()
    filtered = _filter_dataset(df, category=category, intent=intent)

    examples = []
    for _, row in filtered.iloc[offset : offset + n].iterrows():
        examples.append(
            {
                "instruction": row[CUSTOMER_QUERY_COLUMN],
                "category": row[CATEGORY_COLUMN],
                "intent": row[INTENT_COLUMN],
                "response": row[AGENT_RESPONSE_COLUMN],
            }
        )

    return examples


def get_intent_distribution(category: str) -> list[dict[str, Any]]:
    """
    Return the distribution of intents within a category.

    Args:
        category: The category to analyze.
    """
    df = load_dataset()
    filtered = _filter_dataset(df, category=category)

    total = len(filtered)
    if total == 0:
        return []

    counts = filtered[INTENT_COLUMN].value_counts()

    distribution = []
    for intent, count in counts.items():
        distribution.append(
            {
                "intent": str(intent),
                "count": int(count),
                "percentage": round((int(count) / total) * 100, 2),
            }
        )

    return distribution


def search_records(
    query: str,
    n: int = 5,
) -> list[dict[str, Any]]:
    """
    Search records using prioritized matching.

    Priority order:
    1. Exact or partial category matches
    2. Exact or partial intent matches
    3. Customer instruction matches
    4. Agent response matches

    This helps avoid cases where searching for 'refund' returns unrelated
    cancellation rows only because the agent response mentions refunds.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if n < 1:
        raise ValueError("n must be at least 1")

    if n > 20:
        raise ValueError("n must not be greater than 20")

    if offset < 0:
        raise ValueError("offset must not be negative")

    df = load_dataset()
    normalized_query = _normalize_text(query)

    category_matches = df[
        df[CATEGORY_COLUMN].astype(str).str.lower().str.contains(normalized_query, regex=False)
    ]

    intent_matches = df[
        df[INTENT_COLUMN].astype(str).str.lower().str.contains(normalized_query, regex=False)
    ]

    instruction_matches = df[
        df[CUSTOMER_QUERY_COLUMN].astype(str).str.lower().str.contains(normalized_query, regex=False)
    ]

    response_matches = df[
        df[AGENT_RESPONSE_COLUMN].astype(str).str.lower().str.contains(normalized_query, regex=False)
    ]

    combined = pd.concat(
        [category_matches, intent_matches, instruction_matches, response_matches]
    ).drop_duplicates()

    examples = []
    for _, row in combined.head(n).iterrows():
        examples.append(
            {
                "instruction": row[CUSTOMER_QUERY_COLUMN],
                "category": row[CATEGORY_COLUMN],
                "intent": row[INTENT_COLUMN],
                "response": row[AGENT_RESPONSE_COLUMN],
            }
        )

    return examples
