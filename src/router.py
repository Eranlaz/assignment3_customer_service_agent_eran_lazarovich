from enum import Enum

from pydantic import BaseModel, Field

from src.dataset_functions import get_categories, get_intents


class QueryType(str, Enum):
    """
    Supported query types for the customer service data analyst agent.
    """

    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    OUT_OF_SCOPE = "out_of_scope"


class RouteDecision(BaseModel):
    """
    Router output describing how the agent should handle a user query.
    """

    query_type: QueryType = Field(
        description="The selected query type: structured, unstructured, or out_of_scope."
    )
    reason: str = Field(
        description="Short explanation of why this route was selected."
    )


STRUCTURED_KEYWORDS = {
    "how many",
    "count",
    "number of",
    "categories",
    "category",
    "intents",
    "intent",
    "distribution",
    "breakdown",
    "examples",
    "example",
    "show me",
    "list",
    "what categories",
    "what intents",
}

UNSTRUCTURED_KEYWORDS = {
    "summarize",
    "summary",
    "typical",
    "typically",
    "how do agents respond",
    "how do representatives respond",
    "how do customer service representatives respond",
    "common themes",
    "patterns",
    "describe",
    "analyze",
    "respond to",
    "responses to",
}

OUT_OF_SCOPE_KEYWORDS = {
    "president",
    "champions league",
    "football",
    "soccer",
    "weather",
    "stock price",
    "recipe",
    "poem",
    "story",
    "joke",
    "crm software",
    "best crm",
    "software for",
    "recommend software",
    "movie",
    "song",
}


def _normalize_query(query: str) -> str:
    """
    Normalize a user query for simple keyword matching.
    """
    return query.strip().lower()


def _contains_any(text: str, keywords: set[str]) -> bool:
    """
    Return True if any keyword appears in the text.
    """
    return any(keyword in text for keyword in keywords)


def _dataset_terms() -> set[str]:
    """
    Build a set of domain terms from dataset categories, intents, and common synonyms.
    """
    categories = {category.lower() for category in get_categories()}

    intents = set()
    for intent in get_intents():
        normalized_intent = intent.lower()
        intents.add(normalized_intent)
        intents.add(normalized_intent.replace("_", " "))

    synonyms = {
        "dataset",
        "data",
        "customer",
        "customers",
        "customer service",
        "agent",
        "agents",
        "representative",
        "representatives",
        "request",
        "requests",
        "query",
        "queries",
        "refund",
        "refunds",
        "money back",
        "reimbursement",
        "shipping",
        "delivery",
        "order",
        "orders",
        "account",
        "accounts",
        "feedback",
        "complaint",
        "complaints",
        "cancel",
        "cancellation",
        "invoice",
        "payment",
        "subscription",
        "contact",
    }

    return categories | intents | synonyms


def route_query(query: str) -> RouteDecision:
    """
    Classify a user query as structured, unstructured, or out_of_scope.

    Structured queries ask for concrete data operations such as counts, categories,
    examples, or distributions.

    Unstructured queries ask for summaries or qualitative analysis based on the dataset.

    Out-of-scope queries are unrelated to the Bitext customer service dataset and
    should not be answered from general model knowledge.
    """
    normalized_query = _normalize_query(query)

    if not normalized_query:
        return RouteDecision(
            query_type=QueryType.OUT_OF_SCOPE,
            reason="The query is empty.",
        )

    has_structured_signal = _contains_any(normalized_query, STRUCTURED_KEYWORDS)
    has_unstructured_signal = _contains_any(normalized_query, UNSTRUCTURED_KEYWORDS)
    has_out_of_scope_signal = _contains_any(normalized_query, OUT_OF_SCOPE_KEYWORDS)
    has_dataset_signal = _contains_any(normalized_query, _dataset_terms())

    # Strong external/creative/general-knowledge signals should be declined.
    # Example: "Who is the president of France?" or "Write me a poem".
    if has_out_of_scope_signal:
        return RouteDecision(
            query_type=QueryType.OUT_OF_SCOPE,
            reason="The query appears to ask for general knowledge, creative writing, or external recommendations rather than analysis of the dataset.",
        )

    # Summary and qualitative-response questions over dataset concepts.
    if has_unstructured_signal and has_dataset_signal:
        return RouteDecision(
            query_type=QueryType.UNSTRUCTURED,
            reason="The query asks for a summary or qualitative analysis based on customer service data.",
        )

    # Counts, lists, examples, categories, intents, and distributions.
    if has_structured_signal and has_dataset_signal:
        return RouteDecision(
            query_type=QueryType.STRUCTURED,
            reason="The query asks for a concrete data operation such as a count, list, example, or distribution.",
        )

    # Some valid dataset questions may be short, for example: "refund requests".
    if has_dataset_signal and not has_unstructured_signal:
        return RouteDecision(
            query_type=QueryType.STRUCTURED,
            reason="The query mentions dataset-related concepts and can likely be answered with structured tools.",
        )

    return RouteDecision(
        query_type=QueryType.OUT_OF_SCOPE,
        reason="The query does not appear to be about the Bitext customer service dataset.",
    )
