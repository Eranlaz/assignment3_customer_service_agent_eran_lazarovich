from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.router import route_query


TEST_QUERIES = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 5 examples of the SHIPPING category.",
    "Summarize how agents respond to complaint intents.",
    "Show me examples of people wanting their money back.",
    "What is the distribution of intents in the ACCOUNT category?",
    "What's the best CRM software for handling complaints?",
    "Who is the president of France?",
    "Write me a poem about customer service.",
    "Summarize the FEEDBACK category.",
    "How do customer service representatives typically respond to cancellation requests?",
]


def main() -> None:
    for query in TEST_QUERIES:
        decision = route_query(query)

        print("=" * 80)
        print("Query:", query)
        print("Route:", decision.query_type.value)
        print("Reason:", decision.reason)


if __name__ == "__main__":
    main()
