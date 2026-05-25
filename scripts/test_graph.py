from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import build_graph


def print_trace(messages) -> None:
    """
    Print tool calls and observations from the graph execution.
    """
    printed_anything = False

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                printed_anything = True
                print(f"Tool call: {tool_call['name']}({tool_call['args']})")

        if isinstance(message, ToolMessage):
            printed_anything = True
            content = str(message.content)
            if len(content) > 700:
                content = content[:700] + "..."
            print(f"Observation: {content}")

    if not printed_anything:
        print("No tool calls were needed.")


def run_query(query: str) -> None:
    """
    Run one query through the LangGraph agent and print route, trace, and answer.
    """
    graph = build_graph()

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "iterations": 0,
        },
        config={"recursion_limit": 30},
    )

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)
    print()

    print("=" * 80)
    print("ROUTE")
    print("=" * 80)
    print(result.get("route"))
    print(result.get("route_reason"))
    print()

    print("=" * 80)
    print("TRACE")
    print("=" * 80)
    print_trace(result["messages"])
    print()

    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and not message.tool_calls:
            print(message.content)
            break
    print()


def main() -> None:
    test_queries = [
        "What categories exist in the dataset?",
        "How many refund requests did we get?",
        "Show me 5 examples of the SHIPPING category.",
        "Summarize the FEEDBACK category.",
        "Show me examples of people wanting their money back.",
        "What is the distribution of intents in the ACCOUNT category?",
        "What is the total number of ACCOUNT and REFUND records combined?",
        "Compare the intent distributions of ACCOUNT and REFUND.",
        "What's the best CRM software for handling complaints?",
        "Who is the president of France?",
    ]

    for query in test_queries:
        run_query(query)


if __name__ == "__main__":
    main()
