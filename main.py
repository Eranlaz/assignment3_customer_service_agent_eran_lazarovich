from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from src.graph import build_graph


def _shorten(text: str, max_chars: int = 1000) -> str:
    """
    Shorten long tool observations for readable CLI output.
    """
    text = str(text)
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def print_reasoning_trace(messages) -> None:
    """
    Print safe reasoning trace: tool calls and observations.
    This does not print private chain-of-thought.
    """
    printed_anything = False

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                printed_anything = True
                print(f"Tool call: {tool_call['name']}({tool_call['args']})")

        elif isinstance(message, ToolMessage):
            printed_anything = True
            print(f"Observation: {_shorten(message.content)}")

    if not printed_anything:
        print("No tool calls were needed.")


def get_final_answer(messages) -> str:
    """
    Return the final AI answer from the message list.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)

    return "No final answer was produced."


def run_cli() -> None:
    """
    Run the interactive command-line interface.
    """
    graph = build_graph()

    print("Customer Service Data Analyst Agent")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower().startswith("you:"):
            user_input = user_input[4:].strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_input:
            continue

        try:
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "iterations": 0,
                },
                config={"recursion_limit": 30},
            )

        except GraphRecursionError:
            print()
            print("Route: unknown")
            print("Reasoning trace:")
            print("The graph reached its recursion limit.")
            print()
            print(
                "Agent: I could not complete the analysis within the allowed number of steps. "
                "Please ask a more specific dataset question."
            )
            print()
            continue

        print()
        print(f"Route: {result.get('route')}")
        print(f"Route reason: {result.get('route_reason')}")
        print()

        print("Reasoning trace:")
        print_reasoning_trace(result["messages"])
        print()

        print("Agent:", get_final_answer(result["messages"]))
        print()


if __name__ == "__main__":
    run_cli()
