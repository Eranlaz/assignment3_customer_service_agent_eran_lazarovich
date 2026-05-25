import argparse
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from src.graph import build_graph


STORAGE_DIR = Path("storage")
CHECKPOINT_DB_PATH = STORAGE_DIR / "checkpoints.sqlite"


def _shorten(text: str, max_chars: int = 1000) -> str:
    """
    Shorten long tool observations for readable CLI output.
    """
    text = str(text)
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def _current_turn_messages(messages, current_user_input: str):
    """
    Return only messages that belong to the current user turn.
    """
    start_index = 0

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) and str(message.content).strip() == current_user_input.strip():
            start_index = index
            break

    return messages[start_index:]


def print_reasoning_trace(messages, current_user_input: str) -> None:
    """
    Print safe reasoning trace for the current turn only.
    This does not print private chain-of-thought.
    """
    printed_anything = False

    for message in _current_turn_messages(messages, current_user_input):
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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent"
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID used to persist and resume conversation memory.",
    )

    return parser.parse_args()


def run_cli() -> None:
    """
    Run the interactive command-line interface.
    """
    args = parse_args()
    session_id = args.session

    STORAGE_DIR.mkdir(exist_ok=True)

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": session_id,
            },
            "recursion_limit": 30,
        }

        print("Customer Service Data Analyst Agent")
        print(f"Session: {session_id}")
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
                    config=config,
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
            print_reasoning_trace(result["messages"], user_input)
            print()

            print("Agent:", get_final_answer(result["messages"]))
            print()


if __name__ == "__main__":
    run_cli()
