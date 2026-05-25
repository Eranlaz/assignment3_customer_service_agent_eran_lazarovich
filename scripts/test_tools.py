from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.tools import DATASET_TOOLS


def get_tool(name: str):
    for tool in DATASET_TOOLS:
        if tool.name == name:
            return tool

    raise ValueError(f"Tool not found: {name}")


def main() -> None:
    print("=" * 80)
    print("AVAILABLE TOOLS")
    print("=" * 80)
    for tool in DATASET_TOOLS:
        print("-", tool.name)
        print(" ", tool.description)
        print()

    print("=" * 80)
    print("TEST: list_categories")
    print("=" * 80)
    print(get_tool("list_categories").invoke({}))
    print()

    print("=" * 80)
    print("TEST: count_records category=REFUND")
    print("=" * 80)
    print(get_tool("count_records").invoke({"category": "REFUND"}))
    print()

    print("=" * 80)
    print("TEST: show_examples category=SHIPPING n=2")
    print("=" * 80)
    print(get_tool("show_examples").invoke({"category": "SHIPPING", "n": 2}))
    print()

    print("=" * 80)
    print("TEST: intent_distribution category=ACCOUNT")
    print("=" * 80)
    print(get_tool("intent_distribution").invoke({"category": "ACCOUNT"}))
    print()

    print("=" * 80)
    print("TEST: search_records query=money back")
    print("=" * 80)
    print(get_tool("search_records").invoke({"query": "money back", "n": 3}))
    print()


if __name__ == "__main__":
    main()
