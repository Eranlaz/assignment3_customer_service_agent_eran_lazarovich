from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.dataset_functions import (
    count_records,
    get_categories,
    get_examples,
    get_intent_distribution,
    get_intents,
    search_records,
)


def main() -> None:
    print("=" * 80)
    print("CATEGORIES")
    print("=" * 80)
    categories = get_categories()
    print(categories)
    print()

    print("=" * 80)
    print("FIRST CATEGORY INTENTS")
    print("=" * 80)
    first_category = categories[0]
    print("Category:", first_category)
    print(get_intents(category=first_category)[:10])
    print()

    print("=" * 80)
    print("TOTAL RECORD COUNT")
    print("=" * 80)
    print(count_records())
    print()

    print("=" * 80)
    print("COUNT FOR ORDER CATEGORY")
    print("=" * 80)
    print(count_records(category="ORDER"))
    print()

    print("=" * 80)
    print("EXAMPLES FROM ORDER CATEGORY")
    print("=" * 80)
    examples = get_examples(category="ORDER", n=2)
    for index, example in enumerate(examples, start=1):
        print(f"Example {index}")
        print("Instruction:", example["instruction"])
        print("Category:", example["category"])
        print("Intent:", example["intent"])
        print("Response:", example["response"][:300], "...")
        print()

    print("=" * 80)
    print("INTENT DISTRIBUTION FOR ORDER")
    print("=" * 80)
    distribution = get_intent_distribution(category="ORDER")
    for item in distribution[:10]:
        print(item)
    print()

    print("=" * 80)
    print("SEARCH: refund")
    print("=" * 80)
    search_results = search_records("refund", n=3)
    for index, result in enumerate(search_results, start=1):
        print(f"Result {index}")
        print("Instruction:", result["instruction"])
        print("Category:", result["category"])
        print("Intent:", result["intent"])
        print()


if __name__ == "__main__":
    main()
