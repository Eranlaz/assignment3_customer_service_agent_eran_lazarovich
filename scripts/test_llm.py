from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.llm import get_chat_model


def main() -> None:
    model = get_chat_model()

    response = model.invoke(
        "Reply with exactly this text and nothing else: LLM connection works"
    )

    print(response.content)


if __name__ == "__main__":
    main()
