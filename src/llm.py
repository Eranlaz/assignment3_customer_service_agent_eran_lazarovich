from langchain_openai import ChatOpenAI

from src.config import NEBIUS_API_KEY, NEBIUS_MODEL, validate_nebius_config


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI-compatible model client for Nebius Token Factory.

    Returns:
        A LangChain ChatOpenAI model configured to call Nebius Token Factory.
    """
    validate_nebius_config()

    return ChatOpenAI(
        model=NEBIUS_MODEL,
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        temperature=temperature,
        max_retries=2,
    )
