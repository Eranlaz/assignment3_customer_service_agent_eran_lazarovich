import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


DATA_PATH = PROJECT_ROOT / "data" / "bitext_customer_service.csv"

FLAGS_COLUMN = "flags"
CUSTOMER_QUERY_COLUMN = "instruction"
CATEGORY_COLUMN = "category"
INTENT_COLUMN = "intent"
AGENT_RESPONSE_COLUMN = "response"

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL")


def validate_nebius_config() -> None:
    """
    Validate that the required Nebius environment variables are configured.
    """
    if not NEBIUS_API_KEY:
        raise ValueError("Missing NEBIUS_API_KEY. Create a .env file based on .env.example.")

    if not NEBIUS_MODEL:
        raise ValueError("Missing NEBIUS_MODEL. Add a model name to your .env file.")
