import json
import re
from pathlib import Path
from typing import Any


PROFILES_DIR = Path("profiles")


def _safe_user_id(user_id: str) -> str:
    """
    Convert a user id into a safe filename.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id.strip())
    return cleaned or "default"


def _profile_path(user_id: str) -> Path:
    """
    Return the profile JSON path for a user.
    """
    PROFILES_DIR.mkdir(exist_ok=True)
    return PROFILES_DIR / f"{_safe_user_id(user_id)}.json"


def _empty_profile(user_id: str) -> dict[str, Any]:
    """
    Create a new empty user profile.
    """
    return {
        "user_id": user_id,
        "name": None,
        "topics": {},
        "preferences": [],
        "facts": [],
    }


def load_profile(user_id: str) -> dict[str, Any]:
    """
    Load a user's persistent profile from disk.
    """
    path = _profile_path(user_id)

    if not path.exists():
        return _empty_profile(user_id)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_profile(user_id: str, profile: dict[str, Any]) -> None:
    """
    Save a user's persistent profile to disk.
    """
    path = _profile_path(user_id)

    with path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, ensure_ascii=False)


def _add_unique(values: list[str], value: str) -> bool:
    """
    Add a value to a list if it does not already exist.
    """
    normalized_existing = {item.lower() for item in values}

    if value.lower() in normalized_existing:
        return False

    values.append(value)
    return True


def _increment_topic(profile: dict[str, Any], topic: str) -> bool:
    """
    Increment a topic counter in the user's profile.
    """
    topics = profile.setdefault("topics", {})
    topics[topic] = int(topics.get(topic, 0)) + 1
    return True


def _extract_name(message: str) -> str | None:
    """
    Extract a name from simple self-introduction patterns.
    """
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{1,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z .'-]{1,40})",
        r"\bi am named\s+([A-Za-z][A-Za-z .'-]{1,40})",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)

        if match:
            name = match.group(1).strip()
            name = re.split(r"\band\b|\bi\b|\bmy\b", name, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            return name

    return None


def update_profile_from_message(user_id: str, message: str) -> bool:
    """
    Update a user's profile based on a single message.

    The profile stores distilled facts rather than a transcript.
    """
    profile = load_profile(user_id)
    changed = False
    normalized_message = message.lower()

    extracted_name = _extract_name(message)
    if extracted_name and profile.get("name") != extracted_name:
        profile["name"] = extracted_name
        changed = True

    topic_keywords = {
        "refund analysis": ["refund", "refunds", "money back", "reimbursement"],
        "complaints": ["complaint", "complaints"],
        "shipping": ["shipping", "delivery address"],
        "account": ["account", "password", "registration"],
        "orders": ["order", "orders", "cancellation", "cancel"],
        "feedback": ["feedback", "review"],
        "intent distributions": ["intent distribution", "distribution of intents", "breakdown"],
        "examples": ["examples", "show me", "more examples"],
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in normalized_message for keyword in keywords):
            changed = _increment_topic(profile, topic) or changed

    preferences = profile.setdefault("preferences", [])

    preference_patterns = [
        ("step-by-step explanations", ["step by step", "step-by-step", "שלב אחר שלב"]),
        ("detailed explanations", ["detailed", "in depth", "לעומק", "מפורט"]),
        ("concise answers", ["concise", "short answers", "בקצרה"]),
        ("Hebrew explanations", ["hebrew", "עברית"]),
        ("English explanations", ["english", "אנגלית"]),
    ]

    for preference, keywords in preference_patterns:
        if any(keyword in normalized_message for keyword in keywords):
            changed = _add_unique(preferences, preference) or changed

    facts = profile.setdefault("facts", [])

    if "i care" in normalized_message or "i am interested" in normalized_message or "i'm interested" in normalized_message:
        changed = _add_unique(facts, message.strip()) or changed

    if changed:
        save_profile(user_id, profile)

    return changed


def is_profile_question(message: str) -> bool:
    """
    Return True if the user asks what the agent remembers about them.
    """
    normalized_message = message.lower().strip()

    phrases = [
        "what do you remember about me",
        "what do you know about me",
        "what have you learned about me",
        "do you remember me",
        "מה אתה זוכר עליי",
        "מה אתה יודע עליי",
    ]

    return any(phrase in normalized_message for phrase in phrases)


def is_profile_statement(message: str) -> bool:
    """
    Return True if the message looks like a user-profile statement.
    """
    normalized_message = message.lower().strip()

    phrases = [
        "my name is",
        "call me",
        "i am named",
        "i care",
        "i am interested",
        "i'm interested",
        "i prefer",
        "i like",
        "אני מעדיף",
        "קוראים לי",
    ]

    return any(phrase in normalized_message for phrase in phrases)


def format_profile_summary(user_id: str) -> str:
    """
    Format a user's profile as a readable answer.
    """
    profile = load_profile(user_id)

    name = profile.get("name")
    topics = profile.get("topics", {})
    preferences = profile.get("preferences", [])
    facts = profile.get("facts", [])

    has_profile_data = bool(name or topics or preferences or facts)

    if not has_profile_data:
        return (
            "I do not have any saved profile facts about you yet. "
            "You can tell me your name, interests, or preferences."
        )

    lines = ["Here is what I remember about you:"]

    if name:
        lines.append(f"- Your name is {name}.")

    if topics:
        sorted_topics = sorted(
            topics.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        topic_text = ", ".join(f"{topic} ({count})" for topic, count in sorted_topics)
        lines.append(f"- Topics you have shown interest in: {topic_text}.")

    if preferences:
        lines.append("- Preferences: " + ", ".join(preferences) + ".")

    if facts:
        lines.append("- Other profile notes:")
        for fact in facts[-5:]:
            lines.append(f"  - {fact}")

    return "\n".join(lines)
