# guardrails.py

from typing import Optional

BANNED_KEYWORDS = [
    "cat", "cats",
    "dog", "dogs",
    "zodiac", "horoscope", "horoscopes",
    "taylor swift", "swiftie", "swifties",
]

PROMPT_PROBE_KEYWORDS = [
    "system prompt",
    "initial prompt",
    "ignore previous instructions",
    "developer message",
    "jailbreak",
]


def check_banned_topics(user_message: str) -> Optional[str]:
    """
    If the message contains a banned topic, return a refusal string.
    Otherwise return None.
    """
    lower = user_message.lower()

    for kw in BANNED_KEYWORDS:
        if kw in lower:
            return (
                "I am not allowed to respond to that topic. "
                "Please choose a different subject."
            )

    return None


def check_prompt_probing(user_message: str) -> Optional[str]:
    """
    If the user is trying to access or modify the system prompt,
    return a refusal string. Otherwise return None.
    """
    lower = user_message.lower()

    for kw in PROMPT_PROBE_KEYWORDS:
        if kw in lower:
            return (
                "My internal instructions (system prompt) are private and "
                "cannot be revealed or modified. "
                "Let’s focus on your questions instead."
            )

    return None
