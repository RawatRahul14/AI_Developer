# === Python Modules ===
from typing import Dict, List

# === Utility to manage conversation history ===
def update_recent_chats(
        recent_chats: Dict[int, Dict[str, str]],
        latest_question: str,
        answer: str,
        max_chats: int = 3
) -> Dict[int, Dict[str, str]]:
    """
    Updates the conversation history to always contain the last `max_chats` turns.
    Automatically shifts and reindexes so keys remain sequential (1..max_chats).
    """
    # Ensure dictionary is valid
    if not isinstance(recent_chats, Dict):
        recent_chats = {}

    # Append the new chat at the end
    chats = list(recent_chats.values())
    chats.append({
        "question": latest_question.strip(),
        "answer": answer.strip()
    })

    # Keep only the last N
    chats = chats[-max_chats:]

    # Rebuild with proper numeric keys (1..max_chats)
    recent_chats = {i + 1: chat for i, chat in enumerate(chats)}

    return recent_chats