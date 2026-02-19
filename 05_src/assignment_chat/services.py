# services.py

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings

from openai_client import get_client
import json
import requests

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "course_materials"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # adjust to your course model

### Service 1: Semantic search over the notebooks  ###
def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)

def embed_query(text: str) -> List[float]:
    client = get_client()
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    return resp.data[0].embedding



def call_llm_with_context(question: str, context: str) -> str:
    client = get_client()

    system_prompt = (
        "You are an AI study concierge and teaching assistant. "
        "Use ONLY the provided context from course notebooks to answer the user's question. "
        "If the context is insufficient, say you are not sure and avoid guessing."
    )

    resp = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Context from course materials:\n"
                    f"{context}\n\n"
                    "User question:\n"
                    f"{question}"
                ),
            },
        ],
    )

    return resp.output[0].content[0].text

def semantic_query(question: str, k: int = 4) -> str:
    """
    High-level function used by the chat app.

    - Embeds the question
    - Retrieves top-k text chunks from Chroma
    - Calls the model with those chunks as context
    - Returns the model's answer
    """
    collection = get_chroma_collection()
    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents"],  # ignore metadatas/ids in the answer
    )

    # results["documents"] is a list of lists: [[doc1, doc2, ...]]
    docs = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(docs)

    if not context.strip():
        return (
            "I could not find relevant information in the course materials "
            "for that question."
        )

    answer = call_llm_with_context(question, context)
    return answer

### Service 2: Weather API   ###
CITY_COORDS = {
    "toronto": (43.65107, -79.347015),
    "vancouver": (49.2827, -123.1207),
    "montreal": (45.5019, -73.5674),
}

def weather_explainer(city: str = "toronto") -> str:
    """
    Simple API-based service using Open-Meteo (no API key).
    Returns a natural-language weather summary.
    """
    city_key = city.lower()
    if city_key not in CITY_COORDS:
        return (
            f"I only know a few cities for weather right now "
            f"(Toronto, Vancouver, Montreal). You asked for '{city}', "
            "which I don't recognize."
        )

    lat, lon = CITY_COORDS[city_key]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m&current_weather=true"
    )

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return f"I tried to call the weather API but something went wrong: {e}"

    data = resp.json()
    current = data.get("current_weather", {})
    temp = current.get("temperature")
    wind = current.get("windspeed")
    code = current.get("weathercode")

    if temp is None:
        return "The weather API did not return current conditions."

    return (
        f"In {city.capitalize()} right now, it's about {temp}°C with a wind speed "
        f"around {wind} km/h. Weather code {code} suggests general conditions like "
        f"clear, cloudy, or precipitation, depending on the mapping."
    )

### Service 3: function-calling planner  ###
TOOLS = [
    {
        "type": "function",
        "name": "plan_study_session",  
        "description": "Create a structured study plan for a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["topic", "duration_minutes"],
        },
    }
]


def plan_study_session(topic: str, duration_minutes: int):
    blocks = max(1, duration_minutes // 20)
    per_block = duration_minutes / blocks
    return {
        "topic": topic,
        "duration": duration_minutes,
        "blocks": [
            {
                "block": i + 1,
                "minutes": per_block,
                "focus": f"Subtopic {i + 1} of {topic}",
            }
            for i in range(blocks)
        ],
    }

def extract_topic(user_message: str) -> str:
    text = user_message.lower()
    # simple heuristic: look for "review the ..." or "study ..."
    if "review the" in text:
        # e.g. "review the longer context lab"
        start = text.index("review the") + len("review the")
        topic = text[start:].strip(" ?.")
        return topic or "the lab materials"
    if "review" in text:
        start = text.index("review") + len("review")
        topic = text[start:].strip(" ?.")
        return topic or "the lab materials"
    if "study" in text:
        start = text.index("study") + len("study")
        topic = text[start:].strip(" ?.")
        return topic or "the lab materials"
    return "the lab materials"


def planner_service(user_message: str) -> str:
    """
    Service 3: Simple Python-based study planner.

    For now, parse only duration (e.g., '60-minute') and use the whole
    message as topic description.
    """
    import re

    match = re.search(r"(\d+)\s*-?\s*minute", user_message.lower())
    if match:
        duration = int(match.group(1))
    else:
        duration = 60  # default

    topic = extract_topic(user_message)
    plan = plan_study_session(topic=topic, duration_minutes=duration)

    lines = [
        f"Here's a {plan['duration']} minute study plan:",
    ]
    for block in plan["blocks"]:
        lines.append(
            f"- Block {block['block']} ({int(block['minutes'])} min): "
            f"{block['focus']}"
        )
    return "\n".join(lines)
