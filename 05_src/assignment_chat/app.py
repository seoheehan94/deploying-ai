# app.py

from typing import List, Tuple

import gradio as gr

from guardrails import check_banned_topics, check_prompt_probing
from services import semantic_query, weather_explainer, planner_service
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR.parent / ".secrets")


# Persona description (not exposed directly; used conceptually)
ASSISTANT_NAME = "Study Concierge"
PERSONA_DESCRIPTION = (
    "You are an AI lab teaching assistant with a friendly, concise tone. "
    "You help users understand course materials, explain examples, and "
    "answer questions clearly."
)


def trim_history(history: List[Tuple[str, str]], max_turns: int = 8):
    """
    Keep only the last `max_turns` exchanges to simulate a short-term memory.
    """
    if len(history) <= max_turns:
        return history
    # keep last max_turns
    return history[-max_turns:]


def route_message(message: str) -> str:
    """
    Very simple routing:
    - If message mentions 'weather', call weather_explainer
    - If message mentions 'plan' or 'study schedule', call planner_service
    - Otherwise, use semantic_query
    """
    lower = message.lower()
    if "weather" in lower:
        if "vancouver" in lower:
            return weather_explainer("vancouver")
        elif "montreal" in lower:
            return weather_explainer("montreal")
        else:
            return weather_explainer("toronto")

    if "study plan" in lower or "study schedule" in lower or "plan my study" in lower:
        return planner_service(message)

    # Default: semantic search over course notebooks
    return semantic_query(message)


def handle_user_message(message: str, history: list):
    # Guardrails
    banned_reply = check_banned_topics(message)
    if banned_reply is not None:
        history.append((message, banned_reply))
        history = trim_history(history)
        return history, ""

    probe_reply = check_prompt_probing(message)
    if probe_reply is not None:
        history.append((message, probe_reply))
        history = trim_history(history)
        return history, ""

    # Route to one of the services
    try:
        assistant_reply = route_message(message)
    except Exception as e:
        assistant_reply = (
            "Something went wrong while processing your request. "
            f"Error: {e}"
        )

    history.append((message, assistant_reply))
    history = trim_history(history)
    return history, ""


def build_interface():
    with gr.Blocks(title=ASSISTANT_NAME) as demo:
        gr.Markdown(
            f"## {ASSISTANT_NAME}\n"
            "Your conversational assistant for the course materials."
        )

        chatbot = gr.Chatbot(
            label=ASSISTANT_NAME,
            # default type is "tuples", which expects [ [user, assistant], ... ]
            )
        state = gr.State([])  # list of (user, assistant) tuples

        with gr.Row():
            user_input = gr.Textbox(
                show_label=False,
                placeholder="Ask me about the labs or course materials...",
            )
            send_btn = gr.Button("Send")

        def respond(message, history):
            history = history or []
            return handle_user_message(message, history)

        send_btn.click(
            fn=respond,
            inputs=[user_input, state],
            outputs=[chatbot, user_input],
        )

        user_input.submit(
            fn=respond,
            inputs=[user_input, state],
            outputs=[chatbot, user_input],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
