# Assignment 2: AI Study Concierge Chat
The goal of this assignment is to design and implement an AI system with a conversational interface.

This implementation focuses on a simplified but working version of the required services, then layers on routing, guardrails, and memory. The chat client behaves as an AI study concierge (a lab TA) that can answer questions about the course notebooks, give simple weather explanations, and help plan study sessions. All code is in ./05_src/assignment_chat.


## Services
This implementation is based on plain Python modules and Gradio, together with OpenAI’s Responses and Embeddings APIs (via the course API Gateway) and ChromaDB for semantic search.

The main chat logic is in app.py. Backend services are in services.py. Guardrails are in guardrails.py. OpenAI client configuration is centralized in openai_client.py. The semantic index is built with rag_build_index.py.

### Service 2: Semantic Query
The semantic query service is implemented in services.py as semantic_query(question: str).

It uses a persistent ChromaDB collection built from three course notebooks:
- 01_1_introduction.ipynb
- 01_2_longer_context.ipynb
- 01_3_local_model.ipynb

The index construction is handled by rag_build_index.py:
- Loads each .ipynb as JSON from 01_materials/labs/.
- Extracts markdown cells only (the explanatory text).
- Concatenates consecutive markdown cells into chunks of approximately 800–1000 characters.
- Uses the course’s OpenAI API Gateway client (openai_client.get_client()) to create embeddings for each chunk via the Embeddings API.
- Stores chunks in a Chroma persistent collection named course_materials under ./05_src/assignment_chat/chroma_db using chromadb.PersistentClient.

At query time, semantic_query:
- Embeds the user’s question using the same embeddings model and client.
- Queries the Chroma collection for the top‑k relevant chunks (include=["documents"]).
- Concatenates the retrieved chunk texts into a context string.
- Calls the Responses API with a system prompt that tells the model to answer only using the provided context and to admit when the context is insufficient.
- Returns the model’s text answer, which is then shown in the chat.

### Service 2: API Calls
The API-based service is a weather explainer implemented in services.py.

It calls the public Open-Meteo weather API (no API key) to fetch current weather for a small set of supported cities (e.g., Toronto, Vancouver, Montreal) using latitude/longitude coordinates.
​
The function weather_explainer(city: str):
- Builds the Open-Meteo URL with current_weather=true.
- Parses the JSON response to extract temperature, wind speed, and weather code.
- Returns a short, natural-language summary of the conditions rather than raw JSON.

This function is invoked by the router in app.py (in route_message) whenever the user’s message mentions “weather” (and optionally a known city name).

### Service 3: Your Choice (Study Planner)
The third service is a study planner implemented in services.py.

It consists of two functions:
- plan_study_session(topic: str, duration_minutes: int):
    - Splits the total duration into roughly 20‑minute blocks.
    - Returns a simple structured plan (list of blocks with duration and focus).
- planner_service(user_message: str):
    - Uses a regular expression to detect a duration (e.g., “60-minute”, “45 minute”) in the user’s message.
    - Extracts a rough topic phrase from the message using simple string heuristics (for example “review the longer context lab” → topic “longer context lab”).
    - Calls plan_study_session and converts the plan into a readable bullet-point study schedule.

This service is triggered by the router in app.py whenever the message contains phrases like “study plan”, “study schedule”, or “plan my study”.

## User Interface
- The user interface is implemented in Gradio in app.py.
- The assistant has a distinct personality: It is described in code as a friendly AI lab TA (“Study Concierge”) that aims to answer clearly and concisely.
- Memory handling: After each exchange, history is trimmed to the last N turns (e.g., 8) via trim_history.

## Guardrails and Other Limitations
Guardrails are defined in guardrails.py and applied in app.py before any service or model call.

- System prompt protection
    - Messages are checked for “prompt probing” phrases (e.g., “system prompt”, “ignore previous instructions”, “developer message”, “jailbreak”).
    - If detected, the assistant replies with a fixed refusal and does not call any backend service.

- Restricted topics
    - The assistant must not answer about: cats or dogs, horoscopes or zodiac signs, or Taylor Swift.
    - check_banned_topics scans for these keywords; if found, it returns a short refusal and the request is handled locally.

- Other limitations
    - Semantic search only covers three notebooks (01_1_introduction, 01_2_longer_context, 01_3_local_model).
    - The study planner uses simple heuristics for duration and topic extraction, so the plans are intentionally coarse.
    - Routing between services is keyword-based (e.g., “weather”, “study plan”), not learned.

