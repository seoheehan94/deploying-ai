# openai_client.py

import os
from openai import OpenAI


def get_client() -> OpenAI:
    """
    Return an OpenAI client configured to talk to the course API Gateway.
    """
    return OpenAI(
        base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
        api_key="any value",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
    )
