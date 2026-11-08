"""Groq adapter. Converts a Groq chat completion into the shape run_agent expects."""
import json
import os

from dotenv import load_dotenv
from groq import Groq

from app.tools import TOOL_SCHEMAS

load_dotenv()
MODEL = "openai/gpt-oss-120b"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def groq_llm(messages):
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=0.2,  # Groq's tool-use examples recommend 0.0 to 0.5
    )
    msg = resp.choices[0].message

    if not msg.tool_calls:
        content = msg.content or ""
        return {"text": content, "raw": {"role": "assistant", "content": content}}

    raw = {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    }
    calls = []
    for tc in msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = None  # run_agent reports this back to the model as an error
        calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    return {"tool_calls": calls, "raw": raw}


if __name__ == "__main__":
    from app.agent import run_agent

    print(run_agent("How many cars are in this image?", "data/images/test1.jpg", groq_llm))