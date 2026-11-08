"""Hand-written tool-calling loop. The LLM is injected so tests can fake it."""
from app.tools import TOOLS

MAX_STEPS = 4


def run_agent(question, image_path, llm):
    """llm(messages) returns {"text": str} or {"tool_calls": [{"name", "args"}]}."""
    messages = [{"role": "user", "content": f"{question}\nImage: {image_path}"}]
    detections = None

    for _ in range(MAX_STEPS):
        resp = llm(messages)
        if not resp.get("tool_calls"):
            return {"answer": resp["text"], "status": "ok"}

        for call in resp["tool_calls"]:
            name, args = call["name"], call.get("args", {})
            if name not in TOOLS:
                messages.append({"role": "tool", "content": f"error: unknown tool {name}"})
                continue
            try:
                if name == "detect_objects":
                    detections = TOOLS[name](image_path, **args)
                    result = detections
                else:
                    result = TOOLS[name](detections or [], **args)
            except Exception as e:
                # Bad arguments or runtime errors become data for the model, not a crash.
                result = f"error: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "content": str(result)})

    return {"answer": None, "status": "max_steps"}


def fake_llm(messages):
    """Scripted stand-in: first asks for detection, then counts, then answers."""
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    if len(tool_msgs) == 0:
        return {"tool_calls": [{"name": "detect_objects", "args": {}}]}
    if len(tool_msgs) == 1:
        return {"tool_calls": [{"name": "count_labels", "args": {"label": "car"}}]}
    return {"text": f"Result: {tool_msgs[-1]['content']}"}


if __name__ == "__main__":
    print(run_agent("How many cars are in this image?", "data/images/test1.jpg", fake_llm))