"""Handling for POST /ask. Pure logic: no HTTP, no model, no LLM import.

Returns (http_status, body). The HTTP layer in main.py only binds the request and the LLM.
"""
from app import db
from app.agent import run_agent
from app.question_guard import MAX_QUESTION_CHARS, check_question


def handle_question(conn, image_id, question, llm):
    image = db.get_image(conn, image_id)
    if image is None or not db.has_detections(conn, image_id):
        return 404, {"error": "no cached detections for this image; upload it with /detect first"}

    verdict = check_question(question)
    if not verdict.allowed:
        stored = question[:MAX_QUESTION_CHARS] if isinstance(question, str) else ""
        run_id = db.start_run(conn, stored, image_id)
        db.finish_run(conn, run_id, "rejected", None, 0, 0,
                      error="flagged: " + ", ".join(verdict.flags))
        return 400, {"status": "rejected", "run_id": run_id,
                     "error": "question rejected by input guard"}

    result = run_agent(conn, question, image_id, llm)
    if result["status"] == "llm_error":
        return 503, {"status": "llm_error", "run_id": result["run_id"],
                     "error": "model unavailable; try again"}
    return 200, result
