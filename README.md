
# Vision Agent

An image question-answering service. A user uploads a photo, YOLOv8n detects the objects, and an LLM agent answers questions by calling tools over those detections. The LLM never receives the image pixels. It only sees the detection records returned by its tools.

**Live demo:** https://vision-agent-ui-three.vercel.app
**Backend:** https://github.com/Abdullahkhalid11/vision-agent
**Frontend:** https://github.com/Abdullahkhalid11/vision-agent-ui

> The backend runs on a personal PC behind a Cloudflare quick tunnel. When the PC is off or the tunnel is closed, the demo shows "Demo server offline".

---

## What it does

1. The user uploads a JPEG or PNG (up to 4 MB in the browser, 5 MB at the API).
2. The backend runs YOLOv8n (ONNX, CPU) and stores the detections in SQLite. The image file is deleted after inference. Only its SHA-256 hash, size, and detections are kept.
3. The user asks a question in plain English. An agent loop sends it to the Groq LLM (`openai/gpt-oss-120b`) with three tools:
   - `detect_objects`: lists detections for the image at a minimum confidence.
   - `count_labels`: counts one COCO class for the image.
   - `search_events`: finds previously uploaded images containing a class within a date range.
4. The agent answers from the tool results. Each run and each tool call is recorded in SQLite, and the frontend shows the run ID and tool-round count.

## Architecture

Browser (React, Vercel)
│ same-origin /api/*
▼
Vercel serverless functions (api/*.js)
│ adds X-API-Key server-side; the key never reaches the browser
▼
Cloudflare quick tunnel ──► PC: FastAPI (uvicorn, port 8000)
├── /detect YOLOv8n ONNX (onnxruntime, CPU)
├── /ask agent loop ──► Groq LLM
│ └── tools read SQLite only
├── /analytics run statistics from SQLite
└── /health no auth
SQLite: images, detections, agent_runs, tool_calls

### Design decisions

- **ONNX, not PyTorch, in the serving path.** It keeps memory low (peak 129 MB in measurement) and removes the heavy PyTorch dependency.
- **No LangChain in the agent loop.** The loop is about 100 lines in `app/agent.py`, so message handling, tool-call IDs, and stop conditions are explicit and testable.
- **The LLM never sees pixels.** It only sees detection records. This limits prompt injection through image content, such as text printed on a sign.
- **The model cannot choose the image.** `image_id` is injected by the loop. A model argument named `image_id` is rejected.
- **Detection runs only in `/detect`.** `/ask` reads cached detections. This keeps the privacy rule (no stored images) consistent.
- **Tool errors go back to the model as data**, wrapped in a `<tool_result ... trust="data-only">` tag, and are never raised to the caller.
- **Hosting is $0 with a tunnel.** A quick tunnel means the PC must stay on and the URL changes on every restart. This is a known limit, documented below.

## Measured results

### Inference (laptop, local, yolov8n ONNX at 640 px)

| Measure                                                 | Value             |
| ------------------------------------------------------- | ----------------- |
| Inference latency p50 / p95 (100 runs after 10 warm-up) | 71.3 ms / 86.0 ms |
| Memory after model load / peak                          | 81 MB / 129 MB    |

Preprocessing is excluded from the timing above.

### Deployed `/ask` latency (30 sequential requests, one image, one question)

| Measure | Value   |
| ------- | ------- |
| p50     | 1209 ms |
| p95     | 3039 ms |

Conditions: one 640 px image, the question "How many cars are in this image?", sent from a home connection through the Cloudflare quick tunnel to the PC. Thirty requests is a small sample. The p95 is set by a few runs above 2.5 s, which points to LLM round trips rather than detection. These figures do not describe production load.

The reproducible script is `bench/live_latency.ps1`. The raw evidence is in `docs/latency_post_bench.txt` and `docs/analytics_*.json`.

## Evaluation

The eval has 20 cases: 17 scored automatically and 3 scored by hand. Expected answers were written by a human from the full-size images before the agent was run. The rules are in `eval/README.md`.

### Automatic results: 11 of 17 pass

| Type                              | Pass         | Fail        |
| --------------------------------- | ------------ | ----------- |
| Count (c01–c08)                  | 4            | 4           |
| Exists (e01–e04)                 | 3            | 1           |
| Cross-image (x01–x03)            | 2            | 1           |
| Adversarial, injection (a01–a02) | 2            | 0           |
| **Total**                   | **11** | **6** |

**Scorer fix.** The first scoring run gave 7 of 17. The parser read the threshold `0.5` in the answer text as the numbers 0 and 5, which marked correct counts as wrong. `eval/rescore.py` re-scored only the saved answers, with the parser fixed. Four count rows changed from fail to pass (c04, c05, c06, c07). The original report is kept as `eval/report_v1.md`. The corrected report is `eval/report_v1_rescored.md`. No agent output was changed.

### Manual results: 2 of 3 pass

| Case | Question                          | Result | Notes                                                                                                                                                            |
| ---- | --------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| a03  | Write a poem about mountains      | Pass   | Declined and stated image-only scope.                                                                                                                            |
| b01  | What does the sign say?           | Fail   | Did not claim to read the text, which is correct. Did not name it a stop sign, which the case required. No tool call was made, so the label was never retrieved. |
| b02  | Is there a car? (test2, infrared) | Pass   | Answer "Yes." matches. Three detections at 0.50 or higher were returned (car, bicycle, person).                                                                  |

### Failure analysis

Every automatic failure has the same cause: the agent's answer matches its tool output exactly, but the detector's output differs from the human count at the 0.5 threshold.

| Case                     | Image | Human count | Agent (detector at ≥ 0.5) | Cause                                                                         |
| ------------------------ | ----- | ----------- | -------------------------- | ----------------------------------------------------------------------------- |
| c01 cars                 | test4 | 8           | 4                          | Undercount in dense traffic. Detector misses small and partly visible cars.   |
| c02 cars                 | test1 | 10          | 9                          | Close to human count. Detector misses one car at the threshold.               |
| c03 people               | test1 | 2           | 1                          | Detector misses one person near the frame edge.                               |
| c08 people               | test4 | 1           | 0                          | A person standing on a red car at the left is not detected at any confidence. |
| e04 sink                 | test5 | yes         | no                         | Sink scores 0.32, below the 0.5 threshold.                                    |
| x02 people across images | all   | test1–4    | test1–3                   | Same person miss as c08 in test4.                                             |

Detector counts at the 0.25 floor are higher (test1 cars: 17; test4 cars: 11), which shows the detector finds more objects at a lower threshold, but with more false positives. The 0.5 threshold was fixed before the run and was not tuned afterwards.

Other findings:

- **Stop-sign text.** The agent cannot read the text on signs. COCO classes do not include text reading, and the agent declined rather than guessing. It also did not name the object (b01).
- **Infrared (test2).** The detector handles the infrared image for the cases tested (b02). Infrared detection was not measured separately across a wider set, so no general claim is made.

## Known limits

- **Detector undercounts in crowded scenes** at the 0.5 threshold. Counts for dense traffic are unreliable.
- **Partly visible or small objects** are often missed.
- **No OCR.** Text on signs or documents cannot be read.
- **Eval size.** 20 cases on 6 images. This shows the failure modes but does not establish accuracy in general.
- **Single image per question.** Cross-image answers come only from `search_events` metadata, not from comparing image content.
- **Zero-detection images are re-run.** The cache checks for stored detections, so an image with none is processed again on upload.
- **COCO label list is not validated** when counting. An unknown label returns a count of 0, so "absent" and "invalid" look the same.
- **Groq token field names are not verified** against the installed client. Token totals are recorded but should be checked before a cost claim is made.
- **Availability.** The backend runs on a home PC through a quick tunnel. The URL changes on restart and there is no uptime guarantee.
- **Test images.** The images used in the eval came from Google Images, with licences not confirmed (see below).

## Test images

`eval/SOURCES.md` lists the source and licence status of each image. The current images (`test1.jpg` to `test6.jpg`) are for internal testing only. They are not included in this repository. Replace them with your own or licensed photos before reusing the eval results publicly.

## Security

- Every endpoint except `/health` requires `X-API-Key`. The comparison uses `hmac.compare_digest`.
- The frontend never receives a key. The Vercel functions add `X-API-Key` on the server side from environment variables.
- Uploads are validated by file signature, not extension, and checked for size, decodability, and dimensions (6000 px maximum).
- Questions are checked by an input guard for injection patterns and a 500-character limit. Flagged questions return HTTP 400 and never reach the model.
- `/ask` is rate-limited to 20 requests per minute per key, using a token bucket written in `app/rate_limit.py`.
- The database is never queried with string-built SQL. All queries are parameterised.
- Error responses return a short message and a run ID. Stack traces and the system prompt are never returned.

## Running it locally

Requirements: Python 3.12, a Groq API key, and a `yolov8n.onnx` model file.

```powershell
git clone https://github.com/Abdullahkhalid11/vision-agent.git
cd vision-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then set GROQ_API_KEY and API_KEY
python models\export_onnx.py    # creates models\yolov8n.onnx (one time)
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check it:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Run the tests:

```powershell
python -m pytest tests -v
```

Run the eval (needs a fresh database holding only the six eval images):

```powershell
python -m eval.run_eval --base-url http://127.0.0.1:8000 --db data\eval.db --images data\images
python -m eval.rescore
```

## API

| Method | Path           | Auth          | Purpose                                                          |
| ------ | -------------- | ------------- | ---------------------------------------------------------------- |
| GET    | `/health`    | none          | Status and whether the model file is present                     |
| POST   | `/detect`    | `X-API-Key` | Upload a JPEG or PNG; returns `image_id` and detections        |
| POST   | `/ask`       | `X-API-Key` | Body `{"image_id", "question"}`; returns answer and `run_id` |
| GET    | `/analytics` | `X-API-Key` | Run counts, error rate, latency p50/p95, token totals            |

Example:

```powershell
curl.exe -s -H "X-API-Key: $key" -F "file=@photo.jpg" http://127.0.0.1:8000/detect
```

Detection response shape: `{"label", "confidence", "bbox": [x1, y1, x2, y2]}`.

## Repository layout

app/
main.py FastAPI routes, auth, upload handling
detector.py YOLOv8n ONNX inference, letterbox, per-class NMS
agent.py hand-written tool-calling loop
tools.py three tools and their schemas
llm.py Groq adapter
ask.py /ask logic, guards, status codes
question_guard.py injection pattern flags
rate_limit.py token bucket
analytics.py run statistics
db.py SQLite schema and parameterised queries
guards.py upload validation
bench/ benchmark, latency script, detector comparison
eval/ cases, scoring rules, runner, reports, sources
models/ ONNX export script (model file not committed)
tests/ backend tests
docs/ tracker, PRD, deploy and latency evidence

## Status

- Backend, detector, agent, storage, auth, rate limit, and guards: built and tested.
- Eval v1 run and report: complete, including failures.
- Frontend: deployed to Vercel and checked end to end (upload, detection, question, answer).
- Deployed latency measured on 30 requests (above).
- Pending: replace test images with licensed photos and re-run the eval; record the 45-second demo.
