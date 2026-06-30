# Vision Assist — Captioning Service

The AI module of **Vision Assist**, a real-time scene-description system for visually impaired users.

This is a single FastAPI service that captions a video frame with **Qwen2-VL-2B**
(`MarinaMohsen/qwen2-vl-2b-nav-assistant`, a navigation-assistant fine-tune) and classifies the
scene as `SAFE` or `DANGEROUS`. It is the AI node in the wider system:

```
Next.js web app  ──WebSocket──▶  Spring Boot backend  ──HTTP POST /caption──▶  this service (Qwen2-VL)
 (captures frames)                (orchestrator)                                (caption + classify)
```

The browser captures frames and streams them to the Spring backend over a websocket; the backend
forwards each frame to this service's `/caption` endpoint and relays the caption/classification back.

The model wants a GPU, so this service is deployed to **Modal** (serverless GPU — see `modal_app.py`)
and the backend calls it by URL. It also runs on a CPU host (e.g. a Hugging Face Docker Space via the
`Dockerfile`), just much slower.

## API

### `POST /caption`

Request:

```json
{
  "image_base64": "<raw or data-url base64 JPEG/PNG>",
  "prompt": null
}
```

- `image_base64` — base64-encoded frame. A `data:image/...;base64,` prefix is accepted and stripped.
- `prompt` — optional instruction, sent to the model as a **system** message (the user turn carries
  only the image). When omitted, the configured `CAPTION_PROMPT` (a navigation system prompt) is used.
  Qwen2-VL is an instruct model, so both the role and the wording shape the output.

Response `200`:

```json
{
  "caption": "Clear sidewalk ahead. A lamppost is slightly to your right; keep left to pass it.",
  "classification": { "label": "SAFE", "reason": "No danger keywords detected." },
  "metadata": {
    "model": "MarinaMohsen/qwen2-vl-2b-nav-assistant",
    "device": "cuda",
    "input_tokens": 312,
    "output_tokens": 24,
    "total_tokens": 336,
    "latency_ms": 540.2,
    "image_width": 640,
    "image_height": 480,
    "generated_at": "2026-06-17T12:00:00+00:00"
  }
}
```

- `input_tokens` includes the expanded image (vision) tokens, so it scales with the image size.
- `output_tokens` is the number of tokens the model generated.

Errors: `400` (invalid base64 / undecodable image), `503` (model still loading).

### `GET /health`

```json
{ "status": "ok", "model": "MarinaMohsen/qwen2-vl-2b-nav-assistant", "device": "cuda", "model_loaded": true }
```

## Configuration

Environment variables (see `.env.example`):

| Variable          | Default            | Description                                             |
| ----------------- | ------------------ | ------------------------------------------------------- |
| `MODEL_NAME`      | `MarinaMohsen/qwen2-vl-2b-nav-assistant` | Hugging Face model id.            |
| `MAX_NEW_TOKENS`  | `40`               | Max generated tokens per caption — a ceiling, not a target (lower = faster). |
| `CAPTION_PROMPT`  | nav system prompt  | Instruction sent with every frame as a **system** message; default mirrors the model author's Space. |
| `ALLOWED_ORIGINS` | `*`                | Comma-separated CORS origins (set to the backend origin in prod). |
| `PORT`            | `7860`             | Listen port (Hugging Face Spaces require 7860).         |

## Project structure

```
app/
  main.py                 # FastAPI app, lifespan model load, routes: POST /caption, GET /health
  config.py               # env-driven settings
  schemas.py              # Pydantic request/response models
  captioning/model.py     # Captioner: load Qwen2-VL + generate() -> caption, tokens, latency
  classification/classifier.py  # DangerClassifier (context-aware SAFE/DANGEROUS regex)
tests/                    # pytest (classifier cases + API tests with a stubbed model)
modal_app.py              # Modal (serverless GPU) deployment of the FastAPI app
Dockerfile                # CPU/GPU container image (port 7860)
requirements.txt
```

## Local development

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

The first run downloads the model weights (several GB). Qwen2-VL-2B is slow on CPU; a CUDA GPU is used
automatically when available.

Smoke test once `GET /health` reports `model_loaded: true`:

```bash
curl -X POST localhost:7860/caption \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<base64-image>"}'
```

### Tests

The test suite stubs the model, so it runs fast and needs only the lightweight deps
(`fastapi pydantic Pillow pytest httpx`) — no `torch`/`transformers`:

```bash
pytest
```

## Deployment

**Modal (serverless GPU, recommended):** from this directory, `pip install modal`, `modal token new`,
then `modal deploy modal_app.py`. Modal prints the public URL (serving `POST /caption` + `GET /health`)
to use as the backend's `AI_BASE_URL`. Weights are cached in a Modal Volume across cold starts.

**Hugging Face Space (CPU, alternative):** create a **Docker (Blank)** Space and push this repo; the
`Dockerfile` runs the service on port 7860 (CPU → multi-second latency).
