"""
Deploy the Vision Assist captioning service to Modal (serverless GPU).

Reuses the existing FastAPI app (app.main:app) unchanged, runs it on a GPU, and caches the model
weights in a Modal Volume so cold starts after the first one are fast.

Deploy (run from the vision-assist-ai/ directory):
    pip install modal
    modal token new            # one-time browser auth
    modal deploy modal_app.py

Modal prints a public URL like:
    https://<workspace>--vision-assist-ai-web.modal.run
That base URL is your backend's AI_BASE_URL (it serves POST /caption and GET /health).

Note: this file (not the Dockerfile) is what Modal uses; the Dockerfile is for HF Space / other hosts.
Modal's API can shift between versions — if a keyword below is rejected, check https://modal.com/docs.
"""

import modal

app = modal.App("vision-assist-ai")

# Persisted cache for downloaded model weights (avoids re-downloading on every cold start).
model_cache = modal.Volume.from_name("vision-assist-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": "/cache", "MODEL_NAME": "MarinaMohsen/qwen2-vl-2b-nav-assistant"})
    .add_local_dir("app", remote_path="/root/app")
)


@app.function(
    image=image,
    gpu="T4",                  # fits Qwen2-VL-2B (fp16, ~4.4 GB); bump to "L4"/"A10G" for speed
    volumes={"/cache": model_cache},
    timeout=600,               # hard cap per request (kills a runaway generate())
    # --- cost guards ---
    max_containers=1,          # never autoscale to >1 GPU (caps the worst-case burn rate)
    scaledown_window=120,      # stay warm 2 min after the last request, then scale to zero
    # (older Modal versions: use concurrency_limit=1 instead of max_containers)
    # secrets=[modal.Secret.from_name("huggingface")],  # uncomment only if the model repo is gated
)
@modal.concurrent(max_inputs=8)  # handle bursts on the SINGLE container instead of spawning more
@modal.asgi_app()
def web():
    from app.main import app as fastapi_app

    return fastapi_app
