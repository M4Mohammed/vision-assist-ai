# Vision Assist captioning service.
# Runs on a Hugging Face Docker Space OR a GPU server (port 7860). For GPU, run the container with
# the NVIDIA runtime (`--gpus all`) and a recent host driver.
FROM python:3.11

WORKDIR /code

# Model weights cache (mount a volume here to persist across restarts).
ENV HF_HOME=/code/hf_home
ENV PYTHONUNBUFFERED=1

# Install a CUDA 12.8 build of PyTorch + torchvision first. cu128 is required for Blackwell GPUs
# (e.g. RTX PRO 6000); it also runs on older NVIDIA cards and falls back to CPU when no GPU is
# present. Doing this before requirements means torch/torchvision are already satisfied and won't be
# replaced by default wheels. torchvision is needed by Qwen2-VL's AutoProcessor.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Remaining Python dependencies.
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Application package.
COPY app /code/app

# Optional: pre-download the model at build time to cut cold-start latency.
# Pulls several GB into the image, so it is disabled by default (we use a persistent HF_HOME volume).
# RUN python -c "from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; \
#     AutoProcessor.from_pretrained('MarinaMohsen/qwen2-vl-2b-nav-assistant'); \
#     Qwen2VLForConditionalGeneration.from_pretrained('MarinaMohsen/qwen2-vl-2b-nav-assistant')"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
