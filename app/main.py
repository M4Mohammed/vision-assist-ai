"""
Vision Assist captioning service.

A single FastAPI app that the Spring backend calls over HTTP (frame-relay): it captions a
video frame with Qwen2-VL and classifies it as SAFE or DANGEROUS, returning the caption plus
token/latency metadata.
"""

import base64
import binascii
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.captioning.model import Captioner
from app.classification.classifier import DangerClassifier
from app.config import settings
from app.schemas import (
    CaptionMetadata,
    CaptionRequest,
    CaptionResponse,
    Classification,
    HealthResponse,
)

# Module-level singletons. The model is loaded once at startup (lifespan); the classifier is cheap.
captioner = Captioner()
classifier = DangerClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    captioner.load()
    yield


app = FastAPI(
    title="Vision Assist Captioning Service",
    description="Captions a video frame with Qwen2-VL and classifies it as SAFE or DANGEROUS.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_image(image_base64: str) -> Image.Image:
    """Decodes a (possibly data-URL prefixed) base64 string into an RGB image."""
    raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    try:
        image_bytes = base64.b64decode(raw)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}") from exc
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Could not decode image.") from exc


@app.post("/caption", response_model=CaptionResponse)
def caption(request: CaptionRequest) -> CaptionResponse:
    if not captioner.is_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    image = _decode_image(request.image_base64)
    result = captioner.generate(image, prompt=request.prompt)
    label, reason = classifier.classify(result["caption"])

    return CaptionResponse(
        caption=result["caption"],
        classification=Classification(label=label, reason=reason),
        metadata=CaptionMetadata(
            model=captioner.model_name,
            device=captioner.device,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["total_tokens"],
            latency_ms=result["latency_ms"],
            image_width=image.width,
            image_height=image.height,
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if captioner.is_loaded else "loading",
        model=captioner.model_name,
        device=captioner.device,
        model_loaded=captioner.is_loaded,
    )
