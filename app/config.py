"""Runtime configuration, read from environment variables with sensible defaults."""

import os


class Settings:
    # Hugging Face model id for the Qwen2-VL navigation-assistant captioner.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "MarinaMohsen/qwen2-vl-2b-nav-assistant")

    # Upper bound on generated tokens per caption. Lower = faster (better for live 1 FPS).
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "64"))

    # Instruction sent with every frame. Qwen2-VL is an instruct model, so the prompt matters.
    # Tune this to match how the model was fine-tuned for best results.
    CAPTION_PROMPT: str = os.getenv(
        "CAPTION_PROMPT",
        "Describe the path ahead and any obstacles or hazards for a visually impaired person.",
    )

    # Comma-separated list of allowed CORS origins ("*" allows all).
    ALLOWED_ORIGINS: list = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ] or ["*"]

    # Port the service listens on (Hugging Face Spaces require 7860).
    PORT: int = int(os.getenv("PORT", "7860"))


settings = Settings()
