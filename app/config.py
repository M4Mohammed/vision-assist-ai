"""Runtime configuration, read from environment variables with sensible defaults."""

import os


class Settings:
    # Hugging Face model id for the Qwen2-VL navigation-assistant captioner.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "MarinaMohsen/qwen2-vl-2b-nav-assistant")

    # Upper bound on generated tokens per caption. This is a CEILING, not a target length — greedy
    # decoding stops at EOS on its own, so a short caption isn't padded to this. It only bites when
    # the model would ramble past it. Decode latency is ~linear in output tokens, so a lower cap is
    # faster and fits the live ~3s/TTS cadence; 40 matches the model author's reference usage. Watch
    # the response's output_tokens — if captions peg at this value they're being truncated mid-thought
    # (raise it); if they finish well under, raising it only costs latency.
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "40"))

    # Instruction sent with every frame, used as a SYSTEM message (Qwen2-VL is an instruct model, so
    # both the wording AND the role matter). This is the model author's own verbatim prompt from their
    # nav-assistant Space (huggingface.co/spaces/MarinaMohsen/qwen2-2B) — i.e. the configuration this
    # checkpoint was developed/validated against, which is why we mirror it rather than invent one.
    CAPTION_PROMPT: str = os.getenv(
        "CAPTION_PROMPT",
        "You are a real-time assistant for a visually impaired user. Describe the scene accurately "
        "in one or two short sentences. If there is a hazard, obstacle, or danger ahead, describe "
        "what it is and its location (e.g., 'on the left'). If the path is clear, simply describe "
        "the environment and surroundings naturally. Do not say 'no hazards' or 'the scene is safe', "
        "instead describe the scene like objects and people...",
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
