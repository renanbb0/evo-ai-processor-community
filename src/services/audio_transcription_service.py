"""Speech-to-text for incoming audio (WhatsApp voice notes).

Audio used to reach the model as an inline blob, which google-adk's LiteLlm turns
into an `audio_url` content part. OpenAI chat models reject that part
("Invalid user message"), the runner answers 503, and because the blob is kept in
the ADK session history every later turn of that conversation fails the same way.

Transcribing first and sending text keeps the conversation provider-agnostic.

Env:
    AUDIO_TRANSCRIPTION_ENABLED   "false" disables it (default: enabled)
    AUDIO_TRANSCRIPTION_MODEL     default "gpt-4o-mini-transcribe"
    AUDIO_TRANSCRIPTION_LANGUAGE  ISO-639-1 hint, e.g. "pt" (default: auto-detect)
    AUDIO_TRANSCRIPTION_API_KEY   OpenAI key; falls back to OPENAI_API_KEY, then
                                  to the agent's own key when it is an OpenAI key
    AUDIO_TRANSCRIPTION_TIMEOUT   seconds, default 30
"""

import io
import os
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TIMEOUT_SECONDS = 30.0

# Providers whose stored key is accepted by the OpenAI transcription endpoint.
_OPENAI_PROVIDERS = {None, "", "openai"}


def transcription_enabled() -> bool:
    return os.getenv("AUDIO_TRANSCRIPTION_ENABLED", "true").strip().lower() not in {
        "false",
        "0",
        "no",
        "off",
    }


def env_api_key() -> Optional[str]:
    return (
        os.getenv("AUDIO_TRANSCRIPTION_API_KEY") or os.getenv("OPENAI_API_KEY") or None
    )


def is_openai_key(api_key: Optional[str], provider: Optional[str]) -> bool:
    """Whether an agent's key can be reused for OpenAI transcription."""
    if not api_key:
        return False
    if (provider or "").strip().lower() not in _OPENAI_PROVIDERS:
        return False
    # An unlabeled key may belong to any provider. OpenAI keys start with sk-,
    # but so do OpenRouter (sk-or-) and Anthropic (sk-ant-) keys.
    return api_key.startswith("sk-") and not api_key.startswith(("sk-or-", "sk-ant-"))


async def transcribe_audio(
    data: bytes, filename: str, api_key: str
) -> Optional[str]:
    """Return the transcript, or None when it fails or the audio is silent.

    Never raises: a failed transcription must not cost the customer the turn.
    """
    try:
        from openai import AsyncOpenAI

        timeout = float(
            os.getenv("AUDIO_TRANSCRIPTION_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        )
        client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=1)

        # The endpoint picks the decoder from the file name, so it must keep its
        # real extension (.ogg for WhatsApp voice notes).
        audio_file = io.BytesIO(data)
        audio_file.name = filename or "audio.ogg"

        kwargs = {
            "model": os.getenv("AUDIO_TRANSCRIPTION_MODEL", DEFAULT_MODEL),
            "file": audio_file,
        }
        language = os.getenv("AUDIO_TRANSCRIPTION_LANGUAGE")
        if language:
            kwargs["language"] = language

        result = await client.audio.transcriptions.create(**kwargs)
        text = (getattr(result, "text", None) or "").strip()
        logger.info(f"Transcribed audio {filename}: {len(text)} chars")
        return text or None
    except Exception as e:
        logger.error(f"Audio transcription failed for {filename}: {e}")
        return None
