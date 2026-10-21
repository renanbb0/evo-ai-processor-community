"""Audio reaches the model as a transcript, never as an inline blob, when a
transcription key is available.

Inline audio becomes an `audio_url` part that OpenAI rejects ("Invalid user
message"); the processor answers 503 and, since the blob is kept in the session,
every later turn of the conversation fails too.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.schemas.chat import FileData
from src.services import audio_transcription_service as ats
from src.services.adk.runners import runner_utils
from src.services.adk.runners.runner_utils import RunnerUtils


def _utils(key):
    u = RunnerUtils.__new__(RunnerUtils)
    u._resolve_transcription_key = AsyncMock(return_value=key)
    return u


def _artifacts():
    a = MagicMock()
    a.save_artifact = AsyncMock()
    return a


def _file(name, ctype):
    return FileData(
        filename=name, content_type=ctype, data=base64.b64encode(b"bytes-" + name.encode()).decode()
    )


def _run(coro):
    return asyncio.run(coro)


def test_audio_is_sent_as_transcript_not_inline(monkeypatch):
    transcribe = AsyncMock(return_value="quero cancelar o pedido")
    monkeypatch.setattr(runner_utils, "transcribe_audio", transcribe)
    artifacts = _artifacts()

    parts, transcribed = _run(
        _utils("sk-test").process_files(
            [_file("voice.ogg", "audio/opus")], artifacts, "a", "e", "s"
        )
    )

    assert parts == []
    assert transcribed == ["[Audio message transcription]: quero cancelar o pedido"]
    transcribe.assert_awaited_once_with(b"bytes-voice.ogg", "voice.ogg", "sk-test")
    artifacts.save_artifact.assert_awaited_once()


def test_failed_transcription_still_keeps_audio_out_of_the_model(monkeypatch):
    monkeypatch.setattr(runner_utils, "transcribe_audio", AsyncMock(return_value=None))

    parts, transcribed = _run(
        _utils("sk-test").process_files(
            [_file("voice.ogg", "audio/ogg")], _artifacts(), "a", "e", "s"
        )
    )

    assert parts == []
    assert transcribed == [
        "[The customer sent an audio message that could not be transcribed.]"
    ]


def test_without_key_audio_stays_inline(monkeypatch):
    transcribe = AsyncMock()
    monkeypatch.setattr(runner_utils, "transcribe_audio", transcribe)

    parts, transcribed = _run(
        _utils(None).process_files(
            [_file("voice.ogg", "audio/ogg")], _artifacts(), "a", "e", "s"
        )
    )

    assert [p.inline_data.mime_type for p in parts] == ["audio/ogg"]
    assert transcribed == []
    transcribe.assert_not_awaited()


def test_image_is_untouched_and_key_resolved_once(monkeypatch):
    monkeypatch.setattr(runner_utils, "transcribe_audio", AsyncMock(return_value="oi"))
    utils = _utils("sk-test")

    parts, transcribed = _run(
        utils.process_files(
            [
                _file("p.png", "image/png"),
                _file("a.ogg", "audio/ogg"),
                _file("b.ogg", "audio/ogg"),
            ],
            _artifacts(),
            "a",
            "e",
            "s",
        )
    )

    assert [p.inline_data.mime_type for p in parts] == ["image/png"]
    assert len(transcribed) == 2
    utils._resolve_transcription_key.assert_awaited_once()


def test_transcript_lands_in_the_user_message():
    content = RunnerUtils.__new__(RunnerUtils).create_content_with_transcribed_audio(
        "", [], ["[Audio message transcription]: oi"]
    )
    assert content.parts[0].text.strip() == "[Audio message transcription]: oi"
    assert len(content.parts) == 1


@pytest.mark.parametrize(
    "key,provider,expected",
    [
        ("sk-proj-abc", "openai", True),
        ("sk-abc", None, True),
        ("sk-or-v1-abc", None, False),
        ("sk-ant-abc", "anthropic", False),
        ("sk-ant-abc", None, False),
        ("AIzaSy-abc", "gemini", False),
        (None, "openai", False),
    ],
)
def test_is_openai_key(key, provider, expected):
    assert ats.is_openai_key(key, provider) is expected


def test_resolution_prefers_env_and_honours_disable(monkeypatch):
    utils = RunnerUtils.__new__(RunnerUtils)
    monkeypatch.setenv("AUDIO_TRANSCRIPTION_API_KEY", "sk-env")
    assert _run(utils._resolve_transcription_key("a")) == "sk-env"

    monkeypatch.setenv("AUDIO_TRANSCRIPTION_ENABLED", "false")
    assert _run(utils._resolve_transcription_key("a")) is None
