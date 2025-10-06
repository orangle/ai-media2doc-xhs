from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import logging
import os
from typing import Dict, List, Optional

import ffmpeg
import requests

from .video_utils import extract_audio

LOGGER = logging.getLogger(__name__)

IFLOW_API_URL = os.getenv("IFLOW_API_URL", "https://api.iflow.cn/v1/chat/completions")
IFLOW_MODEL_ASR = os.getenv("IFLOW_MODEL_ASR", "qwen3-max")


def _load_whisper() -> Optional[object]:
    spec = importlib.util.find_spec("whisper")
    if spec is None:
        return None
    return importlib.import_module("whisper")


def _get_audio_duration(audio_path: str) -> float:
    probe = ffmpeg.probe(audio_path)
    duration = probe.get("format", {}).get("duration")
    return float(duration) if duration else 0.0


def _call_iflow_for_transcription(audio_path: str) -> List[Dict]:
    api_key = os.getenv("IFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("IFLOW_API_KEY is not set.")

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "model": IFLOW_MODEL_ASR,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise speech-to-text engine. Return the exact transcription of the audio in Chinese.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio_format": "wav",
                        "audio": audio_b64,
                    }
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(IFLOW_API_URL, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]["content"]

    if isinstance(message, list):
        text = "".join(part.get("text", "") for part in message)
    else:
        text = str(message)

    duration = _get_audio_duration(audio_path)
    return [
        {
            "start": 0.0,
            "end": duration,
            "text": text.strip(),
        }
    ]


def transcribe(video_path: str) -> List[Dict]:
    audio_path = extract_audio(video_path)

    whisper_module = _load_whisper()
    if whisper_module is not None:
        try:
            model = whisper_module.load_model("base")
            result = model.transcribe(audio_path, task="transcribe", word_timestamps=True)
            segments = result.get("segments", [])
            if segments:
                return [
                    {
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", 0.0)),
                        "text": segment.get("text", "").strip(),
                    }
                    for segment in segments
                ]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Whisper transcription failed, falling back to iFlow: %s", exc)

    LOGGER.info("Falling back to Qwen3-Max transcription via iFlow API.")
    return _call_iflow_for_transcription(audio_path)
