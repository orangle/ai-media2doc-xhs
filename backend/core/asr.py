from __future__ import annotations

import base64
import contextlib
import importlib
import importlib.util
import json
import logging
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ffmpeg
from shared import iflow_api

from .video_utils import extract_audio

LOGGER = logging.getLogger(__name__)

IFLOW_MODEL_ASR = os.getenv("IFLOW_MODEL_ASR", "qwen3-max")

DEFAULT_TIMEOUT_S = 180.0
DEFAULT_SEGMENT_S = 45.0


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid float for %s=%s, using default %.2f", name, value, default)
        return default


def _load_whisper() -> Optional[object]:
    spec = importlib.util.find_spec("whisper")
    if spec is None:
        return None
    return importlib.import_module("whisper")


def _get_audio_duration(audio_path: str) -> float:
    probe = ffmpeg.probe(audio_path)
    duration = probe.get("format", {}).get("duration")
    return float(duration) if duration else 0.0


def _clamp_segment_length(segment_length: float) -> float:
    if math.isfinite(segment_length) and segment_length > 0:
        return max(30.0, min(segment_length, 60.0))
    return 45.0


def _split_audio_segments(audio_path: str, segment_length: float) -> Iterable[Tuple[float, float, str, bool]]:
    duration = _get_audio_duration(audio_path)
    if duration <= 0:
        yield (0.0, 0.0, audio_path, False)
        return

    segment_length = _clamp_segment_length(segment_length)
    if duration <= segment_length:
        yield (0.0, duration, audio_path, False)
        return

    start = 0.0
    while start < duration:
        end = min(duration, start + segment_length)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            (
                ffmpeg
                .input(audio_path, ss=start, t=end - start)
                .output(
                    temp_path,
                    format="wav",
                    acodec="pcm_s16le",
                    ac=1,
                    ar=16000,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:  # noqa: BLE001
            LOGGER.error("Failed to split audio segment %.2f-%.2f: %s", start, end, exc)
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp_path)
            raise
        yield (start, end, temp_path, True)
        start = end


def _post_process_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    # Normalize common Western punctuation to Chinese full-width variants.
    punctuation_map = {
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
    }
    for src, target in punctuation_map.items():
        text = re.sub(rf"\s*{re.escape(src)}\s*", target, text)
    text = re.sub(r"([。！？；，])\1+", r"\1", text)
    if text and text[-1] not in "。！？；：":
        text = f"{text}。"
    return text


def _call_iflow_for_segment(audio_path: str, timeout: float) -> str:
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    messages = [
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
    ]

    data = iflow_api.chat_completion(
        IFLOW_MODEL_ASR,
        messages,
        timeout_s=timeout,
        temperature=0.2,
    )
    message = data["choices"][0]["message"]["content"]

    if isinstance(message, list):
        text = "".join(part.get("text", "") for part in message)
    else:
        text = str(message)

    return _post_process_text(text)


def _select_whisper_model(whisper_module: object) -> Optional[object]:
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            return _try_load_whisper_models(whisper_module, ("small", "tiny"))
    return _try_load_whisper_models(whisper_module, ("tiny", "small"))


def _try_load_whisper_models(whisper_module: object, preferred: Sequence[str]) -> Optional[object]:
    for model_size in preferred:
        try:
            LOGGER.info("Loading Whisper model: %s", model_size)
            return whisper_module.load_model(model_size)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load Whisper model %s: %s", model_size, exc)
    return None


def _transcribe_with_whisper(
    audio_path: str,
    whisper_module: object,
    segment_length: float,
) -> List[Dict]:
    model = _select_whisper_model(whisper_module)
    if model is None:
        raise RuntimeError("Unable to load Whisper model.")

    segments: List[Dict] = []
    last_end = 0.0
    for start, end, segment_path, cleanup in _split_audio_segments(audio_path, segment_length):
        try:
            result = model.transcribe(
                segment_path,
                task="transcribe",
                word_timestamps=False,
                language="zh",
                temperature=0,
                verbose=False,
            )
        finally:
            if cleanup:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(segment_path)

        chunk_segments = result.get("segments") if isinstance(result, dict) else None
        if chunk_segments:
            for chunk in chunk_segments:
                raw_text = str(chunk.get("text", ""))
                cleaned = _post_process_text(raw_text)
                if not cleaned:
                    continue
                chunk_start = start + float(chunk.get("start", 0.0))
                chunk_end = start + float(chunk.get("end", 0.0))
                chunk_start = max(chunk_start, last_end)
                if chunk_end < chunk_start:
                    chunk_end = chunk_start
                segments.append(
                    {
                        "start": float(chunk_start),
                        "end": float(chunk_end),
                        "text": cleaned,
                    }
                )
                last_end = float(chunk_end)
        else:
            raw_text = str(result.get("text", "")) if isinstance(result, dict) else str(result)
            cleaned = _post_process_text(raw_text)
            if cleaned:
                seg_start = max(start, last_end)
                seg_end = max(seg_start, end)
                segments.append(
                    {
                        "start": float(seg_start),
                        "end": float(seg_end),
                        "text": cleaned,
                    }
                )
                last_end = float(seg_end)

    # Ensure timestamps are monotonically increasing.
    for idx in range(1, len(segments)):
        prev_end = segments[idx - 1]["end"]
        if segments[idx]["start"] < prev_end:
            segments[idx]["start"] = prev_end
        if segments[idx]["end"] < segments[idx]["start"]:
            segments[idx]["end"] = segments[idx]["start"]

    return segments


def _transcribe_with_iflow(
    audio_path: str,
    segment_length: float,
    timeout: float,
) -> List[Dict]:
    segments: List[Dict] = []
    last_end = 0.0
    for start, end, segment_path, cleanup in _split_audio_segments(audio_path, segment_length):
        try:
            text = _call_iflow_for_segment(segment_path, timeout)
        finally:
            if cleanup:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(segment_path)

        if not text:
            last_end = end
            continue
        seg_start = max(start, last_end)
        seg_end = max(seg_start, end)
        segments.append(
            {
                "start": float(seg_start),
                "end": float(seg_end),
                "text": text,
            }
        )
        last_end = float(seg_end)

    return segments


def transcribe(video_path: str) -> List[Dict]:
    audio_path = extract_audio(video_path)

    whisper_enabled = _bool_from_env("WHISPER_ENABLE", True)
    timeout_s = _float_from_env("ASR_TIMEOUT_S", DEFAULT_TIMEOUT_S)
    segment_length = _clamp_segment_length(_float_from_env("ASR_SEGMENT_S", DEFAULT_SEGMENT_S))

    whisper_module = _load_whisper() if whisper_enabled else None

    if whisper_enabled and whisper_module is not None:
        LOGGER.info("Using Whisper for transcription.")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _transcribe_with_whisper,
                    audio_path,
                    whisper_module,
                    segment_length,
                )
                result = future.result(timeout=timeout_s)
            if result:
                return result
        except FuturesTimeout:
            LOGGER.warning("Whisper transcription timed out after %.1fs", timeout_s)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Whisper transcription failed: %s", exc)

    LOGGER.info("Falling back to Qwen3-Max transcription via iFlow API.")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _transcribe_with_iflow,
                audio_path,
                segment_length,
                timeout_s,
            )
            result = future.result(timeout=timeout_s)
        return result
    except FuturesTimeout:
        LOGGER.error("iFlow transcription timed out after %.1fs", timeout_s)
        raise
