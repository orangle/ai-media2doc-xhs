from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from shared import iflow_api

LOGGER = logging.getLogger(__name__)

IFLOW_MODEL_VISION = os.getenv("IFLOW_MODEL_VISION", "qwen3-vl-plus")


def _timeout_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid timeout for %s=%s, falling back to %.1f", name, value, default)
        return default


VISION_TIMEOUT_S = _timeout_from_env("VISION_TIMEOUT_S", 45.0)


def _call_iflow(messages: List[Dict], image_paths: List[str]) -> Any:
    response = iflow_api.chat_vision(
        IFLOW_MODEL_VISION,
        messages,
        images=[{"path": path} for path in image_paths],
        timeout_s=VISION_TIMEOUT_S,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]


_VISUAL_SCHEMA = {
    "place": None,
    "activities": [],
    "objects": [],
    "mood": None,
    "visible_text": None,
}

_VISUAL_KEY_ALIASES = {
    "location": "place",
    "地点": "place",
    "activity": "activities",
    "activities": "activities",
    "活动": "activities",
    "objects": "objects",
    "object": "objects",
    "物体": "objects",
    "mood": "mood",
    "情绪": "mood",
    "text": "visible_text",
    "visible_text": "visible_text",
    "可读文字": "visible_text",
}


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    decoder = json.JSONDecoder()
    # First try strict JSON decoding (allows leading whitespace)
    try:
        obj, _ = decoder.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: search for the first JSON object substring
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    LOGGER.debug("Failed to extract JSON object from: %s", text)
    return {}


def _parse_content_to_dict(content: Any) -> Dict[str, Any]:
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    parsed = _extract_json_object(text)
    if not parsed:
        return {}

    normalized: Dict[str, Any] = {}
    for raw_key, value in parsed.items():
        alias = _VISUAL_KEY_ALIASES.get(raw_key, raw_key if raw_key in _VISUAL_SCHEMA else None)
        if not alias:
            continue
        if alias in {"activities", "objects"}:
            if isinstance(value, list):
                normalized[alias] = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str):
                normalized[alias] = [part.strip() for part in re.split(r"[、,，;；]", value) if part.strip()]
            else:
                normalized.setdefault(alias, [])
        elif alias == "visible_text":
            if isinstance(value, list):
                joined = " ".join(str(item).strip() for item in value if str(item).strip())
                normalized[alias] = joined or None
            elif isinstance(value, str):
                normalized[alias] = value.strip() or None
            else:
                normalized[alias] = None
        else:
            if value is None:
                normalized[alias] = None
            else:
                normalized[alias] = str(value).strip() or None

    result: Dict[str, Any] = {}
    for key, default in _VISUAL_SCHEMA.items():
        if key in {"activities", "objects"}:
            result[key] = list(normalized.get(key, default) or [])
        else:
            value = normalized.get(key, default)
            if isinstance(value, list):
                result[key] = None
            elif isinstance(value, str) and not value.strip():
                result[key] = None
            elif value == []:
                result[key] = None
            else:
                result[key] = value if value is not None else None

    return result


def light_rank(image_paths: List[str]) -> List[Dict]:
    """对候选帧进行轻量问答筛选"""

    if not image_paths:
        return []

    contents: List[Dict] = [
        {
            "type": "input_text",
            "text": (
                "You will review candidate frames from a travel video. For each frame, respond with a JSON object "
                "containing keys: path, has_landmark, has_readable_text, representativeness, brief. "
                "Return a JSON array in the same order as the images. "
                "Only judge based on visible content. If uncertain, use null. "
                "Mark has_landmark true only if a distinctive landmark/attraction is clearly shown. "
                "Mark has_readable_text true only if signage/boards/overlays with readable Chinese words or numbers are present. "
                "representativeness should be a number between 0 and 1 reflecting how well the frame summarizes the scene. "
                "brief should be a concise Chinese phrase mentioning key objects or actions and extract explicit keywords such as ¥, 元, 门票, 开放时间, 站, 出口, 博物馆 when visible. "
                "Do not guess the city or location unless text explicitly states it."
            ),
        }
    ]

    resolved_paths: List[str] = []
    for idx, image_path in enumerate(image_paths, start=1):
        image_path_str = str(Path(image_path).expanduser().resolve())
        resolved_paths.append(image_path_str)
        contents.append({"type": "input_text", "text": f"Frame {idx}: {Path(image_path_str).name}"})
        contents.append({"type": "input_image", "image_path": image_path_str})

    messages = [
        {
            "role": "system",
            "content": "You are a meticulous assistant helping to rank candidate frames. Only answer in strict JSON.",
        },
        {
            "role": "user",
            "content": contents,
        },
    ]

    content = _call_iflow(messages, resolved_paths)

    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.error("Failed to parse light_rank JSON response: %s", text)
        parsed = []

    results: List[Dict] = []
    for idx, path in enumerate(image_paths):
        data = parsed[idx] if idx < len(parsed) and isinstance(parsed[idx], dict) else {}
        results.append(
            {
                "path": path,
                "has_landmark": data.get("has_landmark"),
                "has_readable_text": data.get("has_readable_text"),
                "representativeness": data.get("representativeness"),
                "brief": data.get("brief", ""),
            }
        )

    return results


def analyze_frame(image_path: str) -> Dict:
    """调用 iFlow Qwen3-VL-Plus 模型，识别图片中的地点、活动、物体、情绪、可读文字，返回 JSON 字典"""
    image_path_str = str(Path(image_path).expanduser().resolve())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise vision analyst for travel videos. Only describe what is visible in the image. "
                "Respond strictly in JSON with keys: place (string or null), activities (array of strings), objects (array of strings), "
                "mood (string or null), visible_text (string or null). If unsure, use null and do not guess cities, prices, or hidden details."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Analyze this frame strictly based on visible evidence. List readable numbers or words in visible_text. "
                        "If any field is missing, set it to null (or [] for arrays)."
                    ),
                },
                {
                    "type": "input_image",
                    "image_path": image_path_str,
                },
            ],
        },
    ]

    attempts = 0
    while attempts < 2:
        content = _call_iflow(messages, [image_path_str])
        parsed = _parse_content_to_dict(content)
        if parsed:
            return parsed
        attempts += 1
        LOGGER.warning("Visual extraction returned non-JSON response, retrying (%s/2)", attempts)

    return {key: (value.copy() if isinstance(value, list) else value) for key, value in _VISUAL_SCHEMA.items()}


def extract_visual_facts(frame_paths: List[str]) -> List[Dict]:
    """对多帧调用 analyze_frame，返回列表形式 JSON 结果"""
    results: List[Dict] = []
    for frame_path in frame_paths:
        try:
            results.append(analyze_frame(frame_path))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to analyze frame %s: %s", frame_path, exc)
    return results
