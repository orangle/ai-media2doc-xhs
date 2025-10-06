from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import requests

LOGGER = logging.getLogger(__name__)

IFLOW_API_URL = os.getenv("IFLOW_API_URL", "https://api.iflow.cn/v1/chat/completions")
IFLOW_MODEL_VISION = os.getenv("IFLOW_MODEL_VISION", "qwen3-vl-plus")


def _read_image_as_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    mime = "image/jpeg"
    if path.suffix.lower() in {".png"}:
        mime = "image/png"

    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _call_iflow(messages: List[Dict]) -> Dict:
    api_key = os.getenv("IFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("IFLOW_API_KEY is not set.")

    payload = {
        "model": IFLOW_MODEL_VISION,
        "messages": messages,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(IFLOW_API_URL, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_content_to_dict(content) -> Dict:
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        LOGGER.debug("Vision model returned non-JSON content, wrapping in fallback format: %s", text)
        return {"description": text}


def analyze_frame(image_path: str) -> Dict:
    """调用 iFlow Qwen3-VL-Plus 模型，识别图片中的地点、活动、物体、情绪、可读文字，返回 JSON 字典"""
    image_data_url = _read_image_as_data_url(image_path)

    messages = [
        {
            "role": "system",
            "content": "You analyze travel vlogs frame-by-frame. Return a compact JSON description with keys: location, activities, objects, mood, text.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Please analyze this frame and extract location, activities, objects, mood/emotion, and any readable text. Respond in JSON with lowercase English keys.",
                },
                {
                    "type": "input_image",
                    "image_url": image_data_url,
                },
            ],
        },
    ]

    content = _call_iflow(messages)
    return _parse_content_to_dict(content)


def extract_visual_facts(frame_paths: List[str]) -> List[Dict]:
    """对多帧调用 analyze_frame，返回列表形式 JSON 结果"""
    results: List[Dict] = []
    for frame_path in frame_paths:
        try:
            results.append(analyze_frame(frame_path))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to analyze frame %s: %s", frame_path, exc)
    return results
