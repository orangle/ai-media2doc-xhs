from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import requests

LOGGER = logging.getLogger(__name__)

IFLOW_API_URL = os.getenv("IFLOW_API_URL", "https://api.iflow.cn/v1/chat/completions")
IFLOW_MODEL_FACT = os.getenv("IFLOW_MODEL_FACT", "qwen3-max")

FACT_SCHEMA_PROMPT = """
You are an information extraction assistant for travel videos. Given ASR transcripts and visual recognitions, produce a JSON object with the following keys in Chinese:
- 地点: string
- 费用: string (include currency if present, otherwise use "未知")
- 玩法: list of strings
- 交通: string
- 时间: string
- 注意事项: list of strings
- 标签: list of strings
Ensure every field exists. If information is missing, fill with "未知" or an empty list as appropriate. Respond with valid JSON only.
"""


def _call_iflow(asr_data: List[Dict], visual_data: List[Dict]) -> Dict:
    api_key = os.getenv("IFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("IFLOW_API_KEY is not set.")

    user_content = {
        "type": "input_text",
        "text": json.dumps({"asr": asr_data, "visual": visual_data}, ensure_ascii=False),
    }

    payload = {
        "model": IFLOW_MODEL_FACT,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": FACT_SCHEMA_PROMPT.strip()},
            {"role": "user", "content": [user_content]},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(IFLOW_API_URL, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    text = text.strip()
    if not text:
        return {
            "地点": "未知",
            "费用": "未知",
            "玩法": [],
            "交通": "未知",
            "时间": "未知",
            "注意事项": [],
            "标签": [],
        }

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        LOGGER.error("Failed to parse facts JSON: %s", exc)
        return {
            "地点": "未知",
            "费用": "未知",
            "玩法": [],
            "交通": "未知",
            "时间": "未知",
            "注意事项": [],
            "标签": [],
        }

    parsed.setdefault("地点", "未知")
    parsed.setdefault("费用", "未知")
    parsed.setdefault("玩法", [])
    parsed.setdefault("交通", "未知")
    parsed.setdefault("时间", "未知")
    parsed.setdefault("注意事项", [])
    parsed.setdefault("标签", [])
    return parsed


def extract_facts(asr_data: List[Dict], visual_data: List[Dict]) -> Dict:
    """输入：语音识别结果 & 视觉理解结果，输出结构化 facts JSON"""
    return _call_iflow(asr_data, visual_data)
