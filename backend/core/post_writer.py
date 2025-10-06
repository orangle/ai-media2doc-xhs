from __future__ import annotations

import json
import logging
import os
from typing import Dict

import requests

LOGGER = logging.getLogger(__name__)

IFLOW_API_URL = os.getenv("IFLOW_API_URL", "https://api.iflow.cn/v1/chat/completions")
IFLOW_MODEL_WRITER = os.getenv("IFLOW_MODEL_WRITER", "qwen3-max")

WRITER_PROMPT = """
You are an expert travel copywriter crafting Xiaohongshu (Little Red Book) style posts. Given structured travel facts, write a compelling title and markdown body using emoji, conversational tone, and factual accuracy. Do not invent information beyond the provided facts.
Return JSON with keys: title (string) and markdown (string).
"""


def generate_post(facts: Dict) -> Dict:
    """
    输入：facts JSON
    输出：包含 title 和 markdown 正文的字典，符合小红书风格（Emoji、分段、口语化）
    """
    api_key = os.getenv("IFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("IFLOW_API_KEY is not set.")

    payload = {
        "model": IFLOW_MODEL_WRITER,
        "temperature": 0.6,
        "messages": [
            {"role": "system", "content": WRITER_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(facts, ensure_ascii=False),
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
    content = data["choices"][0]["message"]["content"]

    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    text = text.strip()
    if not text:
        raise RuntimeError("Post writer returned empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        LOGGER.error("Failed to parse writer output: %s", exc)
        raise
