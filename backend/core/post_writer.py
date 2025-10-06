from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from shared import iflow_api

LOGGER = logging.getLogger(__name__)

IFLOW_MODEL_WRITER = os.getenv("IFLOW_MODEL_WRITER", "qwen3-max")


def _timeout_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid timeout for %s=%s, using default %.1f", name, value, default)
        return default


WRITER_TIMEOUT_S = _timeout_from_env("WRITER_TIMEOUT_S", 60.0)

FACT_FIELDS: List[str] = ["地点", "费用", "玩法", "交通", "时间", "注意事项", "标签"]

WRITER_PROMPT = """
You are a Xiaohongshu travel copywriter. Use ONLY confirmed facts provided by the user.
Produce strict JSON with keys: title (string) and markdown (string). You may also include paragraphs (array of strings) when a rewrite_request is supplied.
Markdown requirements:
- Title plus five paragraphs with emojis integrated naturally.
- Reference only facts explicitly marked as high confidence.
- For each missing item, add a separate line starting with "去之前先确认：" followed by the field name.
- Do not introduce new places, prices, or activities beyond the facts.
- Respect the requested style and length guidelines.

If rewrite_request is present in the user payload:
- Keep the title unless an updated title is explicitly requested.
- Regenerate only the paragraph at rewrite_request.paragraph_index using the provided related_facts, while copying other paragraphs from rewrite_request.original_paragraphs verbatim.
- Return the updated full markdown and include an array named paragraphs listing the five final paragraphs in order.
"""

DEFAULT_STYLE = os.getenv("POST_WRITER_STYLE", "攻略")
DEFAULT_LENGTH = os.getenv("POST_WRITER_LENGTH", "中等长度")


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise RuntimeError("Post writer returned empty response.")

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise RuntimeError(f"Post writer returned non-JSON response: {text}")


def generate_post(facts: Dict) -> Dict:
    """
    输入：facts JSON
    输出：包含 title 和 markdown 正文的字典，符合小红书风格（Emoji、分段、口语化）
    """
    style = os.getenv("POST_WRITER_STYLE", DEFAULT_STYLE)
    length = os.getenv("POST_WRITER_LENGTH", DEFAULT_LENGTH)

    missing_fields = set()
    if isinstance(facts.get("missing"), list):
        missing_fields = {str(item) for item in facts["missing"] if str(item).strip()}

    rewrite_request = facts.get("_rewrite_request") if isinstance(facts, dict) else None

    safe_facts = {
        key: value
        for key, value in facts.items()
        if key not in {"missing", "evidence_ids", "_rewrite_request"}
    }

    data = iflow_api.chat_completion(
        IFLOW_MODEL_WRITER,
        [
            {"role": "system", "content": WRITER_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {
                                "facts": safe_facts,
                                "high_confidence_fields": [
                                    key for key in FACT_FIELDS if key in safe_facts and key not in missing_fields
                                ],
                                "missing_fields": sorted(missing_fields),
                                "style": style,
                                "length": length,
                                "rewrite_request": rewrite_request,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            },
        ],
        timeout_s=WRITER_TIMEOUT_S,
        temperature=0.6,
    )
    content = data["choices"][0]["message"]["content"]

    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    try:
        result = _extract_json_object(text)
        if rewrite_request and isinstance(result, dict):
            original_paragraphs = rewrite_request.get("original_paragraphs") or []
            new_paragraphs = result.get("paragraphs")
            if isinstance(new_paragraphs, list) and len(new_paragraphs) == len(original_paragraphs):
                joined = "\n\n".join(str(p) for p in new_paragraphs)
                result["markdown"] = joined
        return result
    except RuntimeError as exc:  # noqa: BLE001
        LOGGER.error("Failed to parse writer output: %s", exc)
        raise
