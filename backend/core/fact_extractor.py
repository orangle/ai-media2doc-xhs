from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Tuple

from shared import iflow_api

LOGGER = logging.getLogger(__name__)

IFLOW_MODEL_FACT = os.getenv("IFLOW_MODEL_FACT", "qwen3-max")


def _timeout_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid timeout for %s=%s, using default %.1f", name, value, default)
        return default


FACT_TIMEOUT_S = _timeout_from_env("FACT_TIMEOUT_S", 60.0)

FACT_SCHEMA_PROMPT = """
You are an extraction assistant for travel vlogs. Use ONLY the provided ASR transcript and visual JSON evidence.
Return strict JSON (no extra text) with keys: 地点 (string|null), 费用 (string|null), 玩法 (array of strings), 交通 (string|null), 时间 (string|null), 注意事项 (array of strings), 标签 (array of strings), missing (array of field names that lack confirmed information).
Rules:
1. Do not guess cities, prices, or facts that are not explicitly present in the evidence.
2. Only output numbers or entities that literally occur in the ASR text or visible_text.
3. If uncertain, use null (or [] for arrays) and include that field name in missing.
4. Keep wording concise and quote the original text segments when possible.
"""

FACT_DEFAULT = {
    "地点": None,
    "费用": None,
    "玩法": [],
    "交通": None,
    "时间": None,
    "注意事项": [],
    "标签": [],
    "missing": [],
}

PRICE_PATTERNS = [
    re.compile(r"[¥￥]\s*\d+(?:\.\d+)?"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:元|块|人民币|RMB)"),
]

TIME_PATTERNS = [
    re.compile(r"\d{1,2}[:：]\d{2}"),
    re.compile(r"\d{1,2}点(?:\d{1,2}分)?"),
    re.compile(r"\d{1,2}月\d{1,2}日"),
]


def _extract_candidates_from_visible_text(visual_data: List[Dict]) -> Tuple[List[str], List[str]]:
    prices: List[str] = []
    times: List[str] = []
    for entry in visual_data:
        text = entry.get("visible_text") or ""
        if not isinstance(text, str):
            continue
        for pattern in PRICE_PATTERNS:
            for match in pattern.findall(text):
                normalized = match.strip()
                if normalized not in prices:
                    prices.append(normalized)
        for pattern in TIME_PATTERNS:
            for match in pattern.findall(text):
                normalized = match.strip()
                if normalized not in times:
                    times.append(normalized)
    return prices, times


def _flatten_text_sources(asr_data: List[Dict], visual_data: List[Dict]) -> str:
    parts: List[str] = []
    for segment in asr_data:
        part = str(segment.get("text", "")).strip()
        if part:
            parts.append(part)
    for entry in visual_data:
        for key in ("place", "mood", "visible_text"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        for key in ("activities", "objects"):
            values = entry.get(key) or []
            if isinstance(values, list):
                parts.extend(str(item).strip() for item in values if str(item).strip())
    return "\n".join(parts)


def _extract_json(text: str) -> Dict:
    text = text.strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for idx in range(len(text)):
        if text[idx] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    LOGGER.error("Failed to extract JSON from response: %s", text)
    return {}


def _normalize_fact_output(parsed: Dict, allowed_text: str, price_candidates: List[str], time_candidates: List[str]) -> Dict:
    result = {key: (value.copy() if isinstance(value, list) else value) for key, value in FACT_DEFAULT.items()}
    missing_fields = set(parsed.get("missing", []))

    def _normalize_string(field: str, value):
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
        else:
            cleaned = str(value).strip()
        if not cleaned:
            return None
        return cleaned

    for field in ("地点", "费用", "交通", "时间"):
        normalized = _normalize_string(field, parsed.get(field))
        if normalized and normalized in allowed_text:
            result[field] = normalized
            missing_fields.discard(field)
        elif normalized:
            LOGGER.debug("Discarding hallucinated value for %s: %s", field, normalized)
            result[field] = None
            missing_fields.add(field)
        else:
            result[field] = None
            missing_fields.add(field)

    for field in ("玩法", "注意事项", "标签"):
        values = parsed.get(field)
        normalized_list: List[str] = []
        if isinstance(values, list):
            for item in values:
                normalized = _normalize_string(field, item)
                if normalized and normalized in allowed_text:
                    normalized_list.append(normalized)
                elif normalized:
                    LOGGER.debug("Discarding hallucinated list item for %s: %s", field, normalized)
        result[field] = normalized_list
        if not normalized_list:
            missing_fields.add(field)
        else:
            missing_fields.discard(field)

    if not result["费用"] and price_candidates:
        result["费用"] = price_candidates[0]
        missing_fields.discard("费用")

    if not result["时间"] and time_candidates:
        result["时间"] = time_candidates[0]
        missing_fields.discard("时间")

    if not result["地点"]:
        missing_fields.add("地点")
    if not result["交通"]:
        missing_fields.add("交通")

    result["missing"] = sorted(set(missing_fields))
    return result


def _call_iflow(asr_data: List[Dict], visual_data: List[Dict]) -> Dict:
    price_candidates, time_candidates = _extract_candidates_from_visible_text(visual_data)
    allowed_text = _flatten_text_sources(asr_data, visual_data)

    user_content = {
        "type": "input_text",
        "text": json.dumps(
            {
                "asr": asr_data,
                "visual": visual_data,
                "visible_price_candidates": price_candidates,
                "visible_time_candidates": time_candidates,
            },
            ensure_ascii=False,
        ),
    }

    data = iflow_api.chat_completion(
        IFLOW_MODEL_FACT,
        [
            {"role": "system", "content": FACT_SCHEMA_PROMPT.strip()},
            {"role": "user", "content": [user_content]},
        ],
        timeout_s=FACT_TIMEOUT_S,
        temperature=0.2,
    )
    content = data["choices"][0]["message"]["content"]

    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    parsed = _extract_json(text)
    if not parsed:
        LOGGER.error("Fact extractor returned empty or invalid JSON.")
        return {key: (value.copy() if isinstance(value, list) else value) for key, value in FACT_DEFAULT.items()}

    return _normalize_fact_output(parsed, allowed_text, price_candidates, time_candidates)


def extract_facts(asr_data: List[Dict], visual_data: List[Dict]) -> Dict:
    """输入：语音识别结果 & 视觉理解结果，输出结构化 facts JSON"""
    return _call_iflow(asr_data, visual_data)
