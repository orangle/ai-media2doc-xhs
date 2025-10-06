from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


@dataclass
class Evidence:
    """Lightweight evidence record that can link facts back to raw signals."""

    id: str
    type: str  # "asr" | "vision"
    confidence: float
    payload: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        data = {"id": self.id, "type": self.type, "confidence": round(self.confidence, 3)}
        data.update(self.payload)
        return data


_ASR_KEYWORDS: Sequence[str] = (
    "元",
    "门票",
    "地铁",
    "公交",
    "站",
    "酒店",
    "美食",
    "餐",
    "开放",
    "时间",
)


def _score_asr_segment(text: str) -> float:
    clean_text = text.strip()
    if not clean_text:
        return 0.0

    length_score = min(1.0, len(clean_text) / 40.0)
    keyword_hits = sum(1 for keyword in _ASR_KEYWORDS if keyword in clean_text)
    keyword_score = min(1.0, keyword_hits / 4.0)

    # Baseline 0.35 so short factual statements are not discarded entirely.
    return round(min(1.0, 0.35 + 0.45 * length_score + 0.2 * keyword_score), 3)


def _score_vision_segment(representativeness: float | None, has_landmark: bool | None, has_text: bool | None) -> float:
    rep = float(representativeness) if representativeness is not None else 0.0
    rep = max(0.0, min(rep, 1.0))
    score = 0.55 * rep
    if has_landmark:
        score += 0.25
    if has_text:
        score += 0.2
    return round(min(score, 1.0), 3)


def build_evidences(asr_segments: Sequence[Dict], keyframe_selection: Dict, vision_results: Sequence[Dict]) -> List[Dict[str, object]]:
    """Construct a list of evidence dicts that the UI can use for provenance."""

    evidences: List[Evidence] = []

    for idx, segment in enumerate(asr_segments, start=1):
        text = str(segment.get("text") or "").strip()
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        confidence = _score_asr_segment(text)
        keywords = [kw for kw in _ASR_KEYWORDS if kw in text]
        evidences.append(
            Evidence(
                id=f"asr_{idx:04d}_{_hash_text(text) if text else idx}",
                type="asr",
                confidence=confidence,
                payload={
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "keywords": keywords,
                },
            )
        )

    chosen_frames = (keyframe_selection or {}).get("chosen", [])
    path_to_frame = {frame.get("path"): frame for frame in chosen_frames}

    for idx, vision_entry in enumerate(vision_results, start=1):
        path = vision_entry.get("image_path")
        frame_info = path_to_frame.get(path, {})
        vlm_info = frame_info.get("vlm", {}) if isinstance(frame_info, dict) else {}
        representativeness = vlm_info.get("representativeness")
        has_landmark = vlm_info.get("has_landmark")
        has_text = vlm_info.get("has_readable_text")

        confidence = _score_vision_segment(representativeness, has_landmark, has_text)

        evidences.append(
            Evidence(
                id=f"vision_{idx:04d}_{_hash_text(str(path)) if path else idx}",
                type="vision",
                confidence=confidence,
                payload={
                    "frame_id": frame_info.get("frame_id"),
                    "path": path,
                    "representativeness": representativeness,
                    "has_landmark": bool(has_landmark) if has_landmark is not None else None,
                    "has_text": bool(has_text) if has_text is not None else None,
                    "place": vision_entry.get("place"),
                    "visible_text": vision_entry.get("visible_text"),
                    "activities": vision_entry.get("activities"),
                    "objects": vision_entry.get("objects"),
                    "mood": vision_entry.get("mood"),
                },
            )
        )

    return [evidence.as_dict() for evidence in evidences]


def _iter_evidence_texts(evidence: Dict[str, object]) -> Iterable[str]:
    if evidence.get("type") == "asr":
        text = evidence.get("text")
        if isinstance(text, str) and text.strip():
            yield text
    elif evidence.get("type") == "vision":
        for key in ("place", "visible_text", "mood"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                yield value
        for key in ("activities", "objects"):
            value = evidence.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        yield item


def _support_score(value: str, evidences: Sequence[Dict[str, object]]) -> tuple[float, List[str]]:
    value = str(value or "").strip()
    if not value:
        return 0.0, []

    matched_ids: List[str] = []
    confidence = 0.0
    for evidence in evidences:
        for text in _iter_evidence_texts(evidence):
            if value in text:
                matched_ids.append(str(evidence.get("id")))
                confidence = max(confidence, float(evidence.get("confidence") or 0.0))
                break
    return confidence, matched_ids


def _attach_list_items(values: Sequence[str], evidences: Sequence[Dict[str, object]]) -> tuple[List[str], List[str], Dict[str, List[str]]]:
    strict_items: List[str] = []
    weak_items: List[str] = []
    evidence_map: Dict[str, List[str]] = {}

    for item in values:
        score, evidence_ids = _support_score(item, evidences)
        if score >= 0.75:
            strict_items.append(item)
            evidence_map[item] = evidence_ids
        elif score >= 0.5:
            weak_items.append(item)
            evidence_map[item] = evidence_ids
    return strict_items, weak_items, evidence_map


def attach_facts(facts: Dict, evidences: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Split facts into strict/weak sets and attach supporting evidence IDs."""

    strict: Dict[str, object] = {"evidence_ids": {}}
    weak: Dict[str, object] = {"evidence_ids": {}}
    missing: set[str] = set()

    source_missing = facts.get("missing") or []
    for field in source_missing:
        if isinstance(field, str) and field.strip():
            missing.add(field)

    for field in ("地点", "费用", "交通", "时间"):
        value = facts.get(field)
        score, evidence_ids = _support_score(value, evidences)
        if score >= 0.75:
            strict[field] = value
            strict.setdefault("evidence_ids", {})[field] = evidence_ids
        elif score >= 0.5:
            weak[field] = value
            weak.setdefault("evidence_ids", {})[field] = evidence_ids
        else:
            missing.add(field)

    for field in ("玩法", "注意事项", "标签"):
        values = facts.get(field) or []
        if not isinstance(values, list):
            values = []
        strict_items, weak_items, evidence_map = _attach_list_items(values, evidences)
        if strict_items:
            strict[field] = strict_items
            strict.setdefault("evidence_ids", {})[field] = [eid for item in strict_items for eid in evidence_map.get(item, [])]
        if weak_items:
            weak[field] = weak_items
            weak.setdefault("evidence_ids", {})[field] = [eid for item in weak_items for eid in evidence_map.get(item, [])]
        if not strict_items and not weak_items:
            missing.add(field)

    return {
        "facts_strict": strict,
        "facts_weak": weak,
        "missing": sorted(missing),
    }

