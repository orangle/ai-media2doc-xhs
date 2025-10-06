from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.core import asr, evidence, fact_extractor, post_writer, video_utils, visual_extractor  # noqa: E402  pylint: disable=wrong-import-position
from shared import iflow_api  # noqa: E402  pylint: disable=wrong-import-position
from tools import exporter  # noqa: E402  pylint: disable=wrong-import-position

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DEFAULT_VL_BUDGET = int(os.getenv("VL_FRAME_BUDGET", "15"))
FACT_FIELDS = ["地点", "费用", "交通", "时间", "玩法", "注意事项", "标签"]


def _get_env_value(name: str, default: str) -> str:
    return os.getenv(name, default)


def _render_config_panel() -> int:
    model_config = {
        "ASR": _get_env_value("IFLOW_MODEL_ASR", "qwen3-max"),
        "VISION": _get_env_value("IFLOW_MODEL_VISION", "qwen3-vl-plus"),
        "FACT": _get_env_value("IFLOW_MODEL_FACT", "qwen3-max"),
        "WRITER": _get_env_value("IFLOW_MODEL_WRITER", "qwen3-max"),
    }
    timeout_config = {
        "ASR": _get_env_value("ASR_TIMEOUT_S", str(asr.DEFAULT_TIMEOUT_S)),
        "VISION": _get_env_value("VISION_TIMEOUT_S", "45"),
        "FACT": _get_env_value("FACT_TIMEOUT_S", "60"),
        "WRITER": _get_env_value("WRITER_TIMEOUT_S", "60"),
    }
    runtime_cfg = iflow_api.get_runtime_config()

    st.markdown("### ⚙️ 当前模型与调用配置")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**模型**")
        for key, value in model_config.items():
            st.write(f"{key}: {value}")
    with cols[1]:
        st.markdown("**超时 (s)**")
        for key, value in timeout_config.items():
            st.write(f"{key}: {value}")
    with cols[2]:
        st.markdown("**执行**")
        st.write(f"线程上限: {runtime_cfg['max_workers']}")
        st.write(f"重试次数: {runtime_cfg['retries']}")
        st.write(f"API: {runtime_cfg['api_url']}")
        st.write(f"缓存目录: {runtime_cfg['cache_dir']}")

    with st.container():
        cols_actions = st.columns([1, 2])
        with cols_actions[0]:
            if st.button("清空缓存"):
                removed = iflow_api.clear_cache()
                st.success(f"已清空 {removed} 条缓存")
        with cols_actions[1]:
            vl_budget = st.slider(
                "VL 帧预算",
                min_value=5,
                max_value=60,
                value=st.session_state.get("vl_budget", DEFAULT_VL_BUDGET),
                help="控制送入视觉模型的帧数量上限，命中缓存后可适当调小以节省成本。",
            )

    st.session_state["vl_budget"] = vl_budget
    return vl_budget


def _save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="uploaded_video_"))
    output_path = temp_dir / f"video{suffix}"
    with open(output_path, "wb") as dest:
        dest.write(uploaded_file.getbuffer())
    return output_path


def _run_pipeline(video_path: Path, vl_budget: int) -> Optional[dict]:
    try:
        asr_result = asr.transcribe(str(video_path))
        scenes = video_utils.detect_scenes(str(video_path))
        frames_info = video_utils.extract_frames(str(video_path), fps=1)
        keyframe_selection = video_utils.select_keyframes(scenes, frames_info, k=9, budget=vl_budget)
        chosen_frames = keyframe_selection.get("chosen", [])
        chosen_paths = [frame["path"] for frame in chosen_frames]
        visual_raw = visual_extractor.extract_visual_facts(chosen_paths)
        visual_result: List[Dict] = []
        for path, raw in zip(chosen_paths, visual_raw):
            enriched = dict(raw)
            enriched["image_path"] = path
            visual_result.append(enriched)

        facts_raw = fact_extractor.extract_facts(asr_result, visual_result)
        evidences = evidence.build_evidences(asr_result, keyframe_selection, visual_result)
        facts_bundle = evidence.attach_facts(facts_raw, evidences)

        writer_payload = dict(facts_bundle.get("facts_strict", {}))
        writer_payload["missing"] = facts_bundle.get("missing", [])
        post = post_writer.generate_post(writer_payload)
        return {
            "facts": facts_bundle,
            "post": post,
            "frames": chosen_frames,
            "keyframe_selection": keyframe_selection,
            "evidences": evidences,
            "visual": visual_result,
            "asr": asr_result,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pipeline failed: %s", exc)
        st.error(f"处理失败：{exc}")
        return None


def _ensure_cover_state(frames: List[Dict]) -> None:
    paths = [frame.get("path") for frame in frames if frame.get("path")]
    cover = st.session_state.get("cover_frame")
    if cover not in paths:
        st.session_state["cover_frame"] = paths[0] if paths else None


def _split_paragraphs(markdown_text: str) -> List[str]:
    paragraphs = []
    for block in (markdown_text or "").split("\n\n"):
        block = block.strip()
        if block:
            paragraphs.append(block)
    return paragraphs


def _collect_related_facts(paragraph: str, facts_strict: Dict[str, object]) -> Dict[str, object]:
    related: Dict[str, object] = {}
    for field in FACT_FIELDS:
        if field == "玩法" or field == "注意事项" or field == "标签":
            values = facts_strict.get(field)
            if isinstance(values, list):
                hits = [item for item in values if isinstance(item, str) and item and item in paragraph]
                if hits:
                    related[field] = hits
        else:
            value = facts_strict.get(field)
            if isinstance(value, str) and value and value in paragraph:
                related[field] = value
    return related


def _render_evidence_items(evidence_ids: List[str], evidence_index: Dict[str, Dict]) -> None:
    unique_ids = []
    for eid in evidence_ids:
        if eid not in unique_ids:
            unique_ids.append(eid)

    if not unique_ids:
        st.write("未找到关联证据")
        return

    for eid in unique_ids:
        evidence_item = evidence_index.get(eid)
        if not evidence_item:
            continue
        if evidence_item.get("type") == "asr":
            st.caption(f"字幕 {evidence_item.get('start', 0)}s - {evidence_item.get('end', 0)}s")
            st.write(evidence_item.get("text") or "")
            if evidence_item.get("keywords"):
                st.code("关键词: " + "、".join(evidence_item["keywords"]))
        elif evidence_item.get("type") == "vision":
            path = evidence_item.get("path")
            if path:
                st.image(path, caption=f"帧 {evidence_item.get('frame_id', '')}")
            repr_score = evidence_item.get("representativeness")
            meta_parts = []
            if repr_score is not None:
                meta_parts.append(f"代表性 {float(repr_score):.2f}")
            if evidence_item.get("has_landmark") is True:
                meta_parts.append("含地标")
            if evidence_item.get("has_text") is True:
                meta_parts.append("含文字")
            if meta_parts:
                st.caption("，".join(meta_parts))
            for key in ("place", "visible_text", "activities", "objects"):
                value = evidence_item.get(key)
                if not value:
                    continue
                if isinstance(value, list):
                    st.write(f"{key}: " + "、".join(str(item) for item in value))
                else:
                    st.write(f"{key}: {value}")


def _render_fact_section(container, title: str, facts_data: Dict[str, object], evidence_index: Dict[str, Dict]) -> None:
    container.markdown(f"**{title}**")
    evidence_map = facts_data.get("evidence_ids", {}) if isinstance(facts_data, dict) else {}
    for field in FACT_FIELDS:
        value = facts_data.get(field) if isinstance(facts_data, dict) else None
        container.markdown(f"_{field}_")
        if isinstance(value, list) and value:
            for item in value:
                container.write(f"- {item}")
        elif isinstance(value, str) and value:
            container.write(value)
        else:
            container.write("（暂无信息）")
        with container.expander("🔎证据", expanded=False):
            evidence_ids = evidence_map.get(field, []) if isinstance(evidence_map, dict) else []
            _render_evidence_items(evidence_ids, evidence_index)


def _handle_rewrite(paragraph_index: int, paragraph_text: str, result: Dict) -> None:
    facts_bundle = result.get("facts", {})
    strict = dict(facts_bundle.get("facts_strict", {}))
    strict.pop("evidence_ids", None)
    payload = dict(strict)
    payload["missing"] = facts_bundle.get("missing", [])
    payload["_rewrite_request"] = {
        "paragraph_index": paragraph_index,
        "paragraph_text": paragraph_text,
        "related_facts": _collect_related_facts(paragraph_text, facts_bundle.get("facts_strict", {})),
        "original_paragraphs": _split_paragraphs(result.get("post", {}).get("markdown", "")),
        "title": result.get("post", {}).get("title"),
    }
    new_post = post_writer.generate_post(payload)
    result["post"] = new_post
    st.session_state["pipeline_result"] = result


st.set_page_config(page_title="AI-Media2Doc 小红书生成", layout="wide")
st.title("🎬 视频转小红书图文")
st.write("上传一段短视频，系统将自动识别语音、理解画面，并生成符合小红书风格的图文内容。")

vl_budget = _render_config_panel()

if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None

uploaded_video = st.file_uploader("上传视频文件", type=["mp4", "mov", "mkv", "avi"])

if uploaded_video is not None:
    video_path = _save_uploaded_file(uploaded_video)
    st.session_state["uploaded_video_path"] = str(video_path)
    st.video(str(video_path))

if st.button("生成图文", type="primary"):
    video_path_str = st.session_state.get("uploaded_video_path")
    if not video_path_str:
        st.warning("请先上传视频文件。")
    else:
        with st.spinner("正在分析视频，请稍候..."):
            result = _run_pipeline(Path(video_path_str), vl_budget)
            st.session_state["pipeline_result"] = result
            st.session_state.pop("rewrite_feedback", None)
        if result:
            st.success("生成完成！")

result = st.session_state.get("pipeline_result")
if result:
    feedback = st.session_state.pop("rewrite_feedback", None)
    if feedback:
        st.success(feedback)

    post = result.get("post") or {}
    facts_bundle = result.get("facts") or {}
    evidences = result.get("evidences") or []
    frames = result.get("frames") or []

    _ensure_cover_state(frames)
    cover_path = st.session_state.get("cover_frame")
    evidence_index = {item.get("id"): item for item in evidences if isinstance(item, dict)}

    cols = st.columns([1.1, 1.4, 1.1])

    with cols[0]:
        st.subheader("关键帧九宫格")
        for row_start in range(0, len(frames), 3):
            row_cols = st.columns(3)
            for offset, col in enumerate(row_cols):
                idx = row_start + offset
                if idx >= len(frames):
                    continue
                frame = frames[idx]
                frame_path = frame.get("path")
                caption_parts = [frame.get("frame_id", f"frame_{idx:02d}")]
                score = frame.get("final_score")
                if isinstance(score, (int, float)):
                    caption_parts.append(f"评分 {float(score):.2f}")
                brief = frame.get("vlm", {}).get("brief") if isinstance(frame.get("vlm"), dict) else None
                if brief:
                    caption_parts.append(brief)
                with col:
                    st.image(frame_path, caption=" | ".join(part for part in caption_parts if part), use_column_width=True)
                    if frame_path == cover_path:
                        st.caption("✅ 当前封面")
                    if st.button("设为封面", key=f"cover_{frame.get('frame_id', idx)}"):
                        st.session_state["cover_frame"] = frame_path
                        st.success("已更新封面")
        if not frames:
            st.info("暂无关键帧，请重新生成。")

    with cols[1]:
        st.subheader(post.get("title", "生成结果"))
        markdown_text = post.get("markdown", "")
        if markdown_text:
            st.markdown(markdown_text)
        paragraphs = _split_paragraphs(markdown_text)
        for idx, paragraph in enumerate(paragraphs):
            st.markdown("---")
            st.markdown(paragraph)
            if st.button("仅重写本段", key=f"rewrite_{idx}"):
                try:
                    _handle_rewrite(idx, paragraph, result)
                    st.session_state["rewrite_feedback"] = f"已重写第 {idx + 1} 段"
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Paragraph rewrite failed: %s", exc)
                    st.session_state["rewrite_feedback"] = f"重写失败：{exc}"
                st.experimental_rerun()

    with cols[2]:
        st.subheader("结构化事实")
        st.json(facts_bundle, expanded=False)
        strict = facts_bundle.get("facts_strict", {})
        weak = facts_bundle.get("facts_weak", {})
        missing = facts_bundle.get("missing", [])
        _render_fact_section(st, "严格可信", strict, evidence_index)
        st.markdown("---")
        _render_fact_section(st, "弱证据（待确认）", weak, evidence_index)
        if missing:
            st.warning("缺失字段：" + "、".join(missing))

    st.markdown("---")

    export_col1, export_col2 = st.columns([1, 1])
    with export_col1:
        st.markdown("**导出结果**")
    with export_col2:
        if st.button("导出"):
            try:
                temp_dir = Path(tempfile.mkdtemp(prefix="bundle_"))
                video_path_str = st.session_state.get("uploaded_video_path", "")
                video_name = Path(video_path_str).stem if video_path_str else "export"
                post_payload = dict(post)
                post_payload.setdefault("video_name", video_name)
                post_payload["cover_path"] = cover_path
                zip_path = exporter.export_bundle(temp_dir, post_payload, facts_bundle, frames)
                with open(zip_path, "rb") as fp:
                    st.download_button("下载导出包", data=fp.read(), file_name=Path(zip_path).name, mime="application/zip")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Export failed: %s", exc)
                st.error(f"导出失败：{exc}")
