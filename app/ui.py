from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.core import asr, fact_extractor, post_writer, video_utils, visual_extractor  # noqa: E402  pylint: disable=wrong-import-position

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="uploaded_video_"))
    output_path = temp_dir / f"video{suffix}"
    with open(output_path, "wb") as dest:
        dest.write(uploaded_file.getbuffer())
    return output_path


def _run_pipeline(video_path: Path) -> Optional[dict]:
    try:
        asr_result = asr.transcribe(str(video_path))
        scenes = video_utils.detect_scenes(str(video_path))
        frames_info = video_utils.extract_frames(str(video_path), fps=1)
        keyframe_selection = video_utils.select_keyframes(scenes, frames_info, k=9, budget=15)
        chosen_frames = keyframe_selection.get("chosen", [])
        chosen_paths = [frame["path"] for frame in chosen_frames]
        visual_result = visual_extractor.extract_visual_facts(chosen_paths)
        facts = fact_extractor.extract_facts(asr_result, visual_result)
        post = post_writer.generate_post(facts)
        return {
            "facts": facts,
            "post": post,
            "frames": chosen_paths,
            "keyframe_selection": keyframe_selection,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pipeline failed: %s", exc)
        st.error(f"处理失败：{exc}")
        return None


st.set_page_config(page_title="AI-Media2Doc 小红书生成", layout="wide")
st.title("🎬 视频转小红书图文")
st.write("上传一段短视频，系统将自动识别语音、理解画面，并生成符合小红书风格的图文内容。")

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
            result = _run_pipeline(Path(video_path_str))
            st.session_state["pipeline_result"] = result
        if result:
            st.success("生成完成！")

result = st.session_state.get("pipeline_result")
if result:
    post = result.get("post")
    facts = result.get("facts")

    if post:
        st.subheader(post.get("title", "生成结果"))
        st.markdown(post.get("markdown", ""))

    if facts:
        st.subheader("结构化信息")
        st.json(facts, expanded=False)

    with st.expander("查看抽帧结果", expanded=False):
        frames = result.get("keyframe_selection", {}).get("chosen", [])
        if frames:
            cols = st.columns(3)
            for idx, frame in enumerate(frames):
                frame_path = frame.get("path")
                caption_parts = [os.path.basename(frame_path or "frame"), f"score {frame.get('final_score', 0):.2f}"]
                brief = frame.get("vlm", {}).get("brief")
                if brief:
                    caption_parts.append(brief)
                with cols[idx % 3]:
                    st.image(frame_path, caption=" | ".join(part for part in caption_parts if part))
        else:
            st.write("未生成帧图像。")
