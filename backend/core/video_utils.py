from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import ffmpeg
from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector


def _ensure_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def extract_audio(video_path: str) -> str:
    """提取音频并返回音频文件路径"""
    input_path = _ensure_path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {input_path}")

    output_dir = input_path.parent
    output_path = output_dir / f"{input_path.stem}_audio.wav"

    stream = ffmpeg.input(str(input_path))
    stream = ffmpeg.output(
        stream.audio,
        str(output_path),
        acodec="pcm_s16le",
        ac=1,
        ar=16000,
    )
    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

    return str(output_path)


def extract_keyframes(video_path: str, fps: int = 1) -> List[str]:
    """按每秒 fps 抽帧，返回帧图片路径列表"""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    input_path = _ensure_path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {input_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{input_path.stem}_frames_"))
    output_template = temp_dir / "frame_%05d.jpg"

    stream = ffmpeg.input(str(input_path))
    stream = ffmpeg.filter(stream, "fps", fps=fps)
    stream = ffmpeg.output(stream, str(output_template), vsync="vfr")
    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

    frame_paths = sorted(str(path) for path in temp_dir.glob("frame_*.jpg"))
    return frame_paths


def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    """使用 PySceneDetect 切分镜头，返回每个镜头的起止时间"""
    input_path = _ensure_path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {input_path}")

    video_manager = VideoManager([str(input_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    duration_tc = None
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scenes = scene_manager.get_scene_list()
        duration_tc = video_manager.get_duration()
    finally:
        video_manager.release()

    scene_ranges: List[Tuple[float, float]] = []
    for start_time, end_time in scenes:
        scene_ranges.append((start_time.get_seconds(), end_time.get_seconds()))

    if not scene_ranges:
        duration = float(duration_tc.get_seconds()) if duration_tc else 0.0
        scene_ranges.append((0.0, duration))

    return scene_ranges
