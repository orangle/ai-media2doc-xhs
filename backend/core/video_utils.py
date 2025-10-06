from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import ffmpeg
import cv2
import numpy as np
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
    """按每秒 fps 抽帧，返回帧图片路径列表（兼容旧接口）"""
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


def detect_scenes(video_path: str) -> List[Dict[str, float]]:
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

    scene_ranges: List[Dict[str, float]] = []
    for start_time, end_time in scenes:
        scene_ranges.append(
            {
                "start": float(start_time.get_seconds()),
                "end": float(end_time.get_seconds()),
            }
        )

    if not scene_ranges:
        duration = float(duration_tc.get_seconds()) if duration_tc else 0.0
        scene_ranges.append({"start": 0.0, "end": duration})

    return scene_ranges


def extract_frames(video_path: str, fps: int = 1) -> List[Dict[str, object]]:
    """抽帧并返回包含帧信息的列表"""

    frame_paths = extract_keyframes(video_path, fps=fps)
    frames: List[Dict[str, object]] = []
    if not frame_paths:
        return frames

    interval = 1.0 / float(fps)
    for idx, path in enumerate(frame_paths):
        frames.append(
            {
                "frame_id": f"frame_{idx:05d}",
                "ts": round(idx * interval, 3),
                "path": path,
            }
        )
    return frames


def _compute_frame_metrics(image_path: str) -> Dict[str, float]:
    image = cv2.imread(image_path)
    if image is None:
        return {"clarity": 0.0, "entropy": 0.0, "edge_density": 0.0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(np.sum(hist)) or 1.0
    probs = hist / total
    entropy = float(-np.sum([p * math.log(p + 1e-12) for p in probs]))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    return {
        "clarity": clarity,
        "entropy": entropy,
        "edge_density": edge_density,
    }


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    min_v = min(values)
    max_v = max(values)
    if math.isclose(max_v, min_v):
        return [0.0 for _ in values]
    span = max_v - min_v
    return [(v - min_v) / span for v in values]


def select_keyframes(
    scenes: List[Dict[str, float]],
    frames: List[Dict[str, object]],
    k: int = 9,
    budget: int = 15,
) -> Dict[str, List[Dict[str, object]]]:
    """根据镜头和启发式 + 轻问答筛选关键帧"""

    if k <= 0:
        raise ValueError("k must be greater than 0")
    if budget <= 0:
        raise ValueError("budget must be greater than 0")

    if not frames:
        return {"chosen": [], "rejected": []}

    sorted_frames = [dict(frame) for frame in sorted(frames, key=lambda item: float(item.get("ts", 0.0)))]

    if not scenes:
        last_ts = float(sorted_frames[-1]["ts"])
        scenes = [{"start": 0.0, "end": last_ts}]

    # 预处理场景索引
    frame_to_scene: Dict[int | None, List[Dict[str, object]]] = defaultdict(list)
    scene_boundaries = sorted(scenes, key=lambda s: s["start"])
    scene_idx = 0
    for frame in sorted_frames:
        ts = float(frame.get("ts", 0.0))
        while scene_idx + 1 < len(scene_boundaries) and ts > scene_boundaries[scene_idx]["end"]:
            scene_idx += 1
        if (
            scene_idx < len(scene_boundaries)
            and scene_boundaries[scene_idx]["start"] <= ts <= scene_boundaries[scene_idx]["end"]
        ):
            frame["scene_index"] = scene_idx
            frame_to_scene[scene_idx].append(frame)
        else:
            frame["scene_index"] = None
            frame_to_scene[None].append(frame)

    # 计算启发式指标
    clarity_values: List[float] = []
    entropy_values: List[float] = []
    edge_values: List[float] = []
    for frame in sorted_frames:
        metrics = _compute_frame_metrics(frame["path"])
        frame["metrics"] = metrics
        clarity_values.append(metrics["clarity"])
        entropy_values.append(metrics["entropy"])
        edge_values.append(metrics["edge_density"])

    clarity_norm = _normalize(clarity_values)
    entropy_norm = _normalize(entropy_values)
    edge_norm = _normalize(edge_values)

    for frame, c, e, ed in zip(sorted_frames, clarity_norm, entropy_norm, edge_norm):
        frame["score"] = 0.5 * c + 0.3 * e + 0.2 * ed

    # 阶段 A：场景中位帧
    median_candidates: List[Dict[str, object]] = []
    for idx, scene in enumerate(scene_boundaries):
        frames_in_scene = frame_to_scene.get(idx, [])
        if not frames_in_scene:
            continue
        mid_ts = (scene["start"] + scene["end"]) / 2
        median_candidates.append(
            min(frames_in_scene, key=lambda f: abs(float(f.get("ts", 0.0)) - mid_ts))
        )

    # 阶段 A：启发式 Top-N
    max_candidates = min(len(sorted_frames), max(k, budget * 2))
    heuristic_sorted = sorted(sorted_frames, key=lambda item: item["score"], reverse=True)

    candidate_pool: List[Dict[str, object]] = []
    seen_ids = set()

    for frame in median_candidates:
        if frame["frame_id"] not in seen_ids:
            candidate_pool.append(frame)
            seen_ids.add(frame["frame_id"])

    for frame in heuristic_sorted:
        if frame["frame_id"] in seen_ids:
            continue
        candidate_pool.append(frame)
        seen_ids.add(frame["frame_id"])
        if len(candidate_pool) >= max_candidates:
            break

    # 阶段 B：轻问答筛选
    vlm_budget = min(budget, len(candidate_pool))
    candidate_pool = candidate_pool[:vlm_budget]

    try:
        from . import visual_extractor

        vlm_results = visual_extractor.light_rank([frame["path"] for frame in candidate_pool])
    except Exception as exc:  # noqa: BLE001
        vlm_results = []
        for frame in candidate_pool:
            frame.setdefault("debug", {})["vlm_error"] = str(exc)

    path_to_vlm = {item.get("path"): item for item in vlm_results}
    for frame in candidate_pool:
        frame["vlm"] = path_to_vlm.get(
            frame["path"],
            {
                "path": frame["path"],
                "has_landmark": None,
                "has_readable_text": None,
                "representativeness": None,
                "brief": "",
            },
        )

    for frame in candidate_pool:
        vlm = frame.get("vlm", {})
        vlm_score = 0.0
        representativeness = vlm.get("representativeness")
        if isinstance(representativeness, (int, float)):
            vlm_score += 0.5 * float(representativeness)
        if vlm.get("has_landmark") is True:
            vlm_score += 0.3
        if vlm.get("has_readable_text") is True:
            vlm_score += 0.2
        frame["final_score"] = 0.6 * frame.get("score", 0.0) + vlm_score

    ranked_candidates = sorted(candidate_pool, key=lambda item: item.get("final_score", 0.0), reverse=True)

    chosen: List[Dict[str, object]] = []
    used_scene = set()
    for frame in ranked_candidates:
        scene_idx = frame.get("scene_index")
        if scene_idx is not None and scene_idx not in used_scene:
            chosen.append(frame)
            used_scene.add(scene_idx)
        if len(chosen) >= k:
            break

    if len(chosen) < k:
        for frame in ranked_candidates:
            if frame in chosen:
                continue
            chosen.append(frame)
            if len(chosen) >= k:
                break

    chosen_ids = {frame["frame_id"] for frame in chosen}
    rejected: List[Dict[str, object]] = []
    for frame in ranked_candidates:
        if frame["frame_id"] in chosen_ids:
            continue
        rejected.append({**frame, "reason": "lower_score"})

    if len(candidate_pool) < len(sorted_frames):
        skipped = [frame for frame in sorted_frames if frame["frame_id"] not in {f["frame_id"] for f in candidate_pool}]
        for frame in skipped:
            rejected.append({**frame, "reason": "not_sent_to_vlm"})

    return {
        "chosen": chosen[:k],
        "rejected": rejected,
    }
