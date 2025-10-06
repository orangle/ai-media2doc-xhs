from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


def _slugify(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in text)
    sanitized = sanitized.strip("_")
    return sanitized or "export"


def _iter_frame_paths(chosen_frames: Iterable[Dict]) -> Iterable[str]:
    for frame in chosen_frames:
        if not isinstance(frame, dict):
            continue
        path = frame.get("path")
        if not path:
            continue
        yield path


def export_bundle(output_dir: str | Path, post: Dict, facts: Dict, chosen_frames: List[Dict]) -> str:
    """Bundle markdown, facts, and selected frames into a portable zip package."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = post.get("video_name") if isinstance(post, dict) else "export"
    if not isinstance(video_name, str):
        video_name = "export"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{_slugify(video_name)}_{timestamp}"

    bundle_dir = output_dir / base_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    title = post.get("title") if isinstance(post, dict) else None
    markdown = post.get("markdown") if isinstance(post, dict) else None
    if not isinstance(markdown, str):
        markdown = ""

    post_lines: List[str] = []
    if isinstance(title, str) and title.strip():
        post_lines.append(f"# {title.strip()}")
    if markdown.strip():
        post_lines.append(markdown.strip())

    post_path = bundle_dir / "post.md"
    with open(post_path, "w", encoding="utf-8") as fp:
        fp.write("\n\n".join(post_lines).strip() + "\n")

    facts_path = bundle_dir / "facts.json"
    with open(facts_path, "w", encoding="utf-8") as fp:
        json.dump(facts, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    images_dir = bundle_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cover_path = post.get("cover_path") if isinstance(post, dict) else None
    copied_files: List[Path] = []
    for idx, path in enumerate(_iter_frame_paths(chosen_frames)):
        src = Path(path)
        if not src.exists():
            continue
        prefix = "00_cover" if cover_path and Path(cover_path) == src else f"{idx + 1:02d}"
        dest_name = f"{prefix}_{src.name}" if prefix else src.name
        dest = images_dir / dest_name
        shutil.copy2(src, dest)
        copied_files.append(dest)

    images_zip_path = bundle_dir / "images.zip"
    with zipfile.ZipFile(images_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in copied_files:
            zipf.write(file_path, arcname=file_path.name)

    final_zip_path = output_dir / f"{base_name}.zip"
    with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(post_path, arcname="post.md")
        zipf.write(facts_path, arcname="facts.json")
        zipf.write(images_zip_path, arcname="images.zip")

    shutil.rmtree(bundle_dir, ignore_errors=True)

    return str(final_zip_path)

