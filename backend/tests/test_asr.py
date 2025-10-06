import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest


class _DummyFFmpegStream:
    def output(self, *args, **kwargs):  # noqa: D401 - simple stub
        return self

    def overwrite_output(self):
        return self

    def run(self, quiet=True):
        return None


def _dummy_probe(path):  # noqa: D401 - simple stub
    return {"format": {"duration": "1.0"}}


class _DummyNumpy:
    float32 = None

    def sum(self, *args, **kwargs):
        return 0.0

    def mean(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):  # noqa: D401 - simple stub
        return lambda *args, **kwargs: 0.0


sys.modules.setdefault(
    "ffmpeg",
    types.SimpleNamespace(
        probe=_dummy_probe,
        input=lambda *args, **kwargs: _DummyFFmpegStream(),
    ),
)
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("numpy", _DummyNumpy())
sys.modules.setdefault(
    "scenedetect",
    types.SimpleNamespace(
        SceneManager=lambda *args, **kwargs: types.SimpleNamespace(
            add_detector=lambda *a, **k: None,
            detect_scenes=lambda *a, **k: None,
            get_scene_list=lambda: [],
        ),
        VideoManager=lambda *args, **kwargs: types.SimpleNamespace(
            start=lambda: None,
            release=lambda: None,
            get_duration=lambda: types.SimpleNamespace(get_seconds=lambda: 0.0),
        ),
    ),
)
sys.modules.setdefault(
    "scenedetect.detectors",
    types.SimpleNamespace(ContentDetector=lambda *args, **kwargs: object()),
)

from backend.core import asr


def test_post_process_text_normalizes_punctuation():
    raw = " 你好 , 世界!! 123  "
    processed = asr._post_process_text(raw)
    assert processed == "你好，世界！123。"


def test_transcribe_prefers_whisper(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(asr, "extract_audio", lambda _: str(audio_path))
    monkeypatch.setenv("WHISPER_ENABLE", "true")
    monkeypatch.setenv("ASR_TIMEOUT_S", "5")
    monkeypatch.setenv("ASR_SEGMENT_S", "40")

    fake_module = object()
    monkeypatch.setattr(asr, "_load_whisper", lambda: fake_module)

    captured = {}

    def fake_whisper(audio, module, segment_length):
        captured["audio"] = audio
        captured["module"] = module
        captured["segment_length"] = segment_length
        return [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "你好",
            }
        ]

    monkeypatch.setattr(asr, "_transcribe_with_whisper", fake_whisper)
    monkeypatch.setattr(asr, "_transcribe_with_iflow", lambda *args, **kwargs: pytest.fail("Fallback should not run"))

    result = asr.transcribe("dummy.mp4")

    assert result == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "你好",
        }
    ]
    assert captured == {
        "audio": str(audio_path),
        "module": fake_module,
        "segment_length": 40.0,
    }


def test_transcribe_falls_back_to_iflow(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(asr, "extract_audio", lambda _: str(audio_path))
    monkeypatch.setenv("WHISPER_ENABLE", "false")
    monkeypatch.setenv("ASR_TIMEOUT_S", "7")
    monkeypatch.setenv("ASR_SEGMENT_S", "10")

    fallback_segments = [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "测试文本。",
        }
    ]

    captured = {}

    def fake_iflow(audio, segment_length, timeout):
        captured["audio"] = audio
        captured["segment_length"] = segment_length
        captured["timeout"] = timeout
        return fallback_segments

    monkeypatch.setattr(asr, "_load_whisper", lambda: None)
    monkeypatch.setattr(asr, "_transcribe_with_iflow", fake_iflow)

    result = asr.transcribe("dummy.mp4")

    assert result == fallback_segments
    assert captured == {
        "audio": str(audio_path),
        "segment_length": 30.0,
        "timeout": 7.0,
    }
