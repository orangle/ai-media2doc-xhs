import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core import evidence


def test_evidence_build_and_attach():
    asr_segments = [
        {
            "start": 0.0,
            "end": 4.2,
            "text": "欢迎来到苏州园林，门票¥80，开放时间很充裕，这条逛园林路线带你看湖水和古桥，非常值得收藏。",
        },
        {"start": 4.2, "end": 6.0, "text": "记得注意安全。"},
    ]
    keyframe_selection = {
        "chosen": [
            {
                "frame_id": "frame_0001",
                "path": "/tmp/frame1.jpg",
                "final_score": 0.83,
                "vlm": {"representativeness": 0.8, "has_landmark": True, "has_readable_text": False},
            }
        ]
    }
    vision_results = [
        {
            "image_path": "/tmp/frame1.jpg",
            "place": "苏州园林",
            "visible_text": "开放时间 09:00",
            "activities": ["逛园林"],
            "objects": ["拱桥"],
        }
    ]

    evidences = evidence.build_evidences(asr_segments, keyframe_selection, vision_results)
    assert len(evidences) == 3  # two ASR segments + one vision frame
    assert any(item["type"] == "vision" and item.get("path") == "/tmp/frame1.jpg" for item in evidences)
    assert any(item["type"] == "asr" and "门票" in item.get("keywords", []) for item in evidences)

    facts = {
        "地点": "苏州园林",
        "费用": "¥80",
        "玩法": ["逛园林"],
        "交通": None,
        "时间": "09:00",
        "注意事项": ["注意安全"],
        "标签": [],
        "missing": ["交通"],
    }

    bundle = evidence.attach_facts(facts, evidences)
    strict = bundle["facts_strict"]
    weak = bundle["facts_weak"]

    assert strict["地点"] == "苏州园林"
    assert strict["费用"] == "¥80"
    assert strict["玩法"] == ["逛园林"]
    assert "地点" in strict["evidence_ids"]
    assert weak["时间"] == "09:00"
    assert "时间" in weak["evidence_ids"]
    assert "交通" in bundle["missing"]
