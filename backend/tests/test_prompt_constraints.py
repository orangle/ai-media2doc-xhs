import json
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core import fact_extractor, post_writer, visual_extractor


def test_visual_parse_with_text_prefix():
    content = (
        "说明如下:\n"
        "{\"location\": \"苏州园林\", \"activities\": \"散步、拍照\", \"objects\": [\"湖水\", \"石桥\"], \"text\": \"¥120\"}"
    )

    parsed = visual_extractor._parse_content_to_dict(content)

    assert parsed["place"] == "苏州园林"
    assert parsed["activities"] == ["散步", "拍照"]
    assert parsed["objects"] == ["湖水", "石桥"]
    assert parsed["visible_text"] == "¥120"


def test_fact_normalization_filters_hallucinations():
    allowed = "上海外滩\n步行\n注意安全\n¥50"
    parsed = {
        "地点": "北京",  # not in allowed text
        "费用": "¥50",
        "玩法": ["步行", "潜水"],
        "交通": None,
        "时间": None,
        "注意事项": ["注意安全"],
        "标签": ["城市"],
        "missing": [],
    }

    result = fact_extractor._normalize_fact_output(parsed, allowed, ["¥50"], [])

    assert result["地点"] is None
    assert "地点" in result["missing"]
    assert result["费用"] == "¥50"
    assert result["玩法"] == ["步行"]  # "潜水" filtered out
    assert result["注意事项"] == ["注意安全"]


def test_post_writer_generates_with_style_env(monkeypatch):
    monkeypatch.setenv("IFLOW_API_KEY", "test-key")
    monkeypatch.setenv("POST_WRITER_STYLE", "故事")
    monkeypatch.setenv("POST_WRITER_LENGTH", "简短")

    captured = {}

    def fake_chat_completion(model, messages, timeout_s=0, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["timeout_s"] = timeout_s
        captured["kwargs"] = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "非JSON前缀 {\"title\":\"标题\",\"markdown\":\"正文\"}"}
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(post_writer.iflow_api, "chat_completion", fake_chat_completion)

    facts = {"地点": "上海外滩", "费用": "¥50", "玩法": ["步行"], "missing": ["时间"]}

    result = post_writer.generate_post(facts)

    assert result == {"title": "标题", "markdown": "正文"}
    assert captured["messages"][1]["content"][0]["text"]
    payload = json.loads(captured["messages"][1]["content"][0]["text"])
    assert payload["style"] == "故事"
    assert payload["length"] == "简短"
    assert "时间" in payload["missing_fields"]
