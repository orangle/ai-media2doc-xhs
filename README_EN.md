# AI-Media2Doc · Generate Xiaohongshu Posts from Video

AI-Media2Doc is an end-to-end workflow that turns short-form videos into Xiaohongshu-style articles. A lightweight **Streamlit** UI wraps a pure-Python backend pipeline that extracts audio and key frames, calls Qwen3 models on the iFlow platform for speech recognition, visual understanding, fact structuring, and finally produces polished Markdown ready to publish.

- ⚖️ MIT licensed and open source – deploy it anywhere.
- 🔐 Privacy-friendly – temporary files stay on your machine.
- 🧠 Unified AI pipeline: ASR → vision → fact extraction → copywriting.
- 🚀 Zero-config runtime – start Streamlit and you are ready to create.

---

## Architecture

```
Upload video
      ↓
Media preprocessing (audio extraction & key frames)
      ↓
ASR (local Whisper fallback or Qwen3-Max)
      ↓
Vision grounding (Qwen3-VL-Plus)
      ↓
Structured fact extraction (Qwen3-Max)
      ↓
Xiaohongshu-style writing (Qwen3-Max)
      ↓
Streamlit rendering
```

| Module | Location | Responsibility |
| --- | --- | --- |
| `video_utils.py` | `backend/core` | Extract audio & key frames via `ffmpeg` + `PySceneDetect` |
| `asr.py` | `backend/core` | Run Whisper locally when available, otherwise call iFlow Qwen3-Max |
| `visual_extractor.py` | `backend/core` | Call Qwen3-VL-Plus to describe each frame |
| `fact_extractor.py` | `backend/core` | Summarise ASR + vision outputs into structured travel facts |
| `post_writer.py` | `backend/core` | Generate Xiaohongshu title & Markdown from facts |
| `app/ui.py` | `app` | Streamlit UI: upload, trigger pipeline, render results |

---

## Repository Layout

```
.
├── app/
│   └── ui.py              # Streamlit entrypoint
├── backend/
│   └── core/              # Processing pipeline modules
├── requirements.txt
├── TECH_DESIGN.md
├── README.md              # Chinese guide
└── variables_template.env
```

---

## Getting Started

1. **Prerequisites**
   - Python 3.10+
   - `ffmpeg` available on the system PATH
   - Network access to the iFlow API (`https://api.iflow.cn`)

2. **Install dependencies**

   ```bash
   git clone https://github.com/hanshuaikang/AI-Media2Doc.git
   cd AI-Media2Doc
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Environment variables**

   Create a `.env` or `variables.env` file based on `variables_template.env` and configure:

   | Variable | Description | Default |
   | --- | --- | --- |
   | `IFLOW_API_KEY` | API key for iFlow | required |
   | `IFLOW_API_URL` | Chat Completions endpoint | `https://api.iflow.cn/v1/chat/completions` |
   | `IFLOW_MODEL_ASR` | Model used for speech recognition | `qwen3-max` |
   | `IFLOW_MODEL_VISION` | Model used for frame understanding | `qwen3-vl-plus` |
   | `IFLOW_MODEL_FACT` | Model used for fact extraction | `qwen3-max` |
   | `IFLOW_MODEL_WRITER` | Model used for copywriting | `qwen3-max` |

   > Whisper is optional – if it is not installed, ASR automatically falls back to the Qwen3-Max API call.

---

## Run the App

```bash
streamlit run app/ui.py
```

The terminal prints the URL (default `http://localhost:8501`). Upload a short video and click **Generate** to trigger the pipeline. The result page includes:

- **Xiaohongshu copy** in Markdown – copy & publish directly.
- **Structured facts** – location, budget, activities, transportation, notes, and tags.
- **Frame gallery** – extracted key frames for manual selection.

Example output:

```markdown
# 🚶‍♀️ Weekend Walk in South Street
- ✅ Must see: Gate plaza at night
- 🍜 Street food combo: Tofu pudding noodles + grilled cold noodles
- 🚇 Metro Line 2, 5 min walk
```

```json
{
  "地点": "城南古街",
  "费用": "人均 60 元",
  "玩法": ["夜游老街", "品尝街边小吃"],
  "交通": "地铁 2 号线城南站",
  "时间": "周末傍晚",
  "注意事项": ["早点排队购买小吃", "夜间注意保暖"],
  "标签": ["夜景", "市集", "城市慢生活"]
}
```

---

## Contributing & License

Contributions are welcome – feel free to submit issues or pull requests for model prompts, UX, or deployment guides. The project is released under the [MIT License](./LICENSE).
