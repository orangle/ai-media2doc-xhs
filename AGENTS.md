# 🧠 Project: AI-Media2Doc → 小红书图文生成系统 (Qwen3 + Codex Cloud 版)

## 🎯 目标
将开源项目 AI-Media2Doc 改造成一个 **纯后端 + Streamlit 前端** 的版本，可在 **Codex Cloud** 上端到端运行。  
使用 iFlow 平台的 **Qwen3-VL-Plus** 做视觉理解，**Qwen3-Max** 做信息抽取与文案生成。

## ⚙️ 功能清单

| 模块 | 作用 | 使用模型 / 技术 |
|---|---|---|
| 视频预处理 | 抽帧、抽音频、镜头切分 | ffmpeg / PySceneDetect |
| 语音识别 | 音频转文字 | Whisper（本地）或 Qwen3-Max（API） |
| 视觉理解 | 画面语义 + 识别文字 | **Qwen3-VL-Plus** |
| 信息抽取 | ASR + 视觉结果 → 结构化 facts | **Qwen3-Max** |
| 文案生成 | facts → 小红书风格图文 | **Qwen3-Max** |
| Web 展示 | 上传视频 + 显示图文 | Streamlit |

## 🧱 文件结构

backend/
├─ core/
│ ├─ asr.py
│ ├─ video_utils.py
│ ├─ visual_extractor.py
│ ├─ fact_extractor.py
│ ├─ post_writer.py
│ ├─ schema.py
│ └─ __init__.py
app/
└─ ui.py
requirements.txt
AGENTS.md

## 🧩 环境要求 & 配置

- Python 3.10+
- ffmpeg 可执行在系统 PATH
- streamlit、requests 等 Python 包
- iFlow 平台 API Key（环境变量 `IFLOW_API_KEY`）
- iFlow 模型端点：  
  `https://api.iflow.cn/v1/chat/completions`  
  模型名：`qwen3-vl-plus`, `qwen3-max`
