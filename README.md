# AI-Media2Doc · 小红书图文生成助手

AI-Media2Doc 是一个开源的端到端应用，帮助创作者将旅行 Vlog、生活记录等短视频快速转成符合小红书风格的图文笔记。项目采用 **Streamlit** 作为 Web 前端，结合一套纯 Python 的后端管线：抽取视频关键帧和音频、调用 iFlow 平台的 Qwen3 系列模型完成语音识别、视觉理解、信息抽取与文案生成，最终在浏览器中展示结果。

- ⚖️ 完全开源，MIT 许可，可在本地或私有环境中部署。
- 🔐 隐私友好，所有临时文件仅保存在本地机器。
- 🧠 一站式 AI 管线：ASR → 视觉理解 → 结构化事实 → 小红书文案。
- 🚀 即开即用：安装依赖后执行一条 Streamlit 命令即可启动。

---

## 架构概览

整个系统围绕一个同步执行的推理管线展开：

```
用户上传视频
      ↓
视频处理（抽音频、关键帧提取）
      ↓
语音识别（本地 Whisper 或 Qwen3-Max）
      ↓
多帧视觉理解（Qwen3-VL-Plus）
      ↓
事实抽取（Qwen3-Max）
      ↓
小红书文案生成（Qwen3-Max）
      ↓
Streamlit 前端展示
```

核心模块说明：

| 模块 | 位置 | 作用 |
| --- | --- | --- |
| `video_utils.py` | `backend/core` | 使用 `ffmpeg` 与 `PySceneDetect` 抽取音频与关键帧 |
| `asr.py` | `backend/core` | 优先调用本地 Whisper，失败时回落到 iFlow Qwen3-Max 完成转写 |
| `visual_extractor.py` | `backend/core` | 逐帧调用 Qwen3-VL-Plus，输出地点、活动、情绪等信息 |
| `fact_extractor.py` | `backend/core` | 根据语音与视觉结果提炼结构化旅行要素 |
| `post_writer.py` | `backend/core` | 把结构化事实转成小红书风格标题与 Markdown |
| `app/ui.py` | `app` | Streamlit 页面逻辑与文件上传、结果渲染 |

---

## 目录结构

```
.
├── app/                # Streamlit 前端
│   └── ui.py
├── backend/
│   └── core/           # 推理管线核心模块
├── requirements.txt    # Python 依赖
├── TECH_DESIGN.md      # 详细技术设计（英文）
├── README_EN.md        # 英文说明
└── variables_template.env
```

---

## 环境准备

1. **系统要求**
   - Python 3.10+
   - `ffmpeg` 可执行文件已加入系统 PATH
   - 可访问 [iFlow 平台](https://iflowapi.com/) 的网络环境

2. **安装依赖**

   ```bash
   git clone https://github.com/hanshuaikang/AI-Media2Doc.git
   cd AI-Media2Doc
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **配置环境变量**

   参考 `variables_template.env` 新建 `.env` 或 `variables.env` 文件，并设置下列变量：

   | 变量 | 说明 | 默认值 |
   | --- | --- | --- |
   | `IFLOW_API_KEY` | iFlow 平台的访问密钥 | 必填 |
   | `IFLOW_API_URL` | iFlow Chat Completions API 地址 | `https://api.iflow.cn/v1/chat/completions` |
   | `IFLOW_MODEL_ASR` | 语音识别模型名称 | `qwen3-max` |
   | `IFLOW_MODEL_VISION` | 视觉理解模型名称 | `qwen3-vl-plus` |
   | `IFLOW_MODEL_FACT` | 事实抽取模型名称 | `qwen3-max` |
   | `IFLOW_MODEL_WRITER` | 文案生成模型名称 | `qwen3-max` |

   > 提示：本地已安装 Whisper 模型时会优先走本地推理；未安装或失败时自动回落到 iFlow 云端模型。

---

## 启动与使用

1. **启动应用**

   ```bash
   streamlit run app/ui.py
   ```

   首次运行会在终端打印访问地址（默认为 `http://localhost:8501`）。

2. **生成图文流程**
   1. 在页面左上角上传一段不超过数分钟的短视频（支持 `mp4`/`mov`/`mkv` 等）。
   2. 点击 “生成图文” 按钮，系统会依次完成抽帧、语音识别、视觉理解与文案生成。
   3. 页面展示包含三个部分：
      - **小红书文案**：带标题的 Markdown 内容，可直接复制粘贴；
      - **结构化信息**：地点、交通、玩法、注意事项等 JSON 结果；
      - **抽帧预览**：自动抽取的关键帧缩略图，便于挑选配图。

3. **示例输出**

   ```markdown
   # 🚶‍♀️ 周末逛城南古街
   - ✅ 必打卡：城门广场夜景
   - 🍜 小吃攻略：豆花面 + 烤冷面
   - 🚇 地铁 2 号线直达，步行 5 分钟
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

## 常见问题

- **Whisper 下载较慢怎么办？** 可以暂时跳过安装，系统会自动使用 iFlow 的 Qwen3-Max 完成转写。
- **如何清理缓存文件？** 上传的视频与抽帧存放在系统临时目录下，会在会话结束后自动删除。
- **可以接入其他模型吗？** 只需在 `backend/core` 模块中替换 API 调用，并调整相应的环境变量即可。

---

## 贡献与许可

欢迎提交 Issue / PR 改进模型提示词、前端体验或部署流程。所有代码以 [MIT License](./LICENSE) 发布，商业或二次开发请保留原作者署名。
