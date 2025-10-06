# AI-Media2Doc · 小红书图文生成助手

## 1. 项目概述
AI-Media2Doc 基于短视频自动生成小红书风格图文内容，核心流程覆盖音视频预处理、语音识别、视觉理解、事实抽取、文案撰写与可视化呈现。系统采用 **Streamlit** 作为前端，后端由纯 Python 组件构成，可在本地 CPU 环境稳定运行。

## 2. 系统结构与模块说明
项目目前完成的能力由五大任务组成：

| 任务 | 模块 | 说明 |
| --- | --- | --- |
| T1 关键帧优化 | `backend/core/video_utils.py` | 基于 ffmpeg 与场景检测选择代表帧，并支持 UI 覆盖帧预算设置 |
| T2 ASR 稳定化 | `backend/core/asr.py` | 先行抽音频，优先 Whisper 分段转写，失败自动回退至 iFlow Qwen3-Max |
| T3 Prompt 强约束 | `backend/core/visual_extractor.py`、`fact_extractor.py`、`post_writer.py` | 统一 JSON Schema、缺失项列举与严格事实守恒 |
| T4 调用稳定性 | `shared/iflow_api.py` | 统一 iFlow API、重试、并发与落盘缓存 |
| T5 证据与导出 | `backend/core/evidence.py`、`tools/exporter.py`、`app/ui.py` | 事实溯源、三栏 UI 展示与一键导出 Markdown/JSON/图包 |

## 3. 环境准备
- 推荐 Python >= 3.10。
- 需要可访问 [iFlow 平台](https://iflowapi.com/) 的网络环境。
- CPU 即可运行，Whisper 模型自动下载到本地缓存。

### macOS 安装指引
```bash
# 安装 Homebrew（若已安装可跳过）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装依赖
brew install python@3.11 ffmpeg
pip3 install --upgrade virtualenv
```

### Linux 安装指引（以 Ubuntu 为例）
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip ffmpeg libsndfile1
```

## 4. 安装步骤
1. **克隆仓库**
   ```bash
   git clone https://github.com/hanshuaikang/AI-Media2Doc-XHS.git
   cd AI-Media2Doc-XHS
   ```
2. **创建并激活虚拟环境**
   - macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
4. **设置环境变量**
   复制 `variables_template.env` 为 `.env` 并填入：
   - `IFLOW_API_KEY`：iFlow 平台密钥
   - `IFLOW_API_URL`：API 地址（默认 `https://api.iflow.cn/v1/chat/completions`）
   - `IFLOW_MODEL_ASR`、`IFLOW_MODEL_VISION`、`IFLOW_MODEL_FACT`、`IFLOW_MODEL_WRITER`
   - 可选：`MAX_WORKERS`、`ASR_TIMEOUT_S`、`ASR_SEGMENT_S`

5. **验证模型连通性**
   ```bash
   python -m shared.iflow_api --ping
   ```
   若返回成功日志，则表示 API Key 与网络配置正确。

## 5. 测试流程
1. 下载示例视频到 `samples/sample.mp4`（自行准备）。
2. 抽取关键帧并验证 ffmpeg：
   ```bash
   python -m backend.core.video_utils --test samples/sample.mp4
   ```
3. 执行 ASR 与视觉理解：
   ```bash
   python -m backend.core.asr samples/sample.mp4
   python -m backend.core.visual_extractor samples/sample.mp4
   ```
4. 生成图文 Markdown：
   ```bash
   python -m backend.core.post_writer --sample samples/sample.mp4
   ```
5. 启动 Streamlit UI（默认端口 8501，可通过 `--server.port` 指定）：
   ```bash
   streamlit run app/ui.py --server.port 7860
   ```

## 6. 正常运行与导出
- 打开浏览器访问 `http://localhost:7860`，上传视频后即可查看三栏布局：关键帧九宫格、Markdown 文案、事实 JSON + 证据。
- 点击“导出”按钮生成压缩包，默认路径为 `output/YYYYMMDD_xhs_post.zip`，包含 `post.md`、`facts.json` 与 `images.zip`。
- 使用 `python tools/eval.py --input output/YYYYMMDD_xhs_post.zip` 可进行简单评测验证。

## 7. 常见问题（Q&A）
- **模型连接错误**：检查 `IFLOW_API_KEY` 是否填写、网络是否可访问 iFlow；可再次运行 `python -m shared.iflow_api --ping` 验证。
- **ffmpeg/whisper 安装失败**：macOS 通过 `brew install ffmpeg`，Linux 通过 `sudo apt-get install ffmpeg libsndfile1`；Whisper 初次运行会自动下载模型，请保持磁盘空间充足。
- **Streamlit 端口占用**：启动时加上 `--server.port 7861` 等备用端口，或释放被占用端口。

## 8. 附录
- **目录结构**：详见 `TECH_DESIGN.md` 与 `backend/core` 内模块划分。
- **环境变量样例**：参考 `.env.example` 或 `variables_template.env`。
- **样例命令集**：
  - macOS：
    ```bash
    source .venv/bin/activate
    streamlit run app/ui.py --server.port 7860
    ```
  - Linux：
    ```bash
    source .venv/bin/activate
    python -m backend.core.video_utils --test samples/sample.mp4
    ```

完成上述步骤后即可在本地或服务器上快速体验 AI-Media2Doc 的自动图文生成流程。
