# AI-Media2Doc 安装与部署指南

## 1. 项目概述
AI-Media2Doc-XHS 用于将短视频自动转换为小红书风格的 Markdown 图文。系统由 Streamlit 前端与 Python 后端组成，后端依托 iFlow 平台的 Qwen3 系列模型完成语音识别、视觉理解、事实抽取与文案生成。

## 2. 系统结构与模块说明
五大核心任务构成完整流水线：

- **T1 关键帧优化**（`backend/core/video_utils.py`）：抽取视频音频与关键帧，并与 UI 的“VL 帧预算”滑杆联动。
- **T2 ASR 稳定化**（`backend/core/asr.py`）：优先使用本地 Whisper，支持长音频分段与 Qwen3-Max 回退。
- **T3 Prompt 强约束**（`backend/core/visual_extractor.py`、`fact_extractor.py`、`post_writer.py`）：确保 JSON Schema 一致、缺失项列举与事实守恒。
- **T4 调用稳定性**（`shared/iflow_api.py`）：封装 iFlow API，提供重试、并发控制、落盘缓存与日志。
- **T5 证据与导出**（`backend/core/evidence.py`、`tools/exporter.py`、`app/ui.py`）：构建证据链、三栏 UI 展示与导出 `post.md`、`facts.json`、`images.zip`。

## 3. 环境准备

### 3.1 通用要求
- Python >= 3.10
- 可访问互联网，能够连通 iFlow 平台
- CPU 即可运行（Whisper 自动下载模型，iFlow 模型走云端 API）

### 3.2 macOS 安装指引
1. 安装 Homebrew（如已安装可跳过）：
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. 安装 Python、ffmpeg 及辅助工具：
   ```bash
   brew install python@3.11 ffmpeg
   pip3 install --upgrade virtualenv
   ```
3. 可选：若需指定 Whisper 模型缓存目录，可设置 `XDG_CACHE_HOME`。

### 3.3 Linux 安装指引（Ubuntu/Debian）
1. 更新并安装系统依赖：
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-venv python3-pip ffmpeg libsndfile1
   ```
2. 可选：若使用代理访问 iFlow，请配置 `HTTP_PROXY`/`HTTPS_PROXY`。

## 4. 安装步骤

### 4.1 克隆仓库
```bash
git clone https://github.com/hanshuaikang/AI-Media2Doc-XHS.git
cd AI-Media2Doc-XHS
```

### 4.2 创建虚拟环境
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

### 4.3 安装依赖
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4 设置环境变量
1. 复制模板：
   ```bash
   cp variables_template.env .env
   ```
2. 编辑 `.env` 或通过命令行导出以下变量：
   - `IFLOW_API_KEY`：必填，来自 iFlow 平台
   - `IFLOW_API_URL`：默认为 `https://api.iflow.cn/v1/chat/completions`
   - `IFLOW_MODEL_ASR`、`IFLOW_MODEL_VISION`、`IFLOW_MODEL_FACT`、`IFLOW_MODEL_WRITER`
   - `MAX_WORKERS`：并发线程数，默认 4
   - `ASR_TIMEOUT_S`：ASR 每段超时时间，默认 120
   - `ASR_SEGMENT_S`：音频分段秒数，默认 45
   - 其他可选项：`HTTP_PROXY`、`HTTPS_PROXY`
3. 通过 `export $(cat .env | xargs)` 或使用 [direnv](https://direnv.net/) 载入变量。

### 4.5 验证模型连通性
执行以下命令检查 API key 与网络是否配置正确：
```bash
python -m shared.iflow_api --ping
```
若输出包含 “Ping success” 或状态码 200，即可继续部署。

## 5. 测试流程

### 5.1 准备示例视频
将测试视频放入 `samples/sample.mp4`（路径可自定义）。

### 5.2 关键帧抽取验证
```bash
python -m backend.core.video_utils --test samples/sample.mp4
```
确保命令完成且生成 `samples/sample_frames/` 目录。

### 5.3 语音识别与视觉理解
```bash
python -m backend.core.asr samples/sample.mp4
python -m backend.core.visual_extractor samples/sample.mp4
```
输出将包括转写片段与视觉 JSON。

### 5.4 事实抽取与文案生成
```bash
python -m backend.core.fact_extractor --sample samples/sample.mp4
python -m backend.core.post_writer --sample samples/sample.mp4
```
得到结构化 JSON 与 Markdown 草稿。

### 5.5 启动前端 UI
```bash
streamlit run app/ui.py --server.port 7860
```
首启动会展示配置面板与“VL 帧预算”滑杆。访问 `http://<服务器 IP>:7860` 进行交互。

## 6. 正常运行与导出
1. 在 UI 上传视频，等待管线完成。页面包含：
   - 左列：关键帧九宫格，可设置封面。
   - 中列：Markdown 文案，提供“仅重写本段”操作。
   - 右列：事实 JSON，每个槽位可展开查看对应证据（字幕片段与帧缩略图）。
2. 点击“导出”按钮生成压缩包，默认存放在 `output/YYYYMMDD_xhs_post.zip`。压缩包内包含：
   - `post.md`：UTF-8 编码的 Markdown
   - `facts.json`：结构化事实，含证据 ID
   - `images.zip`：关键帧图包，封面与九宫格一致
3. 使用 `python tools/eval.py --input output/YYYYMMDD_xhs_post.zip` 进行快速评测，确认结果可交付。

## 7. 常见问题（Q&A）
- **模型连接错误**：
  - 确认 `IFLOW_API_KEY` 正确、API URL 未拼写错误。
  - 检查代理或防火墙；必要时设置 `HTTP_PROXY`/`HTTPS_PROXY`。
  - 再次运行 `python -m shared.iflow_api --ping` 观察日志。
- **ffmpeg/whisper 安装失败**：
  - macOS：执行 `brew install ffmpeg`。
  - Linux：执行 `sudo apt-get install ffmpeg libsndfile1`。
  - Whisper 首次使用会自动下载模型，请保证磁盘空间（>2GB）。
- **Streamlit 端口占用**：
  - 使用 `lsof -i :7860` 找到占用进程并结束。
  - 或调整命令为 `streamlit run app/ui.py --server.port 7861`。
- **缓存文件过多**：
  - UI 顶部提供“清空缓存”按钮，或手动删除 `.cache/` 目录。

## 8. 附录

### 8.1 目录结构
```text
.
├── app/
│   └── ui.py
├── backend/
│   └── core/
│       ├── asr.py
│       ├── evidence.py
│       ├── fact_extractor.py
│       ├── post_writer.py
│       ├── video_utils.py
│       └── visual_extractor.py
├── shared/iflow_api.py
├── tools/exporter.py
├── requirements.txt
└── variables_template.env
```

### 8.2 环境变量样例
```env
IFLOW_API_KEY=your_iflow_key
IFLOW_API_URL=https://api.iflow.cn/v1/chat/completions
IFLOW_MODEL_ASR=qwen3-max
IFLOW_MODEL_VISION=qwen3-vl-plus
IFLOW_MODEL_FACT=qwen3-max
IFLOW_MODEL_WRITER=qwen3-max
MAX_WORKERS=4
ASR_TIMEOUT_S=120
ASR_SEGMENT_S=45
```

### 8.3 样例命令集
- **macOS**
  ```bash
  source .venv/bin/activate
  streamlit run app/ui.py --server.port 7860
  ```
- **Linux**
  ```bash
  source .venv/bin/activate
  python -m backend.core.video_utils --test samples/sample.mp4
  ```

完成以上步骤后，即可在 macOS 开发环境或 Linux 服务器上稳定运行 AI-Media2Doc，实现从短视频到小红书风格图文的一键生成。
