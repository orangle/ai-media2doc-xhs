# 技术设计文档：视频转小红书图文系统（Codex + Qwen3 架构版）

## 一、项目背景与目标

- **背景**  
  旅游短视频、探店视频等常见于社交平台，很多用户希望从视频自动生成图文攻略内容。  
 目前人工视频转文案成本高、效率低，本系统旨在自动化这一流程。

- **目标**  
 输入：一段短视频（≤ 90s）  
 输出：一篇符合小红书风格的图文攻略（标题 + Markdown 正文 + 相关图片），并附带结构化事实信息（地点 / 费用 / 玩法 / 注意事项 等）。

- **前提假设 / 约束**  
  1. 后端使用 iFlow 平台的 **Qwen3-VL-Plus** 做视觉理解，**Qwen3-Max** 做抽取与文案生成。  
  2. 部署环境为 Codex Cloud，所有逻辑在云端运行，前端轻量使用 Streamlit。  
  3. 不再使用本地 OCR（如 PaddleOCR），文字识别由视觉大模型负责。  
  4. 用户操作尽可能简单：上传视频 → 点击生成 → 获取图文。

---

## 二、系统总览流程



用户上传视频
↓
视频预处理（音频抽取 + 帧抽取 + 镜头切分）
↓
语音识别模块（Whisper 本地 / Qwen3-Max 备用）
↓
视觉理解模块（Qwen3-VL-Plus 分析帧图像）
↓
事实抽取模块（融合 ASR + 视觉理解 → 结构化 facts via Qwen3-Max）
↓
文案生成模块（facts → 小红书风格 Markdown via Qwen3-Max）
↓
Streamlit 前端展示 + 导出


---

## 三、模块设计与接口规范

### 3.1 video_utils 模块

- **职责**  
  处理视频预处理：提取音频、关键帧、镜头分割。

- **接口 / 函数**  
  ```python
  def extract_audio(video_path: str) -> str
  def extract_keyframes(video_path: str, fps: int = 1) -> List[str]
  def detect_scenes(video_path: str) -> List[Tuple[float, float]]
  ```


实现要点 / 考虑

- 使用 ffmpeg 抽帧 / 抽音频
- 使用 PySceneDetect 或基于帧差异的方法切分镜头
- 控制帧数，避免生成过多帧影响性能

### 3.2 asr 模块

- **职责**  
  将音频转为带时间戳的文字块。

- **接口**

  ```python
  def transcribe(video_path: str) -> List[Dict]
  # 每个 item 为 {"start": float, "end": float, "text": str}
  ```

- **策略**

  - 优先使用 Whisper 本地模型
  - 若本地不可用，使用 Qwen3-Max API 作为 fallback
  - 返回的文本需包含时间戳，用于后续对齐与证据管理

### 3.3 visual_extractor 模块

- **职责**  
  对帧图像进行视觉 + 文本理解，提取语义信息。

- **接口**

  ```python
  def analyze_frame(image_path: str) -> Dict
  def extract_visual_facts(frame_paths: List[str]) -> List[Dict]
  ```

- **调用方式**

  - 使用 iFlow API 请求 Qwen3-VL-Plus 模型
  - prompt 指令需要涵盖：地点、活动、物体、场景文字识别等
  - 返回 JSON 格式，字段可自由扩展

### 3.4 fact_extractor 模块

- **职责**  
  整合 ASR 文本 + 视觉理解结果，由 Qwen3-Max 抽取结构化 facts。

- **接口**

  ```python
  def extract_facts(asr_data: List[Dict], visual_data: List[Dict]) -> Dict
  ```

- **输出结构（示例）**

  ```json
  {
    "地点": "...",
    "费用": "...",
    "玩法": ["..."],
    "交通": "...",
    "时间": "...",
    "注意事项": ["..."],
    "标签": ["..."]
  }
  ```

- **设计要点**

  - prompt 要包含明确 schema 及字段说明
  - 可做后处理校验（如字段非空、数值合理性）

### 3.5 post_writer 模块

- **职责**  
  将 facts 转化为小红书风格的图文内容。

- **接口**

  ```python
  def generate_post(facts: Dict) -> Dict
  # 返回格式 {"title": str, "markdown": str}
  ```

- **风格要求**

  - 使用 Emoji、段落结构、口语化语气
  - 坚守事实，不允许生成无依据内容
  - 对于低置信度信息可使用“可能 / 约 / 我看到”措辞

### 3.6 Streamlit UI 模块

- **职责**  
  提供交互界面：上传视频、展示、导出。

- **流程**

  1. 上传视频
  2. 显示预览（st.video）
  3. 点击生成 → 后端 pipeline 执行
  4. 展示结果 title + Markdown
  5. 导出 JSON / Markdown

- **端口 / 启动**  
  使用 streamlit run app/ui.py --server.port 7860

## 四、数据与格式标准

- **facts JSON 示例**

  ```json
  {
    "地点": "苏州园林",
    "费用": "50 元",
    "玩法": ["拙政园参观", "园林漫步", "品茶"],
    "交通": "地铁 + 步行",
    "时间": "2 小时",
    "注意事项": ["防晒", "闭园日注意"],
    "标签": ["古镇", "园林", "拍照"]
  }
  ```

- **post 输出格式**

  ```json
  {
    "title": "☀️ 苏州一日游：园林打卡全攻略",
    "markdown": "## 为什么去苏州\n…\n## 路线攻略\n…\n## 费用 & 时间\n…\n## 注意事项\n…"
  }
  ```
