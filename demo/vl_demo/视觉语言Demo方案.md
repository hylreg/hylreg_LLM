# 视觉语言 (VL) Demo 方案文档

## 1. 项目概述

### 1.1 目标

在 `demo` 目录下实现一个**视觉语言 (Vision-Language) Demo**，实现「本地图片 + 文本问题 → VL 模型理解 → 文本回复」的简单闭环，用于验证本仓库在图文多模态场景下的可用性。

### 1.2 核心能力

- **图片输入**：用户提供本地图片路径（或后续扩展为 URL）。
- **文本问题**：用户对图片的提问（如「描述这张图」「图中有什么」）；可选，默认可为「请描述这张图片」。
- **VL 模型推理**：仅使用 **ollama、siliconflow、modelscope** 的视觉/多模态 API，优先 **Qwen 系列 VL 模型**。
- **文本输出**：模型对「图 + 问」的回复，打印到控制台。

### 1.3 约束与原则

- 遵循项目规则：**仅使用 ollama、siliconflow、modelscope** 的 API。
- API Key 等从 **`~/.zshrc` 环境变量** 读取。
- 使用 **uv** 管理 Python 依赖与运行环境。
- Demo 实现前先完成本技术方案文档（当前文档）。

---

## 2. 技术选型

### 2.1 视觉语言模型 (VL)

| 方案           | 说明 |
|----------------|------|
| **Ollama**     | 本地部署，无 API Key；需拉取支持视觉的模型，如 `llava`、`qwen2-vl` 等。 |
| **SiliconFlow** | 云端 API，需 `SILICONFLOW_API_KEY`；支持 Qwen2-VL、Qwen2.5-VL、Qwen3-VL 等，OpenAI 兼容的 `/chat/completions`，消息中可带 `image_url`（URL 或 base64）。 |
| **ModelScope** | 云端/本地均可；VL 模型可通过魔搭 API 或本地 transformers 管线接入，本 Demo 第一版可选实现。 |

**建议**：Demo 默认通过环境变量 `USE_OLLAMA` / `SILICONFLOW_API_KEY` 切换；优先使用 **Qwen 系列 VL 模型**（如 SiliconFlow 的 Qwen2.5-VL、Ollama 的 qwen2-vl）。

### 2.2 输入与输出

- **输入**：本地图片路径（字符串）+ 可选文本问题（字符串，默认「请描述这张图片」）。
- **输出**：模型回复的纯文本，打印到控制台；后续可扩展为返回结构化结果或写入文件。

---

## 3. 系统架构

### 3.1 数据流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  本地图片   │ ──▶ │  VL 客户端  │ ──▶ │  控制台输出 │
│  + 文本问题 │     │  图+问→API  │     │  模型回复   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 3.2 模块职责

| 模块         | 职责 |
|--------------|------|
| **VL 客户端** | 读取本地图片为 base64（或传 URL）；根据环境变量选择 Ollama / SiliconFlow（及可选 ModelScope）调用对应 VL API；将「图片 + 文本问题」发送给模型，解析并返回文本回复。 |
| **入口脚本** | 解析命令行参数（图片路径、可选问题），调用 VL 客户端，打印结果。 |

### 3.3 目录结构

```
demo/
├── vl_demo/
│   ├── README.md              # 项目说明与运行方式
│   ├── 视觉语言Demo方案.md     # 本方案文档
│   ├── run_vl_demo.py         # 入口脚本
│   └── vl_demo/               # 实现包
│       ├── __init__.py
│       └── vl_client.py       # VL 客户端（Ollama / SiliconFlow）
```

---

## 4. 接口与实现要点

### 4.1 VL 客户端接口

- `VLClient(model: str, ...)`  
  - 根据环境变量选择后端（Ollama / SiliconFlow）。
  - `model`：Ollama 模型名（如 `llava`、`qwen2-vl`）或 SiliconFlow 模型名（如 `Qwen/Qwen2.5-VL-72B-Instruct`）。
- `ask(image_path: str, question: str = "请描述这张图片") -> str`  
  - 读取 `image_path` 对应图片，与 `question` 一起发给 VL 模型，返回模型回复文本。

### 4.2 Ollama

- 使用 `/api/chat`，请求体包含 `model`、`messages`。
- 用户消息中：`content` 为文本问题，`images` 为 base64 字符串数组（不含 `data:image/...;base64,` 前缀，按 Ollama 文档约定）。
- 响应中取 `message.content` 作为回复。

### 4.3 SiliconFlow

- 使用 OpenAI 兼容的 `POST /v1/chat/completions`。
- `messages` 中一条 user 消息，`content` 为数组：先 `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`，再 `{"type":"text","text": question}`。
- 使用项目已有的 `ChatOpenAI` + `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL` 即可复用；或直接 `requests` 调用。

### 4.4 ModelScope（可选）

- 第一版可不实现；若实现，可采用魔搭提供的 VL 模型 API 或本地 `transformers` + VL 模型管线，与现有 agents 的 ModelScope 使用方式保持一致（环境变量 `USE_MODELSCOPE`）。

---

## 5. 运行方式

- 在**项目根目录**执行：  
  `uv run python demo/vl_demo/run_vl_demo.py --image /path/to/image.jpg [--question "你的问题"]`
- 或进入 `demo/vl_demo` 后：  
  `uv run python run_vl_demo.py --image /path/to/image.jpg [--question "你的问题"]`
- 环境变量示例：
  - Ollama：`USE_OLLAMA=true`，Ollama 服务需已启动并已拉取 VL 模型。
  - SiliconFlow：`SILICONFLOW_API_KEY=xxx`，可选 `SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/`。

---

## 6. 后续可扩展

- 支持图片 URL 输入。
- 支持多图 + 一问。
- 接入 ModelScope VL API 或本地 VL 管线。
- 将回复写入文件或接入其他 demo（如与语音助手串联）。
