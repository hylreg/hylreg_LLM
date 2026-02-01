# hylreg-LLM

一个基于 LangChain 的简单 RAG (Retrieval-Augmented Generation) 系统，支持多种 LLM 和嵌入模型。

本仓库为 **demo** 工作区，内含多个子项目。使用 [uv](https://docs.astral.sh/uv/) 管理依赖与工作区（与 hylreg_QT、hylreg_OpenCV 一致）。

## 子项目

| 路径 | 说明 |
|------|------|
| [demo/vl_demo](demo/vl_demo/) | 视觉语言 Demo：图片 + 问题 → VL 模型回复 |
| [demo/voice_assistant](demo/voice_assistant/) | 语音助手 Demo（当前为文本模式） |
| [demo/agents](demo/agents/) | 智能体系统：LangChain Agent 封装 |
| [demo/RAG](demo/RAG/) | RAG 系统：文档检索与生成 |

## ✨ 功能特性

- 📄 **文档加载和预处理**：支持文本文档的自动加载和分割
- 🔍 **向量存储和检索**：基于 FAISS 的高效向量检索
- 🎯 **重排序（Reranking）**：支持多种重排序方式，提高检索精度
  - Ollama Reranker（推荐，本地运行）
  - 本地 Qwen Reranker（完全离线）
- 🤖 **多 LLM 支持**：
  - Ollama（本地模型，推荐）
  - 硅基流动 API
  - 魔搭 ModelScope（本地模型）
- 🚀 **易于使用**：简洁的 API，快速上手

## 🚀 快速开始

### 1. 安装依赖

**推荐**：进入子项目目录后再安装与运行。

```bash
# 进入子项目（例如 vl_demo）
cd demo/vl_demo
uv sync
uv run python run_vl_demo.py --image /path/to/image.jpg
# 或
uv run vl-demo --image /path/to/image.jpg
```

若在仓库根目录操作，可先同步工作区依赖，再指定包运行：

```bash
uv sync
uv run --package vl-demo vl-demo --image /path/to/image.jpg
uv run --package voice-assistant voice-assistant
```

各子项目详细说明见其目录下的 `README.md`。运行 **RAG** 示例时，在根目录执行 `uv sync` 后使用：

```bash
uv run python demo/RAG/example_ollama.py
uv run python demo/RAG/benchmark.py
```

### 2. 配置环境

**方式1：使用 Ollama（推荐，本地运行）**

```bash
# 安装 Ollama（如果还没有）
# 访问 https://ollama.ai 下载安装

# 下载需要的模型
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0

# 设置环境变量
export USE_OLLAMA="true"
```

**方式2：使用 API 服务**

```bash
# 硅基流动
export SILICONFLOW_API_KEY="your-api-key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1/"

# 或魔搭 ModelScope（本地模型）
export USE_MODELSCOPE="true"
```

### 3. 准备文档

将文档放在 `demo/RAG/documents/` 目录下（支持 `.txt` 文件）。

### 4. 运行示例

```bash
# 使用 Ollama（推荐）
export USE_OLLAMA="true"
uv run python demo/RAG/example_ollama.py

# 或使用其他 API
uv run python demo/RAG/example.py

# 使用魔搭 ModelScope
export USE_MODELSCOPE="true"
uv run python demo/RAG/example_modelscope.py
```

## 📚 文档

- [RAG 系统详细文档](demo/RAG/README.md) - 完整的使用说明和 API 文档
- [Reranker 部署指南](docs/rag/DEPLOY_RERANKER.md) - 本地 Reranker 部署说明
- [文档索引](docs/README.md) - 所有文档的索引
- [魔搭模型下载指南](docs/魔搭模型下载指南.md) - ModelScope 模型下载使用指南

## 💡 基本使用

```python
from demo.RAG.rag_system import IntelligentRAG

# 初始化 RAG 系统
rag = IntelligentRAG(
    documents_path="./demo/RAG/documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
    ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0",
)

# 构建系统（启用重排序）
rag.build(k=4, use_rerank=True, rerank_top_n=3)

# 查询
result = rag.query("你的问题")
print(result["answer"])
```

## 🔧 系统要求

- Python >= 3.12
- 足够的磁盘空间用于存储向量索引
- 如果使用本地模型，需要足够的 RAM（推荐 8GB+）

## 📦 项目结构

```
hylreg_LLM/
├── README.md                 # 项目主文档（本文件）
├── pyproject.toml            # 项目配置和依赖
├── demo/                     # Demo 工作区子项目
│   ├── vl_demo/             # 视觉语言 Demo
│   ├── voice_assistant/     # 语音助手（llm / audio / core）
│   ├── agents/              # 智能体系统
│   │   ├── __init__.py
│   │   ├── agent_system.py
│   │   ├── example.py       # 智能体示例
│   │   └── README.md
│   └── RAG/                  # RAG 系统
│       ├── __init__.py
│       ├── rag_system.py
│       ├── example.py       # 基础示例
│       ├── example_ollama.py
│       ├── example_modelscope.py
│       ├── benchmark.py
│       ├── documents/       # 文档目录
│       ├── vectorstore/     # 向量存储目录
│       └── README.md
├── scripts/                  # 工具脚本目录
│   ├── download_qwen_reranker.py  # 模型下载脚本
│   └── main.py              # 主入口脚本
├── docs/                     # 文档目录
│   ├── README.md           # 文档索引
│   ├── rag/                # RAG 相关文档
│   │   ├── DEPLOY_RERANKER.md
│   │   ├── BENCHMARK.md
│   │   └── VECTOR_STORE.md
│   ├── 魔搭模型下载指南.md
│   └── ...
└── data/                     # 数据目录
    ├── models/              # 模型文件（可选）
    └── outputs/             # 输出文件
        └── performance_report.json
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[添加许可证信息]
