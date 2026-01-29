# hylreg-LLM

一个基于 LangChain 的简单 RAG (Retrieval-Augmented Generation) 系统，支持多种 LLM 和嵌入模型。

## ✨ 功能特性

- 📄 **文档加载和预处理**：支持文本文档的自动加载和分割
- 🔍 **向量存储和检索**：基于 FAISS 的高效向量检索
- 🎯 **重排序（Reranking）**：支持多种重排序方式，提高检索精度
  - Ollama Reranker（推荐，本地运行）
  - 本地 Qwen Reranker（完全离线）
  - Cohere API（云端服务）
- 🤖 **多 LLM 支持**：
  - Ollama（本地模型，推荐）
  - 硅基流动 API
  - OpenAI API
- 🚀 **易于使用**：简洁的 API，快速上手

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
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

# 或 OpenAI
export OPENAI_API_KEY="your-api-key"
```

### 3. 准备文档

将文档放在 `RAG/documents/` 目录下（支持 `.txt` 文件）。

### 4. 运行示例

```bash
# 使用 Ollama（推荐）
export USE_OLLAMA="true"
uv run python RAG/example_ollama.py

# 或使用其他 API
uv run python RAG/example.py
```

## 📚 文档

- [RAG 系统详细文档](RAG/README.md) - 完整的使用说明和 API 文档
- [Reranker 部署指南](RAG/DEPLOY_RERANKER.md) - 本地 Reranker 部署说明
- [文档索引](docs/README.md) - 所有文档的索引

## 💡 基本使用

```python
from RAG.rag_system import IntelligentRAG

# 初始化 RAG 系统
rag = IntelligentRAG(
    documents_path="./RAG/documents",
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
├── README.md              # 项目主文档（本文件）
├── pyproject.toml         # 项目配置和依赖
├── RAG/                   # RAG 系统核心代码
│   ├── rag_system.py      # RAG 系统实现
│   ├── example_*.py       # 使用示例
│   ├── documents/         # 文档目录
│   └── README.md          # RAG 系统详细文档
└── docs/                  # 文档目录
    └── README.md          # 文档索引
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[添加许可证信息]
