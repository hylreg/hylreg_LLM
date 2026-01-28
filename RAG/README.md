# 简单 RAG 系统

这是一个使用 LangChain 实现的简单 RAG (Retrieval-Augmented Generation) 系统。

## 功能特性

- 文档加载和预处理
- 文档分割
- 向量存储和检索
- **重排序（Reranking）**：使用 Cohere Rerank 提高检索精度
- 基于检索的问答生成
- 支持多种 LLM：Ollama、硅基流动、OpenAI

## 安装依赖

```bash
# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 环境配置

### 方式1: 使用 Ollama（本地模型，推荐用于本地开发）

首先确保已安装并运行 Ollama：

```bash
# 安装 Ollama（如果还没有）
# 访问 https://ollama.ai 下载安装

# 启动 Ollama 服务（通常会自动启动）
# 下载需要的模型，例如：
ollama pull llama2
ollama pull nomic-embed-text
```

设置环境变量：

```bash
export USE_OLLAMA="true"
export OLLAMA_BASE_URL="http://localhost:11434"  # 默认地址，可省略
```

或在 `.env` 文件中配置：

```env
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
```

**使用 Ollama 时的模型配置**：
- 嵌入模型推荐：`nomic-embed-text`、`all-minilm`
- LLM 模型推荐：`llama2`、`llama3`、`mistral`、`qwen` 等

### 方式2: 使用硅基流动 API

设置环境变量：

```bash
export SILICONFLOW_API_KEY="your-siliconflow-api-key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1/"
```

或在 `.env` 文件中配置：

```env
SILICONFLOW_API_KEY=your-siliconflow-api-key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/
```

### 方式3: 使用 OpenAI API

设置环境变量：

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

或在 `.env` 文件中配置：

```env
OPENAI_API_KEY=your-openai-api-key
```

### 方式4: 启用重排序（可选，推荐）

重排序可以显著提高检索质量，使用 Cohere Rerank API：

```bash
export COHERE_API_KEY="your-cohere-api-key"
```

或在 `.env` 文件中配置：

```env
COHERE_API_KEY=your-cohere-api-key
```

**注意**: 
- 优先级顺序为：Ollama > 硅基流动 > OpenAI
- 重排序功能需要设置 `COHERE_API_KEY`，并在 `build()` 方法中设置 `use_rerank=True`

## 使用方法

### 1. 准备文档

将你的文档放在 `documents/` 目录下（支持 `.txt` 文件）。

### 2. 运行示例

```bash
cd RAG
python example.py
```

### 3. 在代码中使用

**使用 Ollama**：
```python
from rag_system import IntelligentRAG

# 初始化 RAG 系统（使用 Ollama）
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="nomic-embed-text",  # Ollama 嵌入模型
    llm_model="llama2",  # Ollama LLM 模型
)

# 构建系统（启用重排序）
rag.build(k=4, use_rerank=True, rerank_top_n=3)

# 查询
result = rag.query("你的问题")
print(result["answer"])
```

**使用硅基流动或 OpenAI**：
```python
from rag_system import IntelligentRAG

# 初始化 RAG 系统
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-3.5-turbo",
)

# 构建系统（启用重排序）
rag.build(k=4, use_rerank=True, rerank_top_n=3)

# 查询
result = rag.query("你的问题")
print(result["answer"])
```

## 文件说明

- `rag_system.py`: RAG 系统核心实现
- `example.py`: 使用示例
- `documents/`: 文档目录
- `README.md`: 本文件

## 注意事项

- **Ollama**: 需要先安装并运行 Ollama 服务，下载相应的模型
- **API Key**: 使用硅基流动或 OpenAI 时需要有效的 API Key
- **重排序**: 启用重排序需要设置 `COHERE_API_KEY`，可以显著提高检索质量
- 首次运行会创建向量存储，可能需要一些时间
- 可以保存向量存储以便后续快速加载
- 系统会自动检测环境变量，优先级：Ollama > 硅基流动 > OpenAI
- 使用 Ollama 时，确保模型已下载（使用 `ollama pull <model_name>`）

## 重排序功能说明

重排序（Reranking）是 RAG 系统的重要优化技术：

1. **工作原理**：
   - 首先使用向量检索获取更多候选文档（通常为 k*2 个）
   - 然后使用 Cohere Rerank 模型对文档进行重新排序
   - 保留最相关的 top_n 个文档用于生成答案

2. **优势**：
   - 提高检索精度，过滤掉语义相似但相关性低的文档
   - 多维度评估文档与问题的相关性
   - 显著提升最终答案的质量

3. **使用建议**：
   - 对于大型文档库，强烈建议启用重排序
   - 设置 `k=4-8`，`rerank_top_n=3-5` 通常效果较好
   - 如果未设置 `COHERE_API_KEY`，系统会自动跳过重排序