# RAG 系统使用指南

基于 LangChain 实现的简单 RAG (Retrieval-Augmented Generation) 系统，支持多种 LLM 和嵌入模型。

## 功能特性

- 📄 文档加载和预处理
- 🔍 向量存储和检索（基于 FAISS）
- 🎯 **重排序（Reranking）**：支持多种方式提高检索精度
  - Ollama Reranker（推荐，本地运行）
  - 本地 Qwen Reranker（完全离线）
- 🤖 支持多种 LLM：Ollama、硅基流动、魔搭 ModelScope

## 安装依赖

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
```

**关于 uv**：`uv` 是一个极快的 Python 包管理器。使用 `uv run python <script>.py` 可以自动管理虚拟环境。安装 uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 环境配置

系统支持多种 LLM 和重排序方式，按以下优先级自动选择：

| 配置方式 | 环境变量 | 说明 |
|---------|---------|------|
| **Ollama（推荐）** | `USE_OLLAMA=true` | 本地运行，无需 API Key |
| 硅基流动 API | `SILICONFLOW_API_KEY`<br>`SILICONFLOW_BASE_URL` | 需要 API Key |
| 魔搭 ModelScope | `USE_MODELSCOPE=true` | 本地模型，需要下载模型 |

### 使用 Ollama（推荐）

```bash
# 1. 安装 Ollama（如果还没有）
# 访问 https://ollama.ai 下载安装

# 2. 下载需要的模型
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0  # Reranker（可选，但推荐）

# 3. 设置环境变量
export USE_OLLAMA="true"
# 或在 .env 文件中配置：
# USE_OLLAMA=true
# OLLAMA_BASE_URL=http://localhost:11434  # 默认地址，可省略
```

### 使用 API 服务

```bash
# 硅基流动
export SILICONFLOW_API_KEY="your-api-key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1/"

# 或魔搭 ModelScope（本地模型）
export USE_MODELSCOPE="true"
```

### 重排序配置

重排序优先级：**Ollama Reranker > 本地 Reranker**

- **Ollama Reranker**（推荐）：如果使用 Ollama，在代码中设置 `ollama_reranker_model` 参数即可
- **本地 Reranker**：需要下载模型，详见 [Reranker 部署指南](DEPLOY_RERANKER.md)

## 快速开始

### 1. 准备文档

将文档放在 `documents/` 目录下（支持 `.txt` 文件）：

```bash
# 在仓库根目录下
mkdir -p demo/RAG/documents
cp your_document.txt demo/RAG/documents/
```

### 2. 运行示例

```bash
# 从项目根目录运行（推荐）
cd /home/lab/Projects/hylreg_LLM

# 使用 Ollama（推荐）
export USE_OLLAMA="true"
uv run python demo/RAG/example_ollama.py

# 或使用其他 API
uv run python demo/RAG/example.py

# 使用魔搭 ModelScope
export USE_MODELSCOPE="true"
uv run python demo/RAG/example_modelscope.py
```

## 使用方法

### 基本使用

**使用 Ollama**：
```python
from demo.RAG.rag_system import IntelligentRAG

# 初始化 RAG 系统
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
    ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0",  # Ollama Reranker
)

# 构建系统（启用重排序）
rag.build(k=4, use_rerank=True, rerank_top_n=3)

# 查询
result = rag.query("你的问题")
print(result["answer"])
```

**使用 API 服务**：
```python
from demo.RAG.rag_system import IntelligentRAG

# 初始化 RAG 系统（需要先设置相应的环境变量）
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
)

# 构建系统（启用重排序，需要配置 Ollama 或本地 Reranker）
rag.build(k=4, use_rerank=True, rerank_top_n=3)

# 查询
result = rag.query("你的问题")
print(result["answer"])
```

**使用魔搭 ModelScope**：
```python
from demo.RAG.rag_system import IntelligentRAG
import os

# 设置使用 ModelScope
os.environ["USE_MODELSCOPE"] = "true"

# 初始化 RAG 系统
# embedding_model 和 llm_model 应该是 ModelScope 模型路径
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="damo/nlp_gte_sentence-embedding_chinese-base",  # ModelScope 嵌入模型
    llm_model="qwen/Qwen-7B-Chat",  # ModelScope LLM 模型
)

# 构建系统
rag.build(k=4, use_rerank=True, rerank_top_n=3)

# 查询
result = rag.query("你的问题")
print(result["answer"])
```

### 参数说明

- `k`: 初始检索的文档块数量（重排序前会检索 `k*2` 个文档）
- `use_rerank`: 是否启用重排序（推荐启用）
- `rerank_top_n`: 重排序后保留的文档数量（通常设置为 3-5）

## 重排序功能

重排序（Reranking）可以显著提高检索质量：

1. **工作原理**：
   - 首先使用向量检索获取更多候选文档（`k*2` 个）
   - 然后使用 Reranker 模型对文档进行重新排序
   - 保留最相关的 `top_n` 个文档用于生成答案

2. **优势**：
   - 提高检索精度，过滤掉语义相似但相关性低的文档
   - 多维度评估文档与问题的相关性
   - 显著提升最终答案的质量

3. **使用建议**：
   - 对于大型文档库，强烈建议启用重排序
   - 设置 `k=4-8`，`rerank_top_n=3-5` 通常效果较好

## 文件说明

- `rag_system.py`: RAG 系统核心实现
- 本目录下的示例与脚本：
  - `example.py`: 使用示例（默认配置）
  - `example_ollama.py`: Ollama 专用示例
  - `example_modelscope.py`: 魔搭 ModelScope 示例
  - `example_local_reranker.py`: 本地 Reranker 示例
  - `example_ollama_reranker.py`: Ollama Reranker 示例
  - `example_modelscope_reranker.py`: ModelScope Reranker 示例
  - `benchmark.py`: 性能测试脚本
- `documents/`: 文档目录
- `DEPLOY_RERANKER.md`: 本地 Reranker 部署指南

## 常见问题

### 1. uv run 权限错误

```bash
# 错误：uv run example.py
# 正确：uv run python demo/RAG/example.py
# 或：从项目根目录运行
cd /home/lab/Projects/hylreg_LLM
uv run python demo/RAG/example.py
```

### 2. Ollama 连接失败

```bash
# 检查 Ollama 服务是否运行
curl http://localhost:11434/api/tags

# 如果失败，启动 Ollama 服务
ollama serve

# 检查环境变量
echo $USE_OLLAMA
```

### 3. 模型未找到

```bash
# 检查已下载的模型
ollama list

# 下载缺失的模型
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
```

### 4. 模块未找到错误

```bash
# 重新同步依赖
cd /home/lab/Projects/hylreg_LLM
uv sync --refresh

# 验证安装
uv run python -c "import langchain; print('✓ LangChain 导入成功')"
```

### 5. API Key 错误

```bash
# 检查环境变量是否正确设置
env | grep -E "(SILICONFLOW|OLLAMA)"
```

## 注意事项

- **Ollama**: 需要先安装并运行 Ollama 服务，下载相应的模型
- **API Key**: 使用硅基流动时需要有效的 API Key
- **重排序**: 如果使用 Ollama，推荐使用 Ollama Reranker（无需 API Key）
- 首次运行会创建向量存储，可能需要一些时间
- 系统会自动检测环境变量，优先级：Ollama > 硅基流动 > 魔搭 ModelScope
- 如果遇到问题，查看控制台输出的错误信息，系统会显示使用的 API 和模型信息

## 更多文档

- [本地 Reranker 部署指南](../docs/rag/DEPLOY_RERANKER.md) - 使用本地 Qwen Reranker 的详细说明
- [性能测试文档](../docs/rag/BENCHMARK.md) - RAG 系统性能测试指南
- [向量存储指南](../docs/rag/VECTOR_STORE.md) - 向量存储使用说明
- [项目主文档](../README.md) - 项目概览和快速开始
