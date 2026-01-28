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
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
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
- 默认嵌入模型：`qwen3-embedding:0.6b`
- 默认 LLM 模型：`qwen3:0.6b`
- 其他推荐模型：`nomic-embed-text`、`llama2`、`llama3`、`mistral` 等

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

## 快速开始

### 完整运行流程

```bash
# 1. 进入项目目录
cd /home/lab/Projects/hylreg_LLM

# 2. 安装依赖
uv sync
# 或使用 pip
pip install -e .

# 3. 设置环境变量（选择一种方式）

# 方式1: 使用 Ollama（推荐）
export USE_OLLAMA="true"
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b

# 方式2: 使用硅基流动 API
export SILICONFLOW_API_KEY="your-siliconflow-api-key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1/"

# 方式3: 使用 OpenAI API
export OPENAI_API_KEY="your-openai-api-key"

# 4. 可选：启用重排序
export COHERE_API_KEY="your-cohere-api-key"

# 5. 准备文档（将文档放在 RAG/documents/ 目录下）
# 示例文档已包含在 RAG/documents/sample.txt

# 6. 运行示例
cd RAG
python example.py

# 或使用 Ollama 专用示例
python example_ollama.py
```

## 使用方法

### 1. 准备文档

将你的文档放在 `documents/` 目录下（支持 `.txt` 文件）。

```bash
# 确保文档目录存在
mkdir -p RAG/documents

# 添加你的文档
cp your_document.txt RAG/documents/
```

### 2. 运行示例

**基础示例**（使用默认配置）：
```bash
cd RAG
python example.py
```

**Ollama 示例**（使用本地 Ollama 模型）：
```bash
cd RAG
# 确保已设置 USE_OLLAMA=true
export USE_OLLAMA="true"
python example_ollama.py
```

**使用 Python 模块方式运行**：
```bash
# 从项目根目录运行
python -m RAG.example
python -m RAG.example_ollama
```

### 3. 在代码中使用

**使用 Ollama**：
```python
from rag_system import IntelligentRAG

# 初始化 RAG 系统（使用 Ollama）
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",  # Ollama 嵌入模型
    llm_model="qwen3:0.6b",  # Ollama LLM 模型
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

# 初始化 RAG 系统（默认使用 Qwen3 模型）
# 如果使用硅基流动或 OpenAI，需要设置相应的环境变量
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",  # 默认使用 Qwen3 嵌入模型
    llm_model="qwen3:0.6b",  # 默认使用 Qwen3 LLM 模型
)

# 构建系统（启用重排序）
rag.build(k=4, use_rerank=True, rerank_top_n=3)

# 查询
result = rag.query("你的问题")
print(result["answer"])
```

## 常用运行命令

### 检查环境配置

```bash
# 检查 Ollama 是否运行
curl http://localhost:11434/api/tags

# 检查已下载的 Ollama 模型
ollama list

# 检查环境变量
echo $USE_OLLAMA
echo $SILICONFLOW_API_KEY
echo $COHERE_API_KEY
```

### 测试不同配置

```bash
# 测试 Ollama 配置
export USE_OLLAMA="true"
cd RAG && python example_ollama.py

# 测试硅基流动配置
unset USE_OLLAMA
export SILICONFLOW_API_KEY="your-key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1/"
cd RAG && python example.py

# 测试 OpenAI 配置
unset USE_OLLAMA
unset SILICONFLOW_API_KEY
export OPENAI_API_KEY="your-key"
cd RAG && python example.py
```

### 调试模式

```bash
# 使用 Python 交互式调试
cd RAG
python -i example.py

# 或在代码中添加调试信息
# 系统会自动打印使用的 API 和模型信息
```

## 文件说明

- `rag_system.py`: RAG 系统核心实现
- `example.py`: 使用示例（默认配置）
- `example_ollama.py`: Ollama 专用示例
- `documents/`: 文档目录
- `README.md`: 本文件

## 故障排查

### 常见问题

**1. Ollama 连接失败**
```bash
# 检查 Ollama 服务是否运行
curl http://localhost:11434/api/tags

# 如果失败，启动 Ollama 服务
ollama serve

# 检查环境变量
echo $USE_OLLAMA
echo $OLLAMA_BASE_URL
```

**2. 模型未找到**
```bash
# 检查已下载的模型
ollama list

# 下载缺失的模型
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
```

**3. API Key 错误**
```bash
# 检查环境变量是否正确设置
env | grep -E "(SILICONFLOW|OPENAI|COHERE|OLLAMA)"

# 验证 API Key 是否有效（以 Cohere 为例）
curl -X POST https://api.cohere.ai/v1/rerank \
  -H "Authorization: Bearer $COHERE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"rerank-multilingual-v3.0","query":"test","documents":["test"]}'
```

**4. 依赖安装问题**
```bash
# 重新安装依赖
pip install --upgrade -e .

# 或使用 uv
uv sync --upgrade
```

**5. 文档加载失败**
```bash
# 检查文档目录是否存在
ls -la RAG/documents/

# 检查文档编码（确保是 UTF-8）
file RAG/documents/*.txt
```

## 注意事项

- **Ollama**: 需要先安装并运行 Ollama 服务，下载相应的模型
- **API Key**: 使用硅基流动或 OpenAI 时需要有效的 API Key
- **重排序**: 启用重排序需要设置 `COHERE_API_KEY`，可以显著提高检索质量
- 首次运行会创建向量存储，可能需要一些时间
- 可以保存向量存储以便后续快速加载
- 系统会自动检测环境变量，优先级：Ollama > 硅基流动 > OpenAI
- 使用 Ollama 时，确保模型已下载（使用 `ollama pull <model_name>`）
- 如果遇到问题，查看控制台输出的错误信息，系统会显示使用的 API 和模型信息

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