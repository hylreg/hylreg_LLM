# 简单 RAG 系统

这是一个使用 LangChain 实现的简单 RAG (Retrieval-Augmented Generation) 系统。

## 功能特性

- 文档加载和预处理
- 文档分割
- 向量存储和检索
- **重排序（Reranking）**：支持 Ollama Reranker 或 Cohere Rerank 提高检索精度
- 基于检索的问答生成
- 支持多种 LLM：Ollama、硅基流动、OpenAI

## 安装依赖

```bash
# 使用 uv 安装依赖（推荐，更快更可靠）
uv sync

# 或使用 pip
pip install -e .
```

**关于 uv**：
- `uv` 是一个极快的 Python 包管理器和项目管理工具
- 使用 `uv run python <script>.py` 可以自动管理虚拟环境并运行脚本
- 无需手动激活虚拟环境，uv 会自动处理
- 安装 uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **重要**：运行 Python 脚本时需要使用 `uv run python <script>.py`，不能直接使用 `uv run <script>.py`

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
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0  # Reranker 模型（可选，但推荐）
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
- 默认 Reranker 模型：`dengcao/Qwen3-Reranker-0.6B:Q8_0`（推荐）
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

重排序可以显著提高检索质量。有两种方式：

**方式A: 使用 Ollama Reranker（推荐，如果使用 Ollama）**

```bash
# 下载 Reranker 模型
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
```

在代码中设置：
```python
rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
    ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0",  # Ollama Reranker
)
```

**方式B: 使用 Cohere Rerank API**

```bash
export COHERE_API_KEY="your-cohere-api-key"
```

或在 `.env` 文件中配置：

```env
COHERE_API_KEY=your-cohere-api-key
```

**注意**: 
- 优先级顺序为：Ollama Reranker > 本地 Reranker > Cohere API
- 如果使用 Ollama，推荐使用 Ollama Reranker（无需 API Key）
- 重排序功能需要在 `build()` 方法中设置 `use_rerank=True`

## 快速开始

### 完整运行流程

```bash
# 1. 进入项目目录
cd /home/lab/Projects/hylreg_LLM

# 2. 安装依赖
cd /home/lab/Projects/hylreg_LLM
uv sync

# 验证依赖安装（可选）
uv run python -c "import langchain; print('LangChain version:', langchain.__version__)"

# 或使用 pip（备选方案）
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

# 重要：使用 uv run 时，建议从项目根目录运行以确保依赖正确加载
# 方式1：从项目根目录运行（推荐）
cd /home/lab/Projects/hylreg_LLM
uv run python RAG/example.py
uv run python RAG/example_ollama.py

# 方式2：进入 RAG 目录后运行（需要先确保依赖已安装）
cd RAG
uv run python example.py
uv run python example_ollama.py

# 方式3：使用传统 Python 方式（需要先激活虚拟环境）
cd RAG
python example.py
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

**使用 uv 运行（推荐）**：
```bash
# 重要：使用 uv run 时，建议从项目根目录运行
# 方式1：从项目根目录运行（推荐）
cd /home/lab/Projects/hylreg_LLM
uv run python RAG/example.py

# Ollama 示例（使用本地 Ollama 模型）
export USE_OLLAMA="true"
uv run python RAG/example_ollama.py

# 方式2：进入 RAG 目录后运行（需要先确保依赖已安装）
cd RAG
uv run python example.py
export USE_OLLAMA="true"
uv run python example_ollama.py
```

**使用传统 Python 方式运行**：
```bash
cd RAG

# 基础示例
python example.py

# Ollama 示例
export USE_OLLAMA="true"
python example_ollama.py
```

**使用 Python 模块方式运行**：
```bash
# 从项目根目录运行
uv run -m RAG.example
uv run -m RAG.example_ollama

# 或使用传统方式
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
    ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0",  # Ollama Reranker 模型
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

**使用 uv 运行**：
```bash
# 测试 Ollama 配置
export USE_OLLAMA="true"
cd RAG && uv run python example_ollama.py

# 测试硅基流动配置
unset USE_OLLAMA
export SILICONFLOW_API_KEY="your-key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1/"
cd RAG && uv run python example.py

# 测试 OpenAI 配置
unset USE_OLLAMA
unset SILICONFLOW_API_KEY
export OPENAI_API_KEY="your-key"
cd RAG && uv run python example.py
```

**使用传统 Python 方式**：
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

**使用 uv 运行**：
```bash
# 使用 uv 运行 Python 交互式环境
cd RAG
uv run python -i example.py

# 或使用 uv 运行并进入交互式调试
uv run python -i example_ollama.py
```

**使用传统 Python 方式**：
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

**1. uv run 权限错误**
```bash
# 错误：uv run example.py
# 错误信息：Permission denied (os error 13)

# 正确方式：需要通过 Python 解释器运行
uv run python example.py
uv run python example_ollama.py

# 或者使用模块方式
uv run -m RAG.example
uv run -m RAG.example_ollama
```

**2. Ollama 连接失败**
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

**4. 模块未找到错误（ModuleNotFoundError）**
```bash
# 错误信息：ModuleNotFoundError: No module named 'langchain.text_splitter'

# ⚡ 快速修复（推荐先尝试这个）
cd /home/lab/Projects/hylreg_LLM
uv sync --refresh
uv run python -c "from langchain.text_splitter import RecursiveCharacterTextSplitter; print('✓ LangChain 导入成功')"

# 如果快速修复失败，按以下步骤排查：

# 步骤1：检查当前目录和 Python 环境
cd /home/lab/Projects/hylreg_LLM
pwd  # 确保在项目根目录

# 步骤2：验证 uv 虚拟环境
uv venv --help  # 检查 uv 是否正常工作

# 步骤3：强制重新同步依赖（清除缓存）
uv sync --refresh

# 步骤4：验证依赖是否已安装
uv run python -c "import langchain; print('LangChain version:', langchain.__version__)"
uv run python -c "from langchain.text_splitter import RecursiveCharacterTextSplitter; print('✓ 导入成功')"

# 步骤5：如果步骤4失败，手动安装 langchain
uv pip install langchain langchain-community langchain-core langchain-openai langchain-ollama langchain-cohere

# 步骤6：检查已安装的包
uv pip list | grep -i langchain

# 步骤7：如果仍然失败，尝试使用 pip 安装（作为备选）
uv run pip install langchain langchain-community langchain-core langchain-openai langchain-ollama langchain-cohere

# 步骤8：验证安装后再次运行
uv run python RAG/example_ollama.py
```

**5. 依赖安装问题**
```bash
# 重新安装依赖
cd /home/lab/Projects/hylreg_LLM
uv sync

# 或使用 pip（如果使用传统方式）
pip install --upgrade -e .

# 检查已安装的包
uv pip list
# 或
pip list
```

**6. 文档加载失败**
```bash
# 检查文档目录是否存在
ls -la RAG/documents/

# 检查文档编码（确保是 UTF-8）
file RAG/documents/*.txt
```

## 注意事项

- **Ollama**: 需要先安装并运行 Ollama 服务，下载相应的模型
- **API Key**: 使用硅基流动或 OpenAI 时需要有效的 API Key
- **重排序**: 如果使用 Ollama，推荐使用 Ollama Reranker（无需 API Key）；也可以使用 Cohere API（需要设置 `COHERE_API_KEY`）。重排序可以显著提高检索质量
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