# 使用 Ollama 部署 Qwen3-Reranker-0.6B 指南

本指南介绍如何将 ModelScope 下载的 Qwen3-Reranker-0.6B 模型用于 Ollama 部署。

## 方案对比

### 方案 1：直接使用本地模型路径（推荐，最简单）

**优点**：
- ✅ 无需转换模型格式
- ✅ 设置简单，直接指定路径即可
- ✅ 完全离线运行

**缺点**：
- ❌ 不能通过 Ollama API 直接调用
- ❌ 需要 Python 环境运行

**使用方法**：

```python
from demo.RAG.rag_system import IntelligentRAG

rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
    reranker_model_path="/home/lab/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0.6B",
)

rag.build(k=4, use_rerank=True, rerank_top_n=3)
```

**运行示例**：

```bash
cd /home/lab/Projects/hylreg_LLM
uv run python demo/RAG/example_modelscope_reranker.py
```

### 方案 2：使用 Ollama 社区模型（推荐，最简单）

**优点**：
- ✅ 设置最简单，只需 `ollama pull`
- ✅ 可以通过 Ollama API 直接调用
- ✅ 模型已优化为 GGUF 格式

**缺点**：
- ❌ 需要重新下载模型（但已优化）

**使用方法**：

```bash
# 下载 Ollama 格式的 Reranker 模型
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
```

```python
from demo.RAG.rag_system import IntelligentRAG

rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
    ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0",
)

rag.build(k=4, use_rerank=True, rerank_top_n=3)
```

### 方案 3：将 PyTorch 模型转换为 GGUF 并导入 Ollama（高级）

**适用场景**：
- 需要完全自定义模型
- 需要特定量化格式
- 已有 PyTorch 模型但想用 Ollama 管理

**步骤**：

1. **安装转换工具**：

```bash
# 安装 llama.cpp（用于模型转换）
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# 安装 Python 依赖
pip install -r requirements.txt
```

2. **转换模型**：

```bash
# 使用 transformers 将 PyTorch 模型转换为 GGUF
python convert-hf-to-gguf.py \
    /home/lab/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0.6B \
    --outdir ./qwen3-reranker-gguf \
    --outtype f16
```

3. **量化模型**（可选，减小模型大小）：

```bash
# 使用 Q8_0 量化（推荐，平衡大小和精度）
./quantize ./qwen3-reranker-gguf/ggml-model-f16.gguf \
    ./qwen3-reranker-gguf/ggml-model-q8_0.gguf q8_0
```

4. **创建 Modelfile**：

```dockerfile
FROM ./qwen3-reranker-gguf/ggml-model-q8_0.gguf

TEMPLATE """{{ .Prompt }}"""

PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER num_predict 1
```

5. **导入到 Ollama**：

```bash
ollama create qwen3-reranker-0.6b -f Modelfile
```

6. **使用模型**：

```python
from demo.RAG.rag_system import IntelligentRAG

rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
    ollama_reranker_model="qwen3-reranker-0.6b",
)

rag.build(k=4, use_rerank=True, rerank_top_n=3)
```

## 推荐方案

**对于大多数用户**：推荐使用**方案 2**（Ollama 社区模型）

- 设置最简单
- 模型已优化
- 社区维护，稳定可靠

**如果已下载 ModelScope 模型**：推荐使用**方案 1**（本地模型路径）

- 无需重新下载
- 直接使用现有模型
- 设置简单

**如果需要完全自定义**：使用**方案 3**（转换并导入）

- 完全控制模型格式
- 可以自定义量化方式
- 适合高级用户

## 性能对比

| 方案 | 延迟 | 内存占用 | 模型大小 | 设置难度 |
|------|------|---------|---------|---------|
| 方案 1（本地路径） | 低 | 中等 | ~1.1 GB | ⭐ 简单 |
| 方案 2（Ollama） | 低 | 中等 | ~0.6 GB (Q8_0) | ⭐ 简单 |
| 方案 3（转换导入） | 低 | 中等 | 可自定义 | ⭐⭐⭐ 复杂 |

## 故障排查

### 方案 1 问题

**问题**：模型加载失败

**解决**：
- 检查模型路径是否正确
- 确认模型文件完整（包含 `config.json`、`model.safetensors` 等）
- 检查是否安装了 `FlagEmbedding`：`uv pip install FlagEmbedding`

### 方案 2 问题

**问题**：Ollama 模型不存在

**解决**：
```bash
# 检查模型是否已下载
ollama list

# 如果不存在，重新下载
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
```

### 方案 3 问题

**问题**：转换失败

**解决**：
- 确保模型是 Hugging Face 格式
- 检查转换工具版本
- 查看转换日志中的错误信息

## 下一步

选择适合您的方案后：

1. 按照对应方案的步骤操作
2. 运行示例脚本测试
3. 根据实际需求调整参数（`k`、`rerank_top_n` 等）
4. 对比不同方案的效果和性能