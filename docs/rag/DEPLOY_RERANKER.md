# 使用本地 Qwen Reranker 部署指南

本指南介绍如何将下载好的 Qwen Reranker 模型集成到 RAG 系统中。

## 前置条件

1. **已下载 Qwen Reranker 模型**
   ```bash
   # 如果还没有下载，执行：
   huggingface-cli download --resume-download Qwen/Qwen3-Reranker-0.6B --local-dir ./Qwen/Qwen3-Reranker-0.6B
   ```

2. **已安装 Ollama 并下载相关模型**
   ```bash
   ollama pull qwen3:0.6b
   ollama pull qwen3-embedding:0.6b
   ```

## 安装依赖

```bash
# 使用 uv 安装依赖（推荐）
cd /home/lab/Projects/hylreg_LLM
uv sync

# 或使用 pip
pip install FlagEmbedding torch
```

## 使用方法

### 方式1：在代码中指定 Reranker 路径

```python
from demo.RAG.rag_system import IntelligentRAG

rag = IntelligentRAG(
    documents_path="./documents",
    embedding_model="qwen3-embedding:0.6b",
    llm_model="qwen3:0.6b",
    reranker_model_path="./Qwen/Qwen3-Reranker-0.6B",  # 本地模型路径
)

rag.build(k=4, use_rerank=True, rerank_top_n=3)
result = rag.query("你的问题")
```

### 方式2：使用示例文件

```bash
# 确保模型路径正确
# 默认路径：./Qwen/Qwen3-Reranker-0.6B

# 运行示例
cd /home/lab/Projects/hylreg_LLM
export USE_OLLAMA="true"
uv run python demo/RAG/example_local_reranker.py
```

## 模型路径说明

- **相对路径**：`./Qwen/Qwen3-Reranker-0.6B`（相对于运行脚本的目录）
- **绝对路径**：`/home/lab/Projects/hylreg_LLM/Qwen/Qwen3-Reranker-0.6B`
- **Hugging Face Hub 路径**：如果模型已缓存，也可以使用 `Qwen/Qwen3-Reranker-0.6B`

## 优先级说明

系统会按以下优先级选择重排序方式：

1. **Ollama Reranker**（如果 `ollama_reranker_model` 已设置）
2. **本地 Reranker**（如果 `reranker_model_path` 已设置）
3. **基础检索器**（不使用重排序）

**注意**：如果同时设置了多个 Reranker，系统会按优先级顺序选择第一个可用的。

## 优势

- ✅ **完全本地化**：无需 API Key，可离线使用
- ✅ **中文支持**：Qwen 系列对中文友好
- ✅ **快速响应**：0.6B 参数量，推理速度快
- ✅ **成本节约**：无需支付 API 费用

## 故障排查

### 1. 模型加载失败

**错误信息**：`加载本地 Reranker 模型失败`

**解决方案**：
- 检查模型路径是否正确
- 确认模型文件完整（包含 `config.json`、`model.safetensors` 等）
- 检查是否有足够的磁盘空间和内存

### 2. FlagEmbedding 未安装

**错误信息**：`ModuleNotFoundError: No module named 'FlagEmbedding'`

**解决方案**：
```bash
uv sync
# 或
pip install FlagEmbedding
```

### 3. CUDA/GPU 相关问题

如果使用 CPU 运行，FlagReranker 会自动使用 CPU。如果需要 GPU 加速：
- 确保已安装 PyTorch GPU 版本
- 确保 CUDA 驱动正确安装

## 性能优化建议

1. **使用 FP16**：代码中已启用 `use_fp16=True`，可减少内存占用
2. **批量处理**：Reranker 会自动批量处理文档对
3. **调整 top_n**：根据实际需求调整 `rerank_top_n` 参数

## 与其他 Reranker 对比

| 特性 | Ollama Reranker | 本地 Qwen Reranker |
|------|----------------|-------------------|
| 成本 | 免费 | 免费 |
| 延迟 | 本地，低延迟 | 本地，低延迟 |
| 离线使用 | ✅ 支持 | ✅ 支持 |
| 中文支持 | ✅ 优秀 | ✅ 优秀 |
| 模型大小 | 通过 Ollama 管理 | 0.6B（约 2GB） |
| 设置难度 | ⭐ 简单（只需 pull 模型） | ⭐⭐ 中等（需下载模型） |
| 推荐场景 | 已使用 Ollama | 需要完全离线控制 |

**推荐**：如果已使用 Ollama，优先使用 Ollama Reranker，设置更简单。

## 下一步

- 尝试不同的 `rerank_top_n` 值，找到最佳平衡点
- 对比使用和不使用重排序的效果
- 考虑使用更大的 Reranker 模型以获得更好的效果
