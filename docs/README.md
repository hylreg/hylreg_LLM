# 项目文档索引

欢迎查阅项目文档！本文档目录提供了所有相关文档的索引和导航。

## 📚 文档目录

### 快速开始
- [项目主 README](../README.md) - 项目概览和快速入门指南
- [RAG 系统文档](../RAG/README.md) - RAG 系统的详细使用说明

### RAG 系统文档
- [RAG 系统使用指南](../RAG/README.md) - RAG 系统完整使用说明
- [Reranker 部署指南](rag/DEPLOY_RERANKER.md) - 本地 Qwen Reranker 部署说明
- [性能测试文档](rag/BENCHMARK.md) - RAG 系统性能测试指南
- [向量存储指南](rag/VECTOR_STORE.md) - 向量存储使用说明

### 工具文档
- [量化使用指南](量化使用指南.md) - llama.cpp 模型量化使用指南
- [魔搭模型下载指南](魔搭模型下载指南.md) - ModelScope 模型下载使用指南
- [Ollama Reranker 部署指南](OLLAMA_RERANKER_DEPLOY.md) - Ollama Reranker 部署说明

### 示例代码
- [RAG 示例代码](../examples/rag/) - RAG 系统各种使用示例
  - [基础示例](../examples/rag/example.py) - 使用 API 服务的示例
  - [Ollama 示例](../examples/rag/example_ollama.py) - 使用 Ollama 的示例代码
  - [ModelScope 示例](../examples/rag/example_modelscope.py) - 使用魔搭 ModelScope 的示例
- [智能体示例代码](../examples/agents/) - 智能体系统使用示例

## 📖 文档结构

```
hylreg_LLM/
├── README.md                    # 项目主文档
├── docs/                        # 文档目录（本文件夹）
│   ├── README.md               # 文档索引（本文件）
│   ├── rag/                    # RAG 相关文档
│   │   ├── DEPLOY_RERANKER.md
│   │   ├── BENCHMARK.md
│   │   └── VECTOR_STORE.md
│   ├── 魔搭模型下载指南.md
│   └── ...
├── RAG/                         # RAG 系统核心代码
│   └── README.md               # RAG 系统详细文档
└── examples/                    # 示例代码目录
    ├── rag/                    # RAG 示例
    └── agents/                 # 智能体示例
```

## 🔍 快速查找

根据您的需求，可以查看以下文档：

- **想要快速开始？** → 查看 [项目主 README](../README.md)
- **需要了解 RAG 系统？** → 查看 [RAG 系统文档](../RAG/README.md)
- **需要部署本地 Reranker？** → 查看 [Reranker 部署指南](rag/DEPLOY_RERANKER.md)
- **需要代码示例？** → 查看 [examples 目录](../examples/)
- **需要下载模型？** → 查看 [魔搭模型下载指南](魔搭模型下载指南.md)

## 📝 添加新文档

如果您需要添加新的文档，请将其放在 `docs/` 目录下，并在此索引文件中添加相应的链接和说明。
