# 向量存储说明

## 当前实现：FAISS（本地文件系统向量数据库）

### 什么是 FAISS？

**FAISS (Facebook AI Similarity Search)** 是一个**本地向量数据库库**，由 Facebook AI Research 开发。它：

- ✅ **不需要额外的数据库服务**：直接在本地文件系统存储向量数据
- ✅ **高性能**：针对相似度搜索进行了优化
- ✅ **轻量级**：无需安装和配置数据库服务器
- ✅ **适合中小规模数据**：适合文档数量在百万级以下的应用

### 向量存储保存在哪里？

向量存储保存在**本地文件系统的指定目录**中。当你调用 `build()` 方法时：

```python
rag.build(
    vectorstore_path="./vectorstore"  # 向量存储保存在这个目录
)
```

保存的目录结构如下：
```
vectorstore/
├── index.faiss      # FAISS 索引文件（包含向量数据）
└── index.pkl        # 元数据文件（包含文档内容和元信息）
```

### 向量存储位置示例

根据示例代码，向量存储默认保存在：
- `demo/RAG/vectorstore/` - 相对于示例脚本的目录

你可以通过以下方式查看：
```bash
ls -la demo/RAG/vectorstore/
```

## 是否需要额外的向量数据库？

### 当前方案（FAISS）：**不需要**

✅ **优点**：
- 无需安装和配置数据库服务器
- 无需网络连接
- 启动快速，适合本地开发和小规模部署
- 数据完全在本地，隐私性好

❌ **限制**：
- 不适合大规模数据（百万级以上）
- 不支持分布式部署
- 不支持多用户并发访问
- 需要手动管理数据备份

### 何时需要其他向量数据库？

如果你需要以下功能，可以考虑切换到其他向量数据库：

#### 1. **大规模数据**（百万级以上文档）
- **Pinecone**：云端托管，自动扩展
- **Weaviate**：开源，支持大规模数据
- **Milvus**：开源，专为大规模向量搜索设计

#### 2. **分布式部署**
- **Milvus**：支持分布式集群
- **Weaviate Cloud**：云端托管，自动扩展

#### 3. **多用户/多应用共享**
- **Pinecone**：云端服务，支持多应用共享
- **Weaviate Cloud**：多租户支持

#### 4. **实时更新和增量索引**
- **Pinecone**：支持实时更新
- **Weaviate**：支持增量更新

## 如何切换到其他向量数据库？

### 方案 1：使用 Pinecone（云端托管）

```python
from langchain_community.vectorstores import Pinecone
import pinecone

# 初始化 Pinecone
pinecone.init(api_key="your-api-key", environment="us-east1-gcp")

# 创建向量存储
vectorstore = Pinecone.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name="my-index"
)
```

### 方案 2：使用 Weaviate（开源/云端）

```python
from langchain_community.vectorstores import Weaviate

# 连接到 Weaviate
vectorstore = Weaviate.from_documents(
    documents=splits,
    embedding=embeddings,
    weaviate_url="http://localhost:8080"
)
```

### 方案 3：使用 Milvus（开源，大规模）

```python
from langchain_community.vectorstores import Milvus

# 连接到 Milvus
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"}
)
```

## 推荐方案

### 对于大多数应用：**继续使用 FAISS**

如果你的应用场景是：
- 📄 文档数量在 10 万以下
- 🏠 单机部署
- 👤 单用户或小团队使用
- 🔒 需要数据隐私（本地存储）

**FAISS 是最佳选择**，无需额外的数据库服务。

### 何时考虑切换：

- 📊 文档数量超过 100 万
- 🌐 需要多用户/多应用共享
- 📈 需要分布式部署
- 🔄 需要实时更新和增量索引

## 总结

- ✅ **当前使用 FAISS**：本地文件系统向量数据库，**不需要额外的数据库服务**
- 📁 **向量存储位置**：保存在 `vectorstore_path` 指定的本地目录
- 🎯 **适合场景**：中小规模应用，单机部署，数据隐私要求高
- 🔄 **可扩展性**：代码支持切换到其他向量数据库（Pinecone、Weaviate、Milvus 等）
