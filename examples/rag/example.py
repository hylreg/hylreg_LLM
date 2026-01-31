"""
RAG 系统使用示例

本示例展示如何使用 RAG 系统，支持多种后端配置。
根据环境变量自动选择后端（Ollama、硅基流动、魔搭 ModelScope）。

使用前请设置相应的环境变量：
- Ollama: USE_OLLAMA=true
- 硅基流动: SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL
- 魔搭 ModelScope: USE_MODELSCOPE=true
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.RAG.rag_system import IntelligentRAG


def main():
    # 设置 API Key（如果还没有设置环境变量）
    # 方式1: 使用 Ollama（本地模型）
    # os.environ["USE_OLLAMA"] = "true"
    # os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"  # 默认地址，可省略
    
    # 方式2: 使用硅基流动 API
    # os.environ["SILICONFLOW_API_KEY"] = "your-siliconflow-api-key"
    # os.environ["SILICONFLOW_BASE_URL"] = "https://api.siliconflow.cn/v1/"
    
    # 方式3: 使用魔搭 ModelScope（本地模型）
    # os.environ["USE_MODELSCOPE"] = "true"
    # 注意：使用 ModelScope 时，embedding_model 和 llm_model 应该是 ModelScope 模型路径
    # 例如：embedding_model="damo/nlp_gte_sentence-embedding_chinese-base"
    #      llm_model="qwen/Qwen-7B-Chat"
    
    # 方式4: 启用重排序
    # 如果使用 Ollama，会自动使用 Ollama Reranker（无需 API Key）
    # 如果不使用 Ollama，可以配置本地 Reranker（详见 DEPLOY_RERANKER.md）
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    documents_path = project_root / "demo" / "RAG" / "documents"
    
    # 初始化 RAG 系统
    # 如果使用 Ollama，需要设置相应的模型名称，例如：
    # embedding_model="qwen3-embedding:0.6b"  # Ollama 嵌入模型
    # llm_model="qwen3:0.6b"  # Ollama LLM 模型
    # 如果使用 Ollama，设置环境变量
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    
    rag = IntelligentRAG(
        documents_path=str(documents_path),  # 文档目录路径
        embedding_model="qwen3-embedding:0.6b",  # 默认使用 Qwen3 嵌入模型
        llm_model="qwen3:0.6b",  # 默认使用 Qwen3 LLM 模型
        chunk_size=1000,
        chunk_overlap=200,
        ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0" if use_ollama else None,  # 如果使用 Ollama，则使用 Ollama Reranker
    )
    
    # 构建 RAG 系统
    # k: 初始检索的文档块数量（重排序前会检索 k*2 个文档）
    # use_rerank: 是否启用重排序（如果使用 Ollama，则使用 Ollama Reranker；否则需要配置本地 Reranker）
    # rerank_top_n: 重排序后保留的文档数量
    # vectorstore_path: 向量存储保存路径（如果路径存在会自动加载，避免重复处理文档）
    # force_rebuild: 是否强制重新构建（默认 False，如果向量存储存在则直接加载）
    vectorstore_path = project_root / "demo" / "RAG" / "vectorstore"
    rag.build(
        k=4, 
        use_rerank=True, 
        rerank_top_n=3,
        vectorstore_path=str(vectorstore_path),  # 自动保存和加载向量存储
        force_rebuild=False  # 设置为 True 可强制重新构建
    )
    
    # 进行查询
    questions = [
        "文档的主要内容是什么？",
        "请总结一下关键信息",
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 50)
        result = rag.query(question)
        print(f"回答: {result['answer']}")
        print(f"\n参考文档数量: {len(result['source_documents'])}")
        print("=" * 50)


if __name__ == "__main__":
    main()