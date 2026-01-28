"""
RAG 系统使用示例
"""

import os
from rag_system import IntelligentRAG


def main():
    # 设置 API Key（如果还没有设置环境变量）
    # 方式1: 使用 Ollama（本地模型）
    # os.environ["USE_OLLAMA"] = "true"
    # os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"  # 默认地址，可省略
    
    # 方式2: 使用硅基流动 API
    # os.environ["SILICONFLOW_API_KEY"] = "your-siliconflow-api-key"
    # os.environ["SILICONFLOW_BASE_URL"] = "https://api.siliconflow.cn/v1/"
    
    # 方式3: 使用 OpenAI API
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    # 方式4: 启用重排序（需要 Cohere API Key）
    # os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
    
    # 初始化 RAG 系统
    # 如果使用 Ollama，需要设置相应的模型名称，例如：
    # embedding_model="qwen3-embedding:0.6b"  # Ollama 嵌入模型
    # llm_model="qwen3:0.6b"  # Ollama LLM 模型
    rag = IntelligentRAG(
        documents_path="./documents",  # 文档目录路径
        embedding_model="qwen3-embedding:0.6b",  # 默认使用 Qwen3 嵌入模型
        llm_model="qwen3:0.6b",  # 默认使用 Qwen3 LLM 模型
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # 构建 RAG 系统
    # k: 初始检索的文档块数量（重排序前会检索 k*2 个文档）
    # use_rerank: 是否启用重排序（需要设置 COHERE_API_KEY）
    # rerank_top_n: 重排序后保留的文档数量
    rag.build(k=4, use_rerank=True, rerank_top_n=3)
    
    # 可选：保存向量存储以便后续使用
    # rag.save_vectorstore("./vectorstore")
    
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