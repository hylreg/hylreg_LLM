"""
使用 Ollama 的 RAG 系统示例
"""

import os
from rag_system import IntelligentRAG


def main():
    # 设置使用 Ollama
    os.environ["USE_OLLAMA"] = "true"
    # 如果 Ollama 运行在其他地址，可以设置：
    # os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    
    # 初始化 RAG 系统（使用 Ollama 模型）
    # 注意：需要先使用 ollama pull 下载相应的模型
    # 例如：ollama pull llama2
    #      ollama pull nomic-embed-text
    rag = IntelligentRAG(
        documents_path="./documents",  # 文档目录路径
        embedding_model="nomic-embed-text",  # Ollama 嵌入模型
        llm_model="llama2",  # Ollama LLM 模型，可以改为 llama3, mistral, qwen 等
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # 构建 RAG 系统
    rag.build(k=4)  # k 表示检索的文档块数量
    
    # 可选：保存向量存储以便后续使用
    # rag.save_vectorstore("./vectorstore")
    
    # 进行查询
    questions = [
        "文档的主要内容是什么？",
        "请总结一下关键信息",
        "RAG 系统的主要优势有哪些？",
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