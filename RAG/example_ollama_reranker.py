"""
使用 Ollama Reranker 的 RAG 系统示例

本示例展示如何使用 Ollama Reranker 进行重排序。
这是推荐的重排序方式，设置简单，无需下载本地模型。

运行前准备：
1. 安装 Ollama: https://ollama.ai
2. 下载所有需要的模型:
   - ollama pull qwen3:0.6b
   - ollama pull qwen3-embedding:0.6b
   - ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
3. 确保 Ollama 服务正在运行

优势：
- 无需 API Key
- 本地运行，延迟低
- 设置简单，只需 pull 模型
"""

import os
from pathlib import Path
from rag_system import IntelligentRAG


def main():
    # 设置使用 Ollama
    os.environ["USE_OLLAMA"] = "true"
    # 如果 Ollama 运行在其他地址，可以设置：
    # os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    
    # 获取脚本所在目录，确保路径正确
    script_dir = Path(__file__).parent
    documents_path = script_dir / "documents"
    
    # 初始化 RAG 系统（使用 Ollama 模型 + Ollama Reranker）
    # 注意：
    # 1. 需要先使用 ollama pull 下载相应的模型
    #    ollama pull qwen3:0.6b
    #    ollama pull qwen3-embedding:0.6b
    #    ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
    # 2. 确保 Ollama 服务正在运行
    
    rag = IntelligentRAG(
        documents_path=str(documents_path),  # 文档目录路径
        embedding_model="qwen3-embedding:0.6b",  # Ollama 嵌入模型
        llm_model="qwen3:0.6b",  # Ollama LLM 模型
        chunk_size=1000,
        chunk_overlap=200,
        ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0",  # Ollama Reranker 模型
    )
    
    # 构建 RAG 系统
    # k: 初始检索的文档块数量（重排序前会检索 k*2 个文档）
    # use_rerank: 是否启用重排序（使用 Ollama Reranker）
    # rerank_top_n: 重排序后保留的文档数量
    rag.build(k=4, use_rerank=True, rerank_top_n=3)
    
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
