"""
使用 Ollama 的 RAG 系统示例

本示例展示如何使用 Ollama 作为后端运行 RAG 系统。
需要先安装 Ollama 并下载相应的模型。

运行前准备：
1. 安装 Ollama: https://ollama.ai
2. 下载模型:
   - ollama pull qwen3:0.6b
   - ollama pull qwen3-embedding:0.6b
   - ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
3. 确保 Ollama 服务正在运行
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.RAG.rag_system import IntelligentRAG


def main():
    # 设置使用 Ollama
    os.environ["USE_OLLAMA"] = "true"
    # 如果 Ollama 运行在其他地址，可以设置：
    # os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    documents_path = project_root / "demo" / "RAG" / "documents"
    
    # 初始化 RAG 系统（使用 Ollama 模型）
    # 注意：需要先使用 ollama pull 下载相应的模型
    # 例如：ollama pull qwen3:0.6b
    #      ollama pull qwen3-embedding:0.6b
    #      ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
    rag = IntelligentRAG(
        documents_path=str(documents_path),  # 文档目录路径
        embedding_model="qwen3-embedding:0.6b",  # Ollama 嵌入模型
        llm_model="qwen3:0.6b",  # Ollama LLM 模型
        chunk_size=1000,
        chunk_overlap=200,
        ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0",  # Ollama Reranker 模型
    )
    
    # 构建 RAG 系统
    # k: 初始检索的文档块数量（重排序前）
    # use_rerank: 是否启用重排序（使用 Ollama Reranker）
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
        "RAG 系统的主要优势有哪些？",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)
        # 对第一个问题启用调试模式，查看检索到的文档内容
        verbose = (i == 1)
        result = rag.query(question, verbose=verbose)
        print(f"回答: {result['answer']}")
        print(f"\n参考文档数量: {len(result['source_documents'])}")
        print("=" * 50)


if __name__ == "__main__":
    main()
