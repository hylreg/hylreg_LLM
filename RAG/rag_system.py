"""
简单的 RAG (Retrieval-Augmented Generation) 系统实现
使用 LangChain 构建
"""

import os
from typing import List, Optional
# LangChain 1.0+ 使用 langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
# LangChain 1.2.7+ 中 RetrievalQA 已被移除，使用新的 API
# 我们将使用 create_retrieval_chain 或自定义实现
# LangChain 1.0+ 不再包含 langchain.chains，直接使用新 API
RETRIEVAL_QA_AVAILABLE = False
RetrievalQA = None
# LangChain 1.0+ 中 PromptTemplate 在 langchain_core.prompts
from langchain_core.prompts import PromptTemplate
# 重排序相关导入 - LangChain 1.0+ 路径
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CohereRerank
    COHERE_RERANK_AVAILABLE = True
except ImportError:
    try:
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors.cohere_rerank import CohereRerank
        COHERE_RERANK_AVAILABLE = True
    except ImportError:
        try:
            from langchain_community.retrievers import ContextualCompressionRetriever
            from langchain_cohere import CohereRerank
            COHERE_RERANK_AVAILABLE = True
        except ImportError:
            ContextualCompressionRetriever = None
            CohereRerank = None
            COHERE_RERANK_AVAILABLE = False

try:
    import cohere
    COHERE_SDK_AVAILABLE = True
except ImportError:
    cohere = None
    COHERE_SDK_AVAILABLE = False

try:
    from FlagEmbedding import FlagReranker
    FLAG_RERANKER_AVAILABLE = True
except ImportError:
    FlagReranker = None
    FLAG_RERANKER_AVAILABLE = False

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class IntelligentRAG:
    """智能 RAG 系统类 - 支持多种 LLM 和嵌入模型"""
    
    def __init__(
        self,
        documents_path: str,
        embedding_model: str = "qwen3-embedding:0.6b",
        llm_model: str = "qwen3:0.6b",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        reranker_model_path: Optional[str] = None,
        ollama_reranker_model: Optional[str] = None,
    ):
        """
        初始化 RAG 系统
        
        Args:
            documents_path: 文档目录路径
            embedding_model: 嵌入模型名称
            llm_model: LLM 模型名称
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            reranker_model_path: 本地 Reranker 模型路径（如："./Qwen/Qwen3-Reranker-0.6B"）
            ollama_reranker_model: Ollama Reranker 模型名称（如："dengcao/Qwen3-Reranker-0.6B:Q8_0"）
        """
        self.documents_path = documents_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reranker_model_path = reranker_model_path
        self.ollama_reranker_model = ollama_reranker_model
        
        # 初始化组件
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.local_reranker = None
        
        # 如果指定了本地 Reranker 路径，初始化它
        if self.reranker_model_path and FLAG_RERANKER_AVAILABLE:
            try:
                print(f"正在加载本地 Reranker 模型: {self.reranker_model_path}")
                self.local_reranker = FlagReranker(self.reranker_model_path, use_fp16=True)
                print("本地 Reranker 模型加载成功")
            except Exception as e:
                print(f"加载本地 Reranker 模型失败: {e}")
                self.local_reranker = None
        
        # 如果指定了 Ollama Reranker 模型，验证它是否存在
        if self.ollama_reranker_model:
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"将使用 Ollama Reranker 模型: {self.ollama_reranker_model}")
        
    def load_documents(self) -> List:
        """加载文档"""
        if os.path.isfile(self.documents_path):
            loader = TextLoader(self.documents_path, encoding="utf-8")
        else:
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
        
        documents = loader.load()
        print(f"已加载 {len(documents)} 个文档")
        return documents
    
    def _create_manual_rerank_retriever(self, base_retriever, cohere_api_key: str, top_n: int):
        """创建手动重排序的检索器包装器"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        
        class RerankRetriever(BaseRetriever):
            """使用 Cohere SDK 手动实现的 Reranker 检索器包装器"""
            
            def __init__(self, base_retriever, cohere_client, top_n):
                # 在 LangChain 1.0+ 中，BaseRetriever 使用 Pydantic 模型
                # 我们需要使用 object.__setattr__ 绕过 Pydantic 验证
                super().__init__()
                object.__setattr__(self, 'base_retriever', base_retriever)
                object.__setattr__(self, 'cohere_client', cohere_client)
                object.__setattr__(self, 'top_n', top_n)
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                # 从基础检索器获取文档
                # LangChain 1.0+ 使用 invoke() 方法
                try:
                    docs = self.base_retriever.invoke(query)
                except AttributeError:
                    # 兼容旧版本 API
                    try:
                        docs = self.base_retriever.get_relevant_documents(query)
                    except AttributeError:
                        # 如果都不行，尝试直接调用
                        docs = self.base_retriever(query)
                
                if len(docs) <= self.top_n:
                    return docs
                
                # 使用 Cohere 重排序
                try:
                    documents = [doc.page_content for doc in docs]
                    results = self.cohere_client.rerank(
                        model="rerank-multilingual-v3.0",
                        query=query,
                        documents=documents,
                        top_n=self.top_n
                    )
                    
                    # 根据重排序结果重新组织文档
                    reranked_docs = []
                    for result in results.results:
                        reranked_docs.append(docs[result.index])
                    
                    return reranked_docs
                except Exception as e:
                    print(f"重排序失败: {e}，返回原始结果")
                    return docs[:self.top_n]
        
        cohere_client = cohere.Client(api_key=cohere_api_key)
        return RerankRetriever(base_retriever, cohere_client, top_n)
    
    def _create_ollama_rerank_retriever(self, base_retriever, top_n: int):
        """创建使用 Ollama Reranker 的检索器包装器"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        import requests
        import json
        
        if not self.ollama_reranker_model:
            raise ValueError("Ollama Reranker 模型未设置，请检查 ollama_reranker_model 参数")
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        class OllamaRerankRetriever(BaseRetriever):
            """使用 Ollama Reranker 的检索器包装器"""
            
            def __init__(self, base_retriever, ollama_model, ollama_url, top_n):
                # 在 LangChain 1.0+ 中，BaseRetriever 使用 Pydantic 模型
                # 我们需要使用 model_construct 或直接设置属性
                super().__init__()
                # 使用 object.__setattr__ 绕过 Pydantic 验证
                object.__setattr__(self, 'base_retriever', base_retriever)
                object.__setattr__(self, 'ollama_model', ollama_model)
                object.__setattr__(self, 'ollama_url', ollama_url)
                object.__setattr__(self, 'top_n', top_n)
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                # 从基础检索器获取文档
                # LangChain 1.0+ 使用 invoke() 方法
                try:
                    docs = self.base_retriever.invoke(query)
                except AttributeError:
                    # 兼容旧版本 API
                    try:
                        docs = self.base_retriever.get_relevant_documents(query)
                    except AttributeError:
                        # 如果都不行，尝试直接调用
                        docs = self.base_retriever(query)
                
                if len(docs) <= self.top_n:
                    return docs
                
                # 使用 Ollama Reranker 重排序
                try:
                    documents = [doc.page_content for doc in docs]
                    
                    # 为每个文档计算相关性分数
                    scores = []
                    for doc_text in documents:
                        # 构造 Reranker 的输入格式：Query: ... Document: ...
                        prompt = f"Query: {query}\nDocument: {doc_text}\nRelevance score:"
                        
                        try:
                            # 调用 Ollama API
                            response = requests.post(
                                f"{self.ollama_url}/api/generate",
                                json={
                                    "model": self.ollama_model,
                                    "prompt": prompt,
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.0,  # 确定性输出
                                        "num_predict": 10,  # 只生成分数部分
                                    }
                                },
                                timeout=30
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            # 尝试从响应中提取分数
                            response_text = result.get("response", "").strip()
                            # 尝试解析数字分数
                            try:
                                # 提取数字（可能是浮点数）
                                import re
                                score_match = re.search(r'[-+]?\d*\.?\d+', response_text)
                                if score_match:
                                    score = float(score_match.group())
                                else:
                                    # 如果没有找到数字，使用响应长度作为简单评分
                                    score = len(response_text)
                            except:
                                score = len(response_text)
                            
                            scores.append(score)
                        except Exception as e:
                            print(f"Ollama Reranker 单个文档评分失败: {e}")
                            scores.append(0.0)  # 默认分数
                    
                    # 根据分数排序文档
                    doc_scores = list(zip(docs, scores))
                    doc_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # 返回 top_n 个文档
                    reranked_docs = [doc for doc, score in doc_scores[:self.top_n]]
                    return reranked_docs
                    
                except Exception as e:
                    print(f"Ollama Reranker 重排序失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return docs[:self.top_n]
        
        return OllamaRerankRetriever(base_retriever, self.ollama_reranker_model, ollama_base_url, top_n)
    
    def _create_local_rerank_retriever(self, base_retriever, top_n: int):
        """创建使用本地 Reranker 的检索器包装器"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        
        if self.local_reranker is None:
            raise ValueError("本地 Reranker 未初始化，请检查 reranker_model_path 参数")
        
        class LocalRerankRetriever(BaseRetriever):
            """使用本地 Reranker 的检索器包装器"""
            
            def __init__(self, base_retriever, reranker, top_n):
                # 在 LangChain 1.0+ 中，BaseRetriever 使用 Pydantic 模型
                # 我们需要使用 object.__setattr__ 绕过 Pydantic 验证
                super().__init__()
                object.__setattr__(self, 'base_retriever', base_retriever)
                object.__setattr__(self, 'reranker', reranker)
                object.__setattr__(self, 'top_n', top_n)
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                # 从基础检索器获取文档
                # LangChain 1.0+ 使用 invoke() 方法
                try:
                    docs = self.base_retriever.invoke(query)
                except AttributeError:
                    # 兼容旧版本 API
                    try:
                        docs = self.base_retriever.get_relevant_documents(query)
                    except AttributeError:
                        # 如果都不行，尝试直接调用
                        docs = self.base_retriever(query)
                
                if len(docs) <= self.top_n:
                    return docs
                
                # 使用本地 Reranker 重排序
                try:
                    documents = [doc.page_content for doc in docs]
                    # 准备查询-文档对
                    pairs = [[query, doc] for doc in documents]
                    
                    # 使用 FlagReranker 进行重排序
                    # rerank 方法返回排序后的结果列表，每个元素包含 'index' 和 'score'
                    reranked_results = self.reranker.rerank(pairs, top_k=self.top_n)
                    
                    # 根据重排序结果重新组织文档
                    reranked_docs = []
                    # FlagReranker 返回格式：列表，每个元素是字典 {'index': int, 'score': float}
                    for result in reranked_results:
                        if isinstance(result, dict) and 'index' in result:
                            reranked_docs.append(docs[result['index']])
                        elif isinstance(result, (int, tuple)):
                            # 兼容不同的返回格式
                            idx = result[0] if isinstance(result, tuple) else result
                            reranked_docs.append(docs[idx])
                    
                    return reranked_docs if reranked_docs else docs[:self.top_n]
                except Exception as e:
                    print(f"本地重排序失败: {e}，返回原始结果")
                    import traceback
                    traceback.print_exc()
                    return docs[:self.top_n]
        
        return LocalRerankRetriever(base_retriever, self.local_reranker, top_n)
    
    def split_documents(self, documents: List) -> List:
        """分割文档为小块"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"文档已分割为 {len(splits)} 个块")
        return splits
    
    def create_vectorstore(self, documents: List):
        """创建向量存储"""
        # 从环境变量获取配置
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
        siliconflow_base_url = os.getenv("SILICONFLOW_BASE_URL")
        
        # 初始化嵌入模型
        if use_ollama:
            # 使用 Ollama 嵌入模型
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=ollama_base_url,
            )
            print(f"使用 Ollama API 创建嵌入模型 (模型: {self.embedding_model}, 地址: {ollama_base_url})")
        elif siliconflow_api_key and siliconflow_base_url:
            # 使用硅基流动 API
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=siliconflow_api_key,
                openai_api_base=siliconflow_base_url,
            )
            print("使用硅基流动 API 创建嵌入模型")
        else:
            # 使用默认 OpenAI API
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            print("使用 OpenAI API 创建嵌入模型")
        
        # 创建向量存储
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("向量存储已创建")
    
    def create_qa_chain(self, k: int = 4, use_rerank: bool = True, rerank_top_n: int = 3):
        """
        创建问答链
        
        Args:
            k: 初始检索的文档块数量（重排序前）
            use_rerank: 是否使用重排序
            rerank_top_n: 重排序后保留的文档数量
        """
        if self.vectorstore is None:
            raise ValueError("请先创建向量存储")
        
        # 从环境变量获取配置
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
        siliconflow_base_url = os.getenv("SILICONFLOW_BASE_URL")
        
        # 初始化 LLM
        if use_ollama:
            # 使用 Ollama LLM
            llm = ChatOllama(
                model=self.llm_model,
                temperature=0,
                base_url=ollama_base_url,
            )
            print(f"使用 Ollama API 创建 LLM (模型: {self.llm_model}, 地址: {ollama_base_url})")
        elif siliconflow_api_key and siliconflow_base_url:
            # 使用硅基流动 API
            llm = ChatOpenAI(
                model=self.llm_model,
                temperature=0,
                openai_api_key=siliconflow_api_key,
                openai_api_base=siliconflow_base_url,
            )
            print("使用硅基流动 API 创建 LLM")
        else:
            # 使用默认 OpenAI API
            llm = ChatOpenAI(model=self.llm_model, temperature=0)
            print("使用 OpenAI API 创建 LLM")
        
        # 创建基础检索器（检索更多文档用于重排序）
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k * 2 if use_rerank else k}  # 如果使用重排序，检索更多文档
        )
        
        # 如果启用重排序，创建带重排序的检索器
        if use_rerank:
            retriever = None
            
            # 优先级1: 使用 Ollama Reranker（如果已设置）
            if self.ollama_reranker_model:
                try:
                    retriever = self._create_ollama_rerank_retriever(
                        base_retriever, rerank_top_n
                    )
                    print(f"已启用 Ollama Reranker 重排序 (模型: {self.ollama_reranker_model}, 保留前 {rerank_top_n} 个文档)")
                except Exception as e:
                    print(f"Ollama Reranker 初始化失败: {e}，尝试其他方式")
                    retriever = None
            
            # 优先级2: 使用本地 Reranker（如果已初始化且 Ollama Reranker 不可用）
            if retriever is None and self.local_reranker is not None:
                try:
                    retriever = self._create_local_rerank_retriever(
                        base_retriever, rerank_top_n
                    )
                    print(f"已启用本地 Reranker 重排序 (模型: {self.reranker_model_path}, 保留前 {rerank_top_n} 个文档)")
                except Exception as e:
                    print(f"本地 Reranker 初始化失败: {e}，尝试使用 Cohere API")
                    retriever = None
            
            # 优先级3: 使用 Cohere API（如果本地 Reranker 不可用或失败）
            if retriever is None:
                cohere_api_key = os.getenv("COHERE_API_KEY")
                if cohere_api_key and COHERE_RERANK_AVAILABLE:
                    try:
                        # 使用 LangChain 的 CohereRerank
                        compressor = CohereRerank(
                            cohere_api_key=cohere_api_key,
                            top_n=rerank_top_n,
                            model="rerank-multilingual-v3.0"  # 支持中文
                        )
                        retriever = ContextualCompressionRetriever(
                            base_compressor=compressor,
                            base_retriever=base_retriever
                        )
                        print(f"已启用 Cohere 重排序 (保留前 {rerank_top_n} 个文档)")
                    except Exception as e:
                        print(f"初始化 Cohere Rerank 失败: {e}，使用基础检索器")
                        retriever = base_retriever
                elif cohere_api_key and COHERE_SDK_AVAILABLE:
                    # 使用 Cohere SDK 手动实现重排序
                    try:
                        retriever = self._create_manual_rerank_retriever(
                            base_retriever, cohere_api_key, rerank_top_n
                        )
                        print(f"已启用 Cohere 重排序 (手动实现，保留前 {rerank_top_n} 个文档)")
                    except Exception as e:
                        print(f"Cohere SDK 重排序失败: {e}，使用基础检索器")
                        retriever = base_retriever
                else:
                    # 如果没有 Cohere API Key 或相关库，使用基础检索器
                    retriever = base_retriever
                    if not cohere_api_key:
                        print("未设置 COHERE_API_KEY 且未配置本地/Ollama Reranker，跳过重排序")
                    else:
                        print("Cohere 相关库未安装，跳过重排序。请安装: pip install cohere langchain-cohere")
        else:
            retriever = base_retriever
            print("未启用重排序")
        
        # 定义提示模板 - 改进版，更好地处理概括性问题
        prompt_template = """你是一个有用的AI助手。请根据以下提供的上下文信息回答问题。

上下文信息：
{context}

问题：{question}

请基于上述上下文信息回答问题。重要提示：
1. 上下文信息中包含了文档的完整内容，请仔细阅读并理解
2. 如果问题是关于"主要内容"、"总结"、"概述"、"关键信息"等概括性问题，请基于上下文中的所有信息进行全面的概括和总结
3. 对于"主要内容是什么"这类问题，上下文信息本身就是文档内容，请直接基于这些内容进行概括
4. 如果问题是具体的事实性问题，请从上下文中找到相关信息并详细回答
5. 只有在上下文中确实完全没有相关信息时，才说"根据提供的上下文，我无法找到相关信息"
6. 尽量使用上下文中的具体信息，不要编造答案

回答："""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 创建问答链 - 使用 LangChain 1.2.7+ 的新 API
        if RETRIEVAL_QA_AVAILABLE and RetrievalQA:
            # 如果旧版 API 可用，使用它
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        else:
            # 使用新的 API - 创建自定义 RAG 链
            from langchain_core.runnables import RunnablePassthrough, RunnableLambda
            from langchain_core.output_parsers import StrOutputParser
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # 创建一个函数来从输入中提取 question 并调用 retriever
            def retrieve_docs(input_dict):
                question = input_dict.get("question", "")
                return retriever.invoke(question)
            
            # 创建一个函数来提取 question 字段
            def extract_question(input_dict):
                return input_dict.get("question", "")
            
            # 创建检索和生成链
            # 使用 RunnableLambda 确保 retriever 接收字符串而不是字典
            rag_chain = (
                {
                    "context": RunnableLambda(retrieve_docs) | format_docs,
                    "question": RunnableLambda(extract_question)
                }
                | PROMPT
                | llm
                | StrOutputParser()
            )
            
            # 包装成兼容的接口
            class CustomRAGChain:
                def __init__(self, chain, retriever):
                    self.chain = chain
                    self.retriever = retriever
                
                def invoke(self, inputs):
                    query = inputs.get("query", "")
                    # 获取源文档 - LangChain 1.0+ 使用 invoke() 方法
                    # retriever 实现了 Runnable 接口，可以直接调用 invoke
                    source_docs = self.retriever.invoke(query)
                    # 生成答案
                    answer = self.chain.invoke({"question": query})
                    return {
                        "result": answer,
                        "source_documents": source_docs
                    }
            
            self.qa_chain = CustomRAGChain(rag_chain, retriever)
        print("问答链已创建")
    
    def build(self, k: int = 4, use_rerank: bool = True, rerank_top_n: int = 3):
        """
        构建完整的 RAG 系统
        
        Args:
            k: 初始检索的文档块数量（重排序前）
            use_rerank: 是否使用重排序
            rerank_top_n: 重排序后保留的文档数量
        """
        print("开始构建 RAG 系统...")
        
        # 1. 加载文档
        documents = self.load_documents()
        
        # 2. 分割文档
        splits = self.split_documents(documents)
        
        # 3. 创建向量存储
        self.create_vectorstore(splits)
        
        # 4. 创建问答链
        self.create_qa_chain(k=k, use_rerank=use_rerank, rerank_top_n=rerank_top_n)
        
        print("RAG 系统构建完成！")
    
    def query(self, question: str, verbose: bool = False) -> dict:
        """
        查询问题
        
        Args:
            question: 用户问题
            verbose: 是否显示调试信息（检索到的文档内容）
            
        Returns:
            包含答案和源文档的字典
        """
        if self.qa_chain is None:
            raise ValueError("请先构建 RAG 系统")
        
        result = self.qa_chain.invoke({"query": question})
        
        # 如果启用详细模式，显示检索到的文档内容
        if verbose:
            print("\n[调试信息] 检索到的文档内容：")
            print("-" * 50)
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n文档 {i}:")
                print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            print("-" * 50)
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def save_vectorstore(self, path: str):
        """保存向量存储到磁盘"""
        if self.vectorstore is None:
            raise ValueError("向量存储不存在")
        self.vectorstore.save_local(path)
        print(f"向量存储已保存到 {path}")
    
    def load_vectorstore(self, path: str):
        """从磁盘加载向量存储"""
        if self.embeddings is None:
            # 从环境变量获取配置
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
            siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
            siliconflow_base_url = os.getenv("SILICONFLOW_BASE_URL")
            
            if use_ollama:
                # 使用 Ollama 嵌入模型
                self.embeddings = OllamaEmbeddings(
                    model=self.embedding_model,
                    base_url=ollama_base_url,
                )
            elif siliconflow_api_key and siliconflow_base_url:
                # 使用硅基流动 API
                self.embeddings = OpenAIEmbeddings(
                    model=self.embedding_model,
                    openai_api_key=siliconflow_api_key,
                    openai_api_base=siliconflow_base_url,
                )
            else:
                # 使用默认 OpenAI API
                self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"向量存储已从 {path} 加载")