"""
智能体系统核心实现
基于 LangChain 构建的智能体框架
"""

from typing import List, Optional, Dict, Any
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.siliconflow import SiliconFlowLLM


class IntelligentAgent:
    """智能体系统"""
    
    def __init__(
        self,
        llm_model: str = "qwen3:0.6b",
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        初始化智能体
        
        Args:
            llm_model: LLM 模型名称
            tools: 工具列表
            system_prompt: 系统提示词
            verbose: 是否显示详细日志
        """
        self.llm_model = llm_model
        self.tools = tools or []
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.verbose = verbose
        
        # 初始化 LLM
        self.llm = self._init_llm()
        
        # 初始化智能体
        self.agent_executor = self._create_agent()
    
    def _init_llm(self):
        """初始化 LLM，根据环境变量自动选择"""
        use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"
        siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if use_ollama:
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"✓ 使用 Ollama: {ollama_base_url}, 模型: {self.llm_model}")
            return ChatOllama(
                model=self.llm_model,
                base_url=ollama_base_url,
                temperature=0.7,
            )
        elif siliconflow_api_key:
            siliconflow_base_url = os.getenv(
                "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/"
            )
            print(f"✓ 使用硅基流动 API: {siliconflow_base_url}, 模型: {self.llm_model}")
            return ChatOpenAI(
                model=self.llm_model,
                api_key=siliconflow_api_key,
                base_url=siliconflow_base_url,
                temperature=0.7,
            )
        elif openai_api_key:
            print(f"✓ 使用 OpenAI API, 模型: {self.llm_model}")
            return ChatOpenAI(
                model=self.llm_model,
                api_key=openai_api_key,
                temperature=0.7,
            )
        else:
            raise ValueError(
                "未找到可用的 LLM 配置。请设置 USE_OLLAMA=true 或提供 API Key"
            )
    
    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个有用的AI助手。你可以使用提供的工具来帮助用户完成任务。
请仔细思考每一步，并使用合适的工具来解决问题。
如果用户的问题无法通过工具解决，请直接回答。"""
    
    def _create_agent(self) -> AgentExecutor:
        """创建智能体执行器"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 如果有工具，创建工具智能体
        if self.tools:
            agent = create_openai_tools_agent(self.llm, self.tools, prompt)
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=self.verbose,
                handle_parsing_errors=True,
            )
        else:
            # 如果没有工具，直接使用 LLM
            return self.llm
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        运行智能体
        
        Args:
            query: 用户查询
            **kwargs: 其他参数
            
        Returns:
            智能体的响应
        """
        if isinstance(self.agent_executor, AgentExecutor):
            result = self.agent_executor.invoke({"input": query, **kwargs})
            return {
                "answer": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
            }
        else:
            # 直接使用 LLM
            response = self.agent_executor.invoke(query)
            return {
                "answer": response.content if hasattr(response, "content") else str(response),
            }
    
    def add_tool(self, tool: BaseTool):
        """添加工具"""
        self.tools.append(tool)
        # 重新创建智能体
        self.agent_executor = self._create_agent()
    
    def clear_tools(self):
        """清空工具"""
        self.tools = []
        # 重新创建智能体
        self.agent_executor = self._create_agent()

