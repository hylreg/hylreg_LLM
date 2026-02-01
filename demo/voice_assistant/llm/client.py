"""
语音助手 LLM 客户端
封装 agents 的 IntelligentAgent，供 pipeline 调用（仅文本输入/输出）
"""

from typing import Optional

# 从 demo.agents 包导入
try:
    from demo.agents.agent_system import IntelligentAgent
except ImportError:
    import sys
    from pathlib import Path
    # 本文件在 demo/voice_assistant/llm/ 下，项目根为 parents[3]
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from demo.agents.agent_system import IntelligentAgent


# 语音助手专用系统提示
VOICE_ASSISTANT_SYSTEM_PROMPT = """你是一个语音助手。请用简洁、口语化的方式回答用户，适合听而不是看。
回答尽量短一些，方便用户听完。若用户用语音提问，请直接给出结论或步骤，避免长列表。"""


class LLMClient:
    """封装 IntelligentAgent，提供 chat(text) -> text 接口"""

    def __init__(
        self,
        llm_model: str = "qwen3:0.6b",
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        self.agent = IntelligentAgent(
            llm_model=llm_model,
            tools=None,
            system_prompt=system_prompt or VOICE_ASSISTANT_SYSTEM_PROMPT,
            verbose=verbose,
        )

    def chat(self, user_text: str) -> str:
        """用户文本 -> 助手回复文本"""
        if not user_text or not user_text.strip():
            return ""
        result = self.agent.run(user_text.strip())
        return (result.get("answer") or "").strip()
