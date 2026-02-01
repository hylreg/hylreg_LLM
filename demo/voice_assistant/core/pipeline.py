"""
语音助手流水线：ASR → LLM → TTS
阶段一：仅文本模式（text_in -> LLM -> text_out）
阶段二：可接入 asr/tts 实现完整语音闭环
"""

from typing import Optional, Callable

from voice_assistant.llm import LLMClient


class VoiceAssistantPipeline:
    """串联 输入(文本/语音) → LLM → 输出(文本/语音)"""

    def __init__(
        self,
        llm_model: str = "qwen3:0.6b",
        system_prompt: Optional[str] = None,
        asr_fn: Optional[Callable[[], str]] = None,
        tts_fn: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ):
        self.llm = LLMClient(
            llm_model=llm_model,
            system_prompt=system_prompt,
            verbose=verbose,
        )
        # 若未提供则使用文本占位
        self._asr = asr_fn
        self._tts = tts_fn

    def run_once_text(self, user_text: str) -> str:
        """单轮文本：用户文本 -> 助手回复文本（不经过 ASR/TTS）"""
        reply = self.llm.chat(user_text)
        return reply

    def run_once(self) -> Optional[str]:
        """
        单轮完整流程：输入(语音/文本) -> LLM -> 输出(语音/文本)。
        若未配置 asr，则使用 stdin 读一行文本；若未配置 tts，则只返回文本。
        """
        if self._asr is not None:
            user_text = self._asr()
        else:
            user_text = input("你说: ").strip()
        if not user_text:
            return None
        reply = self.llm.chat(user_text)
        if self._tts is not None:
            self._tts(reply)
        return reply

    def run_loop_text(self):
        """文本模式循环：从 stdin 读一行 -> 打印回复，直到空行或 Ctrl+C"""
        print("语音助手（文本模式）。输入一句话回车，空行退出。")
        while True:
            try:
                line = input("你说: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            reply = self.run_once_text(line)
            print(f"助手: {reply}\n")
