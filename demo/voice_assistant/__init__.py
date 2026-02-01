"""
语音助手 Demo 实现
ASR → LLM → TTS 串联
"""

from .core import VoiceAssistantPipeline
from .llm import LLMClient

__all__ = ["LLMClient", "VoiceAssistantPipeline"]
