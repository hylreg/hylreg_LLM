#!/usr/bin/env python3
"""
语音助手 Demo 入口
阶段一：文本模式，从终端读入一行 -> LLM 回复 -> 打印。
从项目根目录运行: uv run python demo/voice_assistant/run_voice_assistant.py
"""

import sys
from pathlib import Path

_file = Path(__file__).resolve()
_demo_dir = _file.parents[1]  # demo/，使 voice_assistant 解析为 demo/voice_assistant

if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))

from voice_assistant.core import VoiceAssistantPipeline


def main():
    pipeline = VoiceAssistantPipeline(
        llm_model="qwen3:0.6b",
        verbose=False,
    )
    pipeline.run_loop_text()


if __name__ == "__main__":
    main()
