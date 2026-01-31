"""
语音识别 (ASR) 模块
阶段二接入：优先 ModelScope FunASR，实现 麦克风/音频 -> 文本
当前为占位，仅文本输入时由 pipeline 读 stdin。
"""

from typing import Optional


def transcribe_audio(audio_path: str) -> str:
    """
    将音频文件转成文本（占位）。
    后续接入 ModelScope FunASR 等实现。
    """
    raise NotImplementedError(
        "ASR 未实现。请接入 ModelScope FunASR 或先使用文本模式。"
    )


def listen_and_transcribe() -> str:
    """
    从麦克风录音并转成文本（占位）。
    后续接入 ModelScope FunASR 实时/流式接口。
    """
    raise NotImplementedError(
        "麦克风 ASR 未实现。请接入 ModelScope FunASR 或先使用文本模式。"
    )
