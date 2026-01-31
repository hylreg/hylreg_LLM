"""
语音合成 (TTS) 模块
阶段二接入：优先 ModelScope TTS，实现 文本 -> 音频播放
当前为占位，仅文本输出时由 pipeline 打印。
"""

from typing import Optional


def synthesize_and_play(text: str) -> None:
    """
    将文本合成为语音并播放（占位）。
    后续接入 ModelScope TTS 等实现。
    """
    raise NotImplementedError(
        "TTS 未实现。请接入 ModelScope TTS 或先使用文本模式。"
    )


def synthesize_to_file(text: str, output_path: str) -> None:
    """
    将文本合成为音频文件（占位）。
    后续接入 ModelScope TTS 等实现。
    """
    raise NotImplementedError(
        "TTS 未实现。请接入 ModelScope TTS 或先使用文本模式。"
    )
