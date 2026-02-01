"""语音输入输出：ASR、TTS（阶段二接入）"""

from .asr import transcribe_audio, listen_and_transcribe
from .tts import synthesize_and_play, synthesize_to_file

__all__ = [
    "transcribe_audio",
    "listen_and_transcribe",
    "synthesize_and_play",
    "synthesize_to_file",
]
