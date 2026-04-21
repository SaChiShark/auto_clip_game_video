# 引入所有實作以觸發註冊機制
from .transcribers import whisper_transcriber, CT2_whisper_transcriber
from .diarizers import pyannote_diarizer, whisperx_diarizer
from .mergers import overlap_merger, whisperx_merger