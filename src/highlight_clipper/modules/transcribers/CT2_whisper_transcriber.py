from typing import Any
from .base import ASRTranscriber, NormalizedTranscription
from transformers import pipeline
class CT2Transcriber(ASRTranscriber):
    """
    使用 Whisper 模型的具體 Transcriber 實作。
    """
    def __init__(self, CT2_model: pipeline):
        self.model = CT2_model

    def run(self, audio_path: str) -> NormalizedTranscription:
        """
        呼叫 BELLE_model 模型並進行後處理，將結果轉換為標準格式。
        """
        segments, info = self.model.transcribe(audio_path, beam_size=10, language="zh", vad_filter=False)
        normalized_result: NormalizedTranscription = []
        for segment in segments:
            normalized_result.append({
                'word': segment.text,
                'start': segment.start,
                'end': segment.end
            })
        
        return normalized_result