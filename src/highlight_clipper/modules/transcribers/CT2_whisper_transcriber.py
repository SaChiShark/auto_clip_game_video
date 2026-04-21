from typing import Any
from .base import ASRTranscriber, NormalizedTranscription
from transformers import pipeline
from ...registry import asr_registry

@asr_registry.register("CT2")
def create_ct2_transcriber(model_path: str, device: str, **kwargs) -> ASRTranscriber:
    from faster_whisper import WhisperModel
    print(f"Factory: 正在建立 Transcriber (策略: CT2)...")
    if device != 'cpu':
        model = WhisperModel(model_path, device=device, compute_type="default", num_workers=1)
    else:
        model = WhisperModel(model_path, device="cpu", compute_type="int8", num_workers=1)
    return CT2Transcriber(model)

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