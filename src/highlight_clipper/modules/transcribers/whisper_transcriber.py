from typing import Any
from .base import ASRTranscriber, NormalizedTranscription

class WhisperTranscriber(ASRTranscriber):
    """
    使用 Whisper 模型的具體 Transcriber 實作。
    """
    def __init__(self, whisper_model: Any):
        self.model = whisper_model

    def run(self, audio_path: str) -> NormalizedTranscription:
        """
        呼叫 Whisper 模型並進行後處理，將結果轉換為標準格式。
        """
        print("WhisperTranscriber: 正在使用 Whisper 模型進行辨識...")
        
        raw_result = self.model.transcribe(audio_path, word_timestamps=True)
        
        # --- 核心：後處理 (Post-processing) ---
        normalized_result: NormalizedTranscription = []
        for segment in raw_result.get('segments', []):
            for word_info in segment.get('words', []):
                # Whisper 的 'word' 鍵前面有個多餘的空白，用 strip() 清掉
                normalized_result.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        
        return normalized_result