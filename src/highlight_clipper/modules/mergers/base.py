from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..transcribers.base import NormalizedTranscription

class BaseMerger(ABC):
    """
    抽象基礎類別 (ABC)，定義了所有 Merger (多模態合併器) 必須遵守的合約。
    """
    
    @abstractmethod
    def run(self, 
            transcription_result: NormalizedTranscription, 
            diarization_result: List[Dict[str, Any]],
            **kwargs) -> List[Dict[str, Any]]:
        """
        執行合併的核心公開方法。

        Args:
            transcription_result (NormalizedTranscription): 
                來自 ASR Transcriber 的標準化詞語列表。
            diarization_result (List[Dict[str, Any]]): 
                來自 Diarizer 的說話者時間區段列表。
            **kwargs:
                傳遞給特定演算法的其他參數。

        Returns:
            List[Dict[str, Any]]:
                合併後的句子列表。
                格式: [{'speaker': str, 'text': str, 'start': float, 'end': float, 'words': [...]}, ...]
        """
        pass
