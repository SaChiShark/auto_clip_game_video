from abc import ABC, abstractmethod
from typing import List, Dict, Any

# 定義我們系統內部統一的、標準化的語音辨識結果格式
# 這是一個由「單字字典」組成的列表
# 每個單字字典都必須包含 'word', 'start', 'end' 這三個鍵
NormalizedWord = Dict[str, Any] # {'word': str, 'start': float, 'end': float}
NormalizedTranscription = List[NormalizedWord]

class ASRTranscriber(ABC):
    """
    抽象基礎類別 (ABC)，定義了所有 ASR Transcriber 必須遵守的合約。
    """
    
    @abstractmethod
    def run(self, audio_path: str) -> NormalizedTranscription:
        """
        對指定的音訊檔案進行語音辨識。

        Args:
            audio_path (str): 要處理的音訊檔案路徑。

        Returns:
            NormalizedTranscription:
                一個標準化的結果。無論底層用的是哪個模型，
                都必須回傳這個格式：
                [
                    {'word': '你好', 'start': 1.23, 'end': 1.55},
                    {'word': '世界', 'start': 1.60, 'end': 2.10},
                    ...
                ]
        """
        pass