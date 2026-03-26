from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDiarizer(ABC):
    """
    抽象基礎類別 (ABC)，定義了所有 Diarizer (說話者辨識器) 必須遵守的合約。
    """
    
    @abstractmethod
    def run(self, audio_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        對指定的音訊檔案進行說話者辨識。

        Args:
            audio_path (str): 要處理的音訊檔案路徑。
            **kwargs: 其他模型可能需要的特定參數 (如 num_speakers)。

        Returns:
            List[Dict[str, Any]]:
                包含說話者片段資訊的列表。
                格式必須為:
                [
                    {'speaker': 'SPEAKER_00', 'start': 0.5, 'end': 2.3},
                    {'speaker': 'SPEAKER_01', 'start': 2.5, 'end': 4.1},
                    ...
                ]
        """
        pass
