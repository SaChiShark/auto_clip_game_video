from typing import Any, Dict, List
from .base import BaseDiarizer

class PyannoteDiarizer(BaseDiarizer):
    """
    一個使用 pyannote.audio 的說話者辨識器。
    它接收一個已經初始化好的 Pyannote Pipeline 物件。
    """
    def __init__(self, diarization_pipeline: Any):
        """
        初始化 PyannoteDiarizer。

        Args:
            diarization_pipeline (Any): 一個已經被載入的 pyannote.audio.Pipeline 物件。
        """
        print("PyannoteDiarizer: 初始化完成，已接收外部 Pyannote Pipeline。")
        self.pipeline = diarization_pipeline

    def run(self, audio_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        使用被注入的 Pipeline，對指定的音訊檔案進行說話者辨識。

        Args:
            audio_path (str): 要處理的音訊檔案路徑。
            **kwargs: 傳遞給 pyannote pipeline 的額外參數 (例如 num_speakers=4)。

        Returns:
            List[Dict[str, Any]]: 包含說話者片段資訊的列表。
        """
        print(f"PyannoteDiarizer: 正在使用預載 Pipeline 處理 {audio_path}")
        
        output_segments = []
        try:
            # 使用 kwargs 將外部指定的參數傳進去 (如 num_speakers)
            diarization = self.pipeline(audio_path, **kwargs)
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    'speaker': speaker,
                    'start': turn.start,
                    'end': turn.end
                }
                output_segments.append(segment)
            
            return output_segments
        
        except Exception as e:
            print(f"說話者辨識過程中發生錯誤: {e}")
            # 失敗時回傳空列表
            return []
