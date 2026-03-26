from typing import Any, Dict, List, Optional
from .base import BaseDiarizer

class WhisperXDiarizer(BaseDiarizer):
    """
    一個使用 whisperX 的說話者辨識器。
    它接收一個已經初始化好的 whisperx.DiarizationPipeline 物件。
    """
    def __init__(self, diarization_pipeline: Any):
        """
        初始化 WhisperXDiarizer。

        Args:
            diarization_pipeline (Any): 一個已經被載入的 whisperx.DiarizationPipeline 物件。
        """
        print("WhisperXDiarizer: 初始化完成，已接收外部 WhisperX Pipeline。")
        self.pipeline = diarization_pipeline

    def run(self, audio_path: str, num_speakers: Optional[int] = 4, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        使用被注入的 Pipeline，對指定的音訊檔案進行說話者辨識。

        Args:
            audio_path (str): 要處理的音訊檔案路徑。
            num_speakers (int, optional): 指定說話者數量，預設為 4。
            min_speakers (int, optional): 最小說話者數量。
            max_speakers (int, optional): 最大說話者數量。
            **kwargs: 其他可能的擴展參數。

        Returns:
            List[Dict[str, Any]]: 包含說話者片段資訊的列表。
        """
        print(f"WhisperXDiarizer: 正在使用預載 Pipeline 處理 {audio_path}")
        
        output_segments = []
        try:
            diarize_segments = self.pipeline(audio_path, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers, **kwargs)
            
            for _, row in diarize_segments.iterrows():
                segment = {
                    'speaker': row['speaker'],
                    'start': row['start'],
                    'end': row['end']
                }
                output_segments.append(segment)
            
            return output_segments
        
        except Exception as e:
            print(f"WhisperX 說話者辨識過程中發生錯誤: {e}")
            return []
