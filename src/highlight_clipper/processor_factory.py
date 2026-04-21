import os
import torch

# 匯入所有需要的元件藍圖
from .video_processor import VideoProcessor
from .modules.transcribers.base import ASRTranscriber
from .modules.diarizers.base import BaseDiarizer
from .modules.mergers.base import BaseMerger

# 匯入註冊表
from .registry import asr_registry, diarization_registry, merger_registry

# 匯入 modules 來觸發所有的 @register 裝飾器
import src.highlight_clipper.modules

class ProcessorFactory:
    """
    統一從 Registry 建立並組裝 VideoProcessor 的工廠類別。
    """
    def create_processor(self, asr_strategy: str = "whisper", asr_model_path: str = 'turbo', diarizer_strategy: str = "whisperx", merger_strategy: str = "whisperx") -> VideoProcessor:
        """
        工廠的主要方法。
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 透過 Registry 建立各個對應的模組，並將參數動態傳入
        transcriber = asr_registry.create(asr_strategy, model_path=asr_model_path, device=device)
        diarizer = diarization_registry.create(diarizer_strategy, device=device)
        merger = merger_registry.create(merger_strategy)
        
        # 組裝並回傳最終產品
        processor = VideoProcessor(
            transcriber=transcriber,
            diarizer=diarizer,
            merger=merger,
        )
        print("ProcessorFactory: VideoProcessor 建立完成！")
        return processor
