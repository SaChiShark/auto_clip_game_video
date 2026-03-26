import os
import whisper
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import torch

# 匯入所有需要的元件藍圖
from .video_processor import VideoProcessor
from .modules.transcribers.base import ASRTranscriber
from .modules.transcribers.base import ASRTranscriber
from .modules.transcribers.whisper_transcriber import WhisperTranscriber
from .modules.transcribers.CT2_whisper_transcriber import CT2Transcriber
from .modules.diarizers.base import BaseDiarizer
from .modules.mergers.base import BaseMerger

# 從環境變數讀取 Token，集中管理
HF_TOKEN = os.environ.get("HF_ACCESS_TOKEN")

class ProcessorFactory:
    """
    一個簡單工廠，負責建立並組態一個完整的 VideoProcessor 物件。
    """
    def create_processor(self, asr_strategy: str = "whisper",asr_model_path: str = 'turbo',diarizer_strategy: str = "whisperx",merger_strategy: str = "whisperx") -> VideoProcessor:
        """
        工廠的主要方法。

        Args:
            asr_strategy (str): 決定使用哪種 ASR 模型的策略名稱。

        Returns:
            VideoProcessor: 一個已經組態完畢、隨時可以使用的 VideoProcessor 實例。
        """
        print("ProcessorFactory: 收到訂單，開始建立 VideoProcessor...")
        
        # --- 這裡就是原本在 main.py 中的所有建立邏輯 ---
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 建立 Transcriber
        transcriber = self._create_transcriber(asr_strategy, asr_model_path, device)
        
        # 建立 Diarizer
        diarizer = self._create_diarizer(device,diarizer_strategy)
        merger = self._create_merger(merger_strategy)
        # 組裝並回傳最終產品
        processor = VideoProcessor(
            transcriber=transcriber,
            diarizer=diarizer,
            merger=merger,
            # 未來還可以注入 merger, highlight_strategy 等
        )
        print("ProcessorFactory: VideoProcessor 建立完成！")
        return processor

    def _create_transcriber(self, strategy: str,model_path: str, device: str) -> ASRTranscriber:
        """根據策略建立 Transcriber"""
        print(f"Factory: 正在建立 Transcriber (策略: {strategy})...")
        if strategy == "whisper":
            whisper_model_instance = whisper.load_model(model_path, device=device)
            return WhisperTranscriber(whisper_model=whisper_model_instance)
        elif strategy == "CT2":
            if device != 'cpu':
                model = WhisperModel(model_path, device=device, compute_type="default", num_workers=1)
            else:
                model = WhisperModel(model_path, device="cpu", compute_type="int8", num_workers=1)
            return CT2Transcriber(model)
        else:
            raise ValueError(f"未知的 ASR 策略: {strategy}")
            
    def _create_diarizer(self, device: str, strategy: str = "whisperx") -> BaseDiarizer:
        """建立 Diarizer"""
        print(f"Factory: 正在建立 Diarizer (策略: {strategy})...")
        if strategy == "pyannote":
            diarization_pipeline_instance = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            ).to(torch.device(device))
            from .modules.diarizers.pyannote_diarizer import PyannoteDiarizer
            return PyannoteDiarizer(diarization_pipeline=diarization_pipeline_instance)
        elif strategy == "whisperx":
            from whisperx.diarize import DiarizationPipeline
            token = HF_TOKEN
            if not token:
                print("警告: 尚未設定 HF_ACCESS_TOKEN 環境變數，WhisperX 可能無法下載模型。")
            diarization_pipeline_instance = DiarizationPipeline(token=HF_TOKEN, device=device)
            from .modules.diarizers.whisperx_diarizer import WhisperXDiarizer
            return WhisperXDiarizer(diarization_pipeline=diarization_pipeline_instance)
        else:
            raise ValueError(f"未知的 Diarization 策略: {strategy}")
    
    def _create_merger(self, strategy: str = "whisperx") -> BaseMerger:
        """建立 Merger"""
        print(f"Factory: 正在建立 Merger (策略: {strategy})...")
        if strategy == "overlap":
            from .modules.mergers.overlap_merger import OverlapMerger
            return OverlapMerger()
        elif strategy == "whisperx":
            from .modules.mergers.whisperx_merger import WhisperXMerger
            return WhisperXMerger()
        else:
            raise ValueError(f"未知的 Merger 策略: {strategy}")