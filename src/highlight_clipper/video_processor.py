# src/highlight_clipper/video_processor.py

import os
import json
import uuid
from datetime import datetime
import torch
import whisper
from pyannote.audio import Pipeline

# 假設你已經將 modules 中的函式重構成 Class 以便管理模型
from .modules.transcribers.base import ASRTranscriber # 載入一次 Whisper 模型
from .modules.diarizers.base import BaseDiarizer       # 載入一次 Pyannote/WhisperX 模型
from .modules.converter import convert_mp4_to_wav
from .modules.merger import Merger
from .modules.audio_utils import mute_non_speech_segments

class VideoProcessor:
    def __init__(self,diarizer: BaseDiarizer, transcriber: ASRTranscriber,merger:Merger , output_root: str = "outputs"):
        print("VideoProcessor: 正在初始化引擎...")
        # --- 依賴注入階段 ---
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.merger = merger
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        print("引擎初始化完成。")

    def process(self, video_path: str) -> bool:
        """
        處理單一影片的完整管線。
        這個方法可以被重複呼叫。

        Args:
            video_path (str): 要處理的影片檔案路徑。

        Returns:
            bool: 合併後的對話資料，如果失敗則回傳 None。
        """
        if not os.path.exists(video_path):
            print(f"錯誤：找不到影片檔案：{video_path}")
            return False

        # --- 每次處理都獨立的任務狀態 ---
        job_id = self._generate_job_id()
        job_output_dir = os.path.join(self.output_root, job_id)
        os.makedirs(job_output_dir, exist_ok=True)
        
        print(f"\n--- 開始新任務 Job ID: {job_id} ---")
        print(f"來源影片: {video_path}")
        print(f"輸出目錄: {job_output_dir}")

        #try:
        # 步驟 1: 提取音訊
        wav_path = self._extract_audio(video_path, job_output_dir)
        
        # 步驟 2: 分析 (使用預載的模型)
        print(f"[{job_id}] 正在進行語音轉文字...")
        transcription_result = self.transcriber.run(wav_path)
        
        print(f"[{job_id}] 正在進行說話者辨識...")
        diarization_result = self.diarizer.run(wav_path)
        
        # 步驟 3: 合併
        print(f"[{job_id}] 正在合併分析結果...")
        merged_result = self.merger.run(transcription_result, diarization_result)

        # 步驟 4: 儲存與記錄
        self._save_results(merged_result, job_id, job_output_dir)
        self._update_manifest(job_id, video_path, status="completed")
        
        print(f"--- 任務 {job_id} 成功完成 ---")
        return True

        #except Exception as e:
        #    print(f"--- 任務 {job_id} 處理失敗 ---")
        #    print(f"錯誤訊息: {e}")
        #    self._update_manifest(job_id, video_path, status="failed")
        #    return False

    def _extract_audio(self, video_path, job_output_dir):
        print(f"[{os.path.basename(job_output_dir)}] 正在提取音訊...")
        wav_output_path = os.path.join(job_output_dir, "audio.wav")
        wav_path = convert_mp4_to_wav(video_path, wav_output_path)
        if not wav_path:
            raise Exception("音訊提取失敗。")
        return wav_path

    def _save_results(self, merged_result, job_id, job_output_dir):
        json_path = os.path.join(job_output_dir, "merged_dialogue.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=4)
        print(f"[{job_id}] 結果已儲存至: {json_path}")
        
    def _generate_job_id(self) -> str:
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        unique_part = uuid.uuid4().hex[:6]
        return f"{time_str}_{unique_part}"

    def _update_manifest(self, job_id, video_path, status):
        manifest_path = os.path.join(self.output_root, "_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
        else:
            manifest_data = []
        
        manifest_data.append({
            "job_id": job_id,
            "source_video_path": os.path.abspath(video_path),
            "status": status,
            "processed_at": datetime.now().isoformat()
        })
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=4)