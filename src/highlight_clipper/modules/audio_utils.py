import numpy as np
import scipy.io.wavfile as wavfile

def mute_non_speech_segments(audio_path: str, transcription_result: list, output_path: str) -> str:
    """
    根據語音轉文字的結果，把沒有文字的部分靜音，以提升說話者辨識的精準度。

    Args:
        audio_path (str): 輸入的 WAV 檔案路徑。
        transcription_result (list): 正規化的轉錄結果（包含字典的列表，字典中需有 'start' 和 'end' 鍵）。
        output_path (str): 輸出的 WAV 檔案路徑。

    Returns:
        str: 處理後的檔案路徑。
    """
    # 讀取音訊檔案 (取樣率與音訊陣列資料)
    samplerate, data = wavfile.read(audio_path)
    
    # 建立相同長度且初始全為 0 的遮罩 (0 代表靜音，1 代表有聲音)
    if data.ndim > 1:
        mask = np.zeros(data.shape[0], dtype=np.float32)
    else:
        mask = np.zeros_like(data, dtype=np.float32)

    # 根據語音轉文字的時間標記將遮罩設為 1
    for segment in transcription_result:
        start_time = segment.get('start', 0.0)
        end_time = segment.get('end', 0.0)
        
        # 轉換秒數至陣列索引
        start_idx = int(start_time * samplerate)
        end_idx = int(end_time * samplerate)
        
        # 邊界防呆
        start_idx = max(0, min(start_idx, data.shape[0]))
        end_idx = max(start_idx, min(end_idx, data.shape[0]))
        
        mask[start_idx:end_idx] = 1.0

    # 如果是多聲道，將遮罩做 Broadcasting (讓它支援多聲道相乘)
    if data.ndim > 1:
        mask = mask[:, np.newaxis]

    # 套用遮罩並轉回原始的資料型別 (如 int16 等)
    processed_data = (data * mask).astype(data.dtype)
    
    # 將處理後的結果寫出
    wavfile.write(output_path, samplerate, processed_data)
    
    return output_path
