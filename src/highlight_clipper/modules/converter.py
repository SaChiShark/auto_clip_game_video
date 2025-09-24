# mp4_to_wav.py

import os
import subprocess

def convert_mp4_to_wav(mp4_path: str, wav_path: str = None) -> str:
    """
    使用 ffmpeg 將 MP4 檔案的音軌轉換為 16kHz 單聲道的 WAV 檔案。

    Args:
        mp4_path (str): 輸入的 MP4 檔案路徑。
        wav_path (str, optional): 輸出的 WAV 檔案路徑。
                                  如果未提供，將會以 MP4 的檔名在同目錄下生成 .wav 檔案。

    Returns:
        str: 成功轉換後的 WAV 檔案路徑。如果轉換失敗，則回傳 None。
    """
    # 檢查輸入檔案是否存在
    if not os.path.exists(mp4_path):
        print(f"錯誤：找不到輸入檔案 '{mp4_path}'")
        return None

    # 如果未指定輸出路徑，則自動生成一個
    if wav_path is None:
        # 將副檔名從 .mp4 改為 .wav
        base_name = os.path.splitext(mp4_path)[0]
        wav_path = base_name + ".wav"

    print(f"準備將 '{mp4_path}' 轉換為 '{wav_path}'...")

    # 建立 ffmpeg 指令
    # -i: 輸入檔案
    # -vn: 忽略影片軌
    # -acodec pcm_s16le: 設定音訊編碼為 16-bit PCM (WAV 標準)
    # -ar 16000: 設定音訊取樣率為 16kHz
    # -ac 1: 設定聲道為 1 (單聲道)
    # -y: 如果輸出檔案已存在，直接覆蓋
    command = [
        'ffmpeg',
        '-i', mp4_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',
        wav_path
    ]

    try:
        print("正在執行 ffmpeg 指令...")
        # 使用 subprocess 執行指令
        # capture_output=True 會捕獲 stdout 和 stderr
        # text=True 會將輸出解碼為文字
        result = subprocess.run(
            command, 
            check=True,        # 如果指令返回非零碼 (錯誤)，則會拋出 CalledProcessError
            capture_output=True,
            text=True
        )
        print("ffmpeg 標準輸出:")
        print(result.stdout)
        print("ffmpeg 標準錯誤輸出 (通常包含轉換進度):")
        print(result.stderr)
        print(f"檔案成功轉換並儲存於: {wav_path}")
        return wav_path
        
    except FileNotFoundError:
        print("錯誤：找不到 'ffmpeg' 指令。")
        print("請確認 ffmpeg 已安裝並且其路徑已加入系統的 PATH 環境變數中。")
        return None
    except subprocess.CalledProcessError as e:
        print("ffmpeg 執行時發生錯誤:")
        print(f"返回碼: {e.returncode}")
        print("ffmpeg 標準輸出:")
        print(e.stdout)
        print("ffmpeg 標準錯誤輸出:")
        print(e.stderr)
        return None

# --- 腳本獨立執行時的測試區塊 ---
if __name__ == '__main__':
    # 請將 'path/to/your/video.mp4' 替換成你自己的影片檔案路徑
    test_mp4_file = 'test.mp4'

    if test_mp4_file == 'path/to/your/video.mp4' or not os.path.exists(test_mp4_file):
        print("-" * 50)
        print("警告：請設定 `test_mp4_file` 變數為一個有效的 MP4 檔案路徑。")
        print("您也可以從命令列執行此腳本: python mp4_to_wav.py <你的影片檔案路徑>")
        print("-" * 50)
    else:
        # 執行轉換
        # 未指定輸出路徑，將在同目錄下生成 .wav 檔
        output_wav_path = convert_mp4_to_wav(test_mp4_file)

        if output_wav_path:
            print(f"\n轉換成功！音訊檔案位於: {output_wav_path}")
            # 可以在這裡檢查檔案大小等資訊
            file_size_mb = os.path.getsize(output_wav_path) / (1024 * 1024)
            print(f"檔案大小: {file_size_mb:.2f} MB")
        else:
            print("\n轉換失敗。請檢查上面的錯誤訊息。")