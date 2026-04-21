# main.py

import os
import argparse
import time

# 從我們設計的套件結構中，匯入核心的 VideoProcessor Class
from src.highlight_clipper.processor_factory import ProcessorFactory

def main():
    """
    程式主函式
    """
    # 1. 設定命令列參數解析器
    parser = argparse.ArgumentParser(description="Highlight Clipper")
    
    # 加入一個 "必須的" 參數：影片路徑
    parser.add_argument(
        "--video_path", 
        type=str, 
        default="videos/test.mp4", 
        help="要處理的影片檔案路徑"
    )
    
    # 加入一個 "可選的" 參數：指定輸出目錄
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="存放處理結果的根目錄 (預設為: output)"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"錯誤：找不到指定的影片檔案 -> {args.video_path}")
        return # 結束程式

    factory = ProcessorFactory()
    model_path = os.path.join("models", "Breeze-ASR-25-CT2")
    processor = factory.create_processor(asr_strategy="CT2",asr_model_path=model_path,diarizer_strategy="whisperx")
    processor.process(video_path=args.video_path)
    print(f"請至 '{args.output_dir}' 資料夾查看結果。")


if __name__ == "__main__":
    main()
    