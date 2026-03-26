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
    parser = argparse.ArgumentParser(description="語音精華自動剪輯器 (Highlight Clipper)")
    
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

    # 2. 檢查輸入的影片檔案是否存在
    if not os.path.exists(args.video_path):
        print(f"錯誤：找不到指定的影片檔案 -> {args.video_path}")
        return # 結束程式

    # --- 核心流程開始 ---
    
    start_time = time.time()
    
    #try:
    # 3. 初始化 VideoProcessor 引擎
    # 這一步會比較久，因為它會載入所有 AI 模型
    factory = ProcessorFactory()
    model_path = os.path.join("models", "Breeze-ASR-25-CT2")
    processor = factory.create_processor(asr_strategy="CT2",asr_model_path=model_path,diarizer_strategy="whisperx")
    # 4. 呼叫 process 方法，啟動處理流程
    processor.process(video_path=args.video_path)

    #except Exception as e:
    #    print(f"在處理過程中發生未預期的嚴重錯誤: {e}")
    
    end_time = time.time()
    print(f"\n全部流程執行完畢，總耗時: {end_time - start_time:.2f} 秒。")
    print(f"請至 '{args.output_dir}' 資料夾查看結果。")


if __name__ == "__main__":
    # 當這個腳本被直接執行時，呼叫 main 函式
    main()
    