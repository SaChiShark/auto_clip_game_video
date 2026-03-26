from typing import List, Dict, Any

# 匯入我們在 transcribers/base.py 中定義的標準化格式
from .transcribers.base import NormalizedTranscription

class Merger:
    """
    一個專門負責將標準化的語音辨識結果和說話者辨識結果合併的類別。
    """
    def __init__(self, max_sentence_pause: float = 0.8):
        """
        初始化 Merger。

        Args:
            max_sentence_pause (float): 
                同一個說話者，話語之間的最大停頓秒數。
                如果超過這個時間，即使是同一個人，也會被切分為兩個句子。
                預設為 0.8 秒。
        """
        self.max_sentence_pause = max_sentence_pause

    def run(self, 
              transcription_result: NormalizedTranscription, 
              diarization_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        執行合併的核心公開方法。

        Args:
            transcription_result (NormalizedTranscription): 
                來自 ASR Transcriber 的標準化詞語列表。
                格式: [{'word': str, 'start': float, 'end': float}, ...]
            diarization_result (List[Dict[str, Any]]): 
                來自 Diarizer 的說話者時間區段列表。
                格式: [{'speaker': str, 'start': float, 'end': float}, ...]

        Returns:
            List[Dict[str, Any]]:
                合併後的句子列表。
                格式: [{'speaker': str, 'text': str, 'start': float, 'end': float, 'words': [...]}, ...]
        """
        if not transcription_result:
            return [] # 如果沒有辨識出任何文字，直接回傳空列表

        # 步驟 1: 為每一個詞分配說話者
        words_with_speakers = self._assign_speaker_to_words(transcription_result, diarization_result)
        
        # 步驟 2: 將帶有說話者標籤的詞語聚合成句子
        sentences = self._group_words_to_sentences(words_with_speakers)
        
        return sentences

    def _assign_speaker_to_words(self, 
                                 words: NormalizedTranscription, 
                                 speaker_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        私有方法：為每個詞找到對應的說話者。
        優化版：使用最大重疊時間 (Max Overlap) 與最近距離容錯機制。
        """
        for word in words:
            word_start = word['start']
            word_end = word['end']
            
            best_speaker = 'UNKNOWN'
            max_overlap = 0.0
            min_distance = float('inf')
            closest_speaker = 'UNKNOWN'

            for turn in speaker_turns:
                turn_start = turn['start']
                turn_end = turn['end']
                
                # 計算重疊時間
                overlap_start = max(word_start, turn_start)
                overlap_end = min(word_end, turn_end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn['speaker']
                
                # 若無重疊，計算詞跟 turn 的最短距離，以防詞語剛好落在縫隙中
                if overlap == 0:
                    if word_end <= turn_start:
                        dist = turn_start - word_end
                    else:
                        dist = word_start - turn_end
                        
                    if dist < min_distance:
                        min_distance = dist
                        closest_speaker = turn['speaker']

            # 如果有重疊就用重疊最多的，如果沒有任何重疊，但距離很近 (例如相距不到 0.5 秒)，就用最近的
            if max_overlap > 0:
                word['speaker'] = best_speaker
            elif min_distance < 0.5: 
                word['speaker'] = closest_speaker
            else:
                word['speaker'] = 'UNKNOWN'
                
        return words

    def _group_words_to_sentences(self, 
                                  words_with_speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        私有方法：將詞語聚合成句子。
        """
        if not words_with_speakers:
            return []

        sentences = []
        current_sentence_words = []
        
        # 從第一個詞開始
        current_sentence_words.append(words_with_speakers[0])
        
        for i in range(1, len(words_with_speakers)):
            current_word = words_with_speakers[i]
            previous_word = words_with_speakers[i-1]
            
            speaker_changed = current_word['speaker'] != previous_word['speaker']
            pause_exceeded = (current_word['start'] - previous_word['end']) > self.max_sentence_pause
            
            # 如果說話者改變，或停頓時間過長，就結束當前句子
            if speaker_changed or pause_exceeded:
                # 將目前累積的詞語打包成一個句子
                sentences.append(self._finalize_sentence(current_sentence_words))
                # 開始一個新句子
                current_sentence_words = [current_word]
            else:
                # 繼續在當前句子中添加詞語
                current_sentence_words.append(current_word)
        
        # 不要忘記處理最後一個句子
        if current_sentence_words:
            sentences.append(self._finalize_sentence(current_sentence_words))
            
        return sentences

    def _finalize_sentence(self, words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        私有方法：將詞語列表打包成一個標準化的句子字典。
        """
        if not words:
            return {}
        
        # 將詞語的文字內容用空白連接起來
        sentence_text = ' '.join([word['word'] for word in words])
        
        return {
            'speaker': words[0]['speaker'],
            'text': sentence_text,
            'start': words[0]['start'],
            'end': words[-1]['end'],
            #'words': words # 保留原始的詞語資訊
        }


# --- 腳本獨立執行時的測試區塊 ---
if __name__ == '__main__':
    # 1. 準備模擬的測試資料
    mock_transcription = [
        {'word': '你好', 'start': 1.1, 'end': 1.5},
        {'word': '啊', 'start': 1.5, 'end': 1.8},
        {'word': '我是', 'start': 3.2, 'end': 3.5},
        {'word': 'B', 'start': 3.5, 'end': 4.0},
        {'word': '沒錯', 'start': 4.5, 'end': 5.0}, # 長時間停頓
        {'word': '就是我', 'start': 6.1, 'end': 6.8},
        {'word': 'CDE', 'start': 8.0, 'end': 9.0} # UNKNOWN
    ]
    
    mock_diarization = [
        {'speaker': 'SPEAKER_00', 'start': 0.5, 'end': 2.5},
        {'speaker': 'SPEAKER_01', 'start': 3.0, 'end': 7.0}
    ]
    
    # 2. 建立 Merger 物件
    # 我們可以用不同的停頓時間來測試
    merger = Merger(max_sentence_pause=1.0)
    
    # 3. 執行合併
    final_sentences = merger.run(mock_transcription, mock_diarization)
    
    # 4. 漂亮地印出結果
    print("--- 合併結果 ---")
    import json
    print(json.dumps(final_sentences, indent=2, ensure_ascii=False))

    """
    預期的輸出結果:
    [
      {
        "speaker": "SPEAKER_00",
        "text": "你好 啊",
        "start": 1.1,
        "end": 1.8,
        "words": [...]
      },
      {
        "speaker": "SPEAKER_01",
        "text": "我是 B",
        "start": 3.2,
        "end": 4.0,
        "words": [...]
      },
      {
        "speaker": "SPEAKER_01",
        "text": "沒錯",
        "start": 4.5,
        "end": 5.0,
        "words": [...]
      },
      {
        "speaker": "SPEAKER_01",
        "text": "就是我",
        "start": 6.1,
        "end": 6.8,
        "words": [...]
      },
      {
        "speaker": "UNKNOWN",
        "text": "CDE",
        "start": 8.0,
        "end": 9.0,
        "words": [...]
      }
    ]
    """