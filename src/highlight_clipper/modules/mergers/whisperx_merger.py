import pandas as pd
from typing import List, Dict, Any
from .base import BaseMerger
from ..transcribers.base import NormalizedTranscription

class WhisperXMerger(BaseMerger):
    """
    使用 WhisperX assign_word_speakers 的 Merger 策略。
    """
    def __init__(self, max_sentence_pause: float = 0.8):
        """
        初始化 WhisperXMerger。

        Args:
            max_sentence_pause (float): 同一個說話者話語間的最大停頓秒數。預設 0.8。
        """
        self.max_sentence_pause = max_sentence_pause

    def run(self, 
            transcription_result: NormalizedTranscription, 
            diarization_result: List[Dict[str, Any]],
            **kwargs) -> List[Dict[str, Any]]:
        """
        呼叫 whisperx.assign_word_speakers 來完成映射。
        """
        import whisperx
        
        if not transcription_result:
            return []

        # 1. 轉換 diarization_result 為 whisperX 需要的 DataFrame 格式
        df_data = []
        for i, d in enumerate(diarization_result):
            # 保護機制，避免有些外部實作可能沒給 start 或 end 導致錯誤
            start = d.get('start')
            end = d.get('end')
            if start is not None and end is not None:
                df_data.append({
                    'segment': i,
                    'label': d['speaker'],
                    'speaker': d['speaker'],
                    'start': start,
                    'end': end
                })
        
        if not df_data:
            return self._group_words_to_sentences(transcription_result)

        diarize_df = pd.DataFrame(df_data)

        # 2. 轉換 transcription_result 為 whisperX 需要的 dict 格式
        transcript_result = {
            "segments": [
                {
                    "start": transcription_result[0]["start"],
                    "end": transcription_result[-1]["end"],
                    "text": " ".join([w["word"] for w in transcription_result]),
                    "words": transcription_result
                }
            ],
            "word_segments": transcription_result
        }

        # 3. 呼叫 whisperx 作指派
        # 該函式會將 'speaker' 欄位更新回 transcript_result 的每一個 word dict 當中
        whisperx.assign_word_speakers(diarize_df, transcript_result)

        # 4. 提出分配完畢的 words
        words_with_speakers = transcript_result["segments"][0]["words"]
        
        # 容錯：有時如果完全配對不上，whisperX 不會寫入 speaker
        for w in words_with_speakers:
            if 'speaker' not in w:
                w['speaker'] = 'UNKNOWN'

        # 5. 按照原有的邏輯把單字合成句子
        sentences = self._group_words_to_sentences(words_with_speakers)
        
        return sentences

    def _group_words_to_sentences(self, 
                                  words_with_speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not words_with_speakers:
            return []

        sentences = []
        current_sentence_words = []
        
        current_sentence_words.append(words_with_speakers[0])
        
        for i in range(1, len(words_with_speakers)):
            current_word = words_with_speakers[i]
            previous_word = words_with_speakers[i-1]
            
            speaker_changed = current_word.get('speaker', 'UNKNOWN') != previous_word.get('speaker', 'UNKNOWN')
            pause_exceeded = (current_word['start'] - previous_word['end']) > self.max_sentence_pause
            
            if speaker_changed or pause_exceeded:
                sentences.append(self._finalize_sentence(current_sentence_words))
                current_sentence_words = [current_word]
            else:
                current_sentence_words.append(current_word)
        
        if current_sentence_words:
            sentences.append(self._finalize_sentence(current_sentence_words))
            
        return sentences

    def _finalize_sentence(self, words: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not words:
            return {}
        
        sentence_text = ' '.join([word['word'] for word in words])
        
        return {
            'speaker': words[0].get('speaker', 'UNKNOWN'),
            'text': sentence_text,
            'start': words[0]['start'],
            'end': words[-1]['end'],
        }
