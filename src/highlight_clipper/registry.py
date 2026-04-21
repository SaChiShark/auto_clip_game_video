from typing import Callable, Any, Dict

class ComponentRegistry:
    """
    元件註冊表，用於依賴注入與策略模式的動態尋找。
    """
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str):
        """
        裝飾器，用來將一個產生器函數註冊到註冊表。
        """
        def decorator(creator_func: Callable[..., Any]):
            if key in self._registry:
                print(f"警告：{self.name} 註冊表中的鍵值 '{key}' 已被覆蓋。")
            self._registry[key] = creator_func
            return creator_func
        return decorator

    def create(self, key: str, **kwargs) -> Any:
        """
        根據給定的 key 執行對應的產生器函數，並帶入參數。
        """
        if key not in self._registry:
            raise ValueError(f"{self.name}: 未知的策略 '{key}'。可用策略: {list(self._registry.keys())}")
        
        creator_func = self._registry[key]
        return creator_func(**kwargs)

# 系統中各個維度的全局註冊表
asr_registry = ComponentRegistry("ASR Transcriber")
diarization_registry = ComponentRegistry("Diarizer")
merger_registry = ComponentRegistry("Merger")
