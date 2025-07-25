from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH: Path = Path(__file__).resolve().parent / "model/pullback_blend.pkl"

@lru_cache(maxsize=1)
def get_model():
    """애플리케이션 시작 시 1회만 모델을 메모리에 적재."""
    return joblib.load(MODEL_PATH)