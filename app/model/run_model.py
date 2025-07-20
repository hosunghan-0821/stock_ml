# run_model.py
# ─────────────────────────────────────────────────────────
# 1. 저장된 블렌딩 / 단일 모델 로드
# 2. 단일 · 배치 예측 실행 (CSV 전달 시 배치)
# ─────────────────────────────────────────────────────────
import sys
import joblib
import pandas as pd

# 0) 학습 당시 사용된 입력 컬럼 ------------------------------
FEATURES = [
    "1차파동수익률(%)",
    "1차파동거래대금배율",
    "재료강도",
    "시황",
    "상승파동일수",
    "시가총액 대비 메인 거래대금",
]

# 1) 모델 로드 ---------------------------------------------
MODEL_PATH = "pullback_blend.pkl"   # ← 파일명 변경
try:
    model_obj = joblib.load(MODEL_PATH)
    print("✓ 모델 로드 완료 :", MODEL_PATH)
except FileNotFoundError:
    sys.exit(f"모델 파일이 없습니다 → {MODEL_PATH}")

# 2) 예측 helper -------------------------------------------
def blend_predict(df: pd.DataFrame, obj):
    """df: 원본 스케일 DataFrame"""
    if isinstance(obj, dict):                       # ⓐ 블렌딩(dict)
        α   = obj["alpha"]
        xgb = obj["xgb"]; rf = obj["rf"]
        return α * xgb.predict(df) + (1 - α) * rf.predict(df)
    else:                                           # ⓑ 단일 파이프라인
        return obj.predict(df)

# 3‑A) 단일 샘플 예측 --------------------------------------
sample = pd.DataFrame([{
    "1차파동수익률(%)":            15.3,
    "1차파동거래대금배율":         2.1,
    "재료강도":                   5,
    "시황":                      7,
    "상승파동일수":                8,
    "시가총액 대비 메인 거래대금":  1.8,
}])

missing = [c for c in FEATURES if c not in sample.columns]
if missing:
    sys.exit(f"샘플에 다음 컬럼이 누락되었습니다: {missing}")

sample = sample[FEATURES]          # 열 순서 맞추기(파이프라인은 이름 매칭이기도 함)

pred = blend_predict(sample, model_obj)[0]
print(f"단일 샘플 예상 조정폭 ≈ {pred:.2f}%")