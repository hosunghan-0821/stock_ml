# run_model.py
# ─────────────────────────────────────────────────────────
# 1. 저장된 모델(pkl) 로드
# 2. 단일·배치 예측 실행
# ─────────────────────────────────────────────────────────
import joblib, pandas as pd, datetime as dt, sys

# 0) 학습 당시 사용된 입력 컬럼 ----------------------------
FEATURES = [
    "1차파동수익률(%)",
    "1차파동거래대금배율",
    "재료강도",
    "시황"
]   # cat_cols 가 없으므로 숫자 4개만 필요

# 1) 모델 로드 -------------------------------------------
MODEL_PATH = "pullback_rf.pkl"     # ← 파일명 맞춰 주세요
try:
    model = joblib.load(MODEL_PATH)
    print("✓ 모델 로드 완료 :", MODEL_PATH)
except FileNotFoundError:
    sys.exit(f"모델 파일이 없습니다 → {MODEL_PATH}")

# 2‑A) 단일 샘플 예측 -------------------------------------
single = pd.DataFrame([{
    "1차파동수익률(%)":     120.0,
    "1차파동거래대금배율":  20.0,
    "재료강도":              7,
    "시황":        5
}])

# ❶ 컬럼 누락 검사
missing = [c for c in FEATURES if c not in single.columns]
if missing:
    sys.exit(f"샘플에 다음 컬럼이 누락되었습니다: {missing}")

# ❷ 학습 순서대로 재정렬
single = single[FEATURES]

pred_single = model.predict(single)[0]
print(f"단일 샘플 예상 조정폭 ≈ {pred_single:.2f}%")