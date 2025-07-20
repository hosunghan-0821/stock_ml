#!/usr/bin/env bash
# --------------------------------------------
#  📦  Data → Model 파이프라인 실행 스크립트
# --------------------------------------------
# - 가상환경 활성화 (필요 시)
# - 단계별 타임스탬프/오류 로깅
# - 어느 한 단계라도 실패하면 즉시 종료

set -euo pipefail            # 오류·미정의 변수·파이프 오류 잡기
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

echo "=== $(date) | PIPELINE START ===" | tee -a "$LOG_FILE"

# (1) 가상환경 진입 – 필요 없으면 주석 처리
# source .venv/bin/activate

# (2) 전처리
echo "[1/2] make_preprocessing.py" | tee -a "$LOG_FILE"
python app/model/make_preprocessing.py 2>&1 | tee -a "$LOG_FILE"

# (3) 모델 학습
echo "[2/2] make_model.py" | tee -a "$LOG_FILE"
python app/model/make_model.py 2>&1 | tee -a "$LOG_FILE"

echo "=== $(date) | PIPELINE DONE ===" | tee -a "$LOG_FILE"


# (4) 서비스(or 추론) 실행
echo "[3/3] app.main (inference/service)" | tee -a "$LOG_FILE"
python -m app.main 2>&1 | tee -a "$LOG_FILE"

echo "=== $(date) | PIPELINE DONE ===" | tee -a "$LOG_FILE"