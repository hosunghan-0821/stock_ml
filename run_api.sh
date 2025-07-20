#!/usr/bin/env bash


# (4) 서비스(or 추론) 실행
echo "[3/3] app.main (inference/service)" | tee -a "$LOG_FILE"
python -m app.main 2>&1 | tee -a "$LOG_FILE"

echo "=== $(date) | PIPELINE DONE ===" | tee -a "$LOG_FILE"