import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from app.models import get_model
from app.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])

@router.post("", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    model = Depends(get_model),
):
    """모델 예측 API"""
    # 1. 요청 → 데이터프레임 변환
    input_df = pd.DataFrame([request.model_dump(by_alias=True)])

    # 2. 모델 예측
    pred = model.predict(input_df)[0]
    pred = np.round(pred, 2)

    # 3. 결과 반환
    return PredictionResponse(
        inputs=request,
        predicted_adjust_pct=pred,
    )