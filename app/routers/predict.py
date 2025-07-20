import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from app.features import first_wave_features, get_name_from_ticker, get_peak_price
from app.models import get_model  # ← 의존성 주입용 팩토리
from app.schemas import (  # ← 방금 만든 스키마들
    PredictionRequest, PredictionResponse,
    RisePredictionRequest, RisePredictionResponse,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
def predict(
        request: PredictionRequest,
        model=Depends(get_model),
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


# -----------------------------------------------------------
# 2) 상승 구간 조정폭·조정일 예측   POST /predict/rise
# -----------------------------------------------------------
@router.post(
    "/rise",
    response_model=RisePredictionResponse,
    summary="지정 구간 조정 예측",
    description=(
            "ticker, 구간(start_date~end_date), 재료 강도를 입력받아 "
            "**예상 조정폭(%)** 과 **조정 지속 거래일 수** 둘 다 반환합니다."
    ),
)
def predict_rise(
        request: RisePredictionRequest,
        model=Depends(get_model),
):
    """상승 구간: 조정폭·조정일 동시 예측"""

    # 1) 피처 DF
    X = first_wave_features(
        ticker=request.ticker,
        market=request.market,
        date_start=request.start_date,
        date_peak=request.end_date,
        material_strength=request.material_strength,
    )
    name = get_name_from_ticker(request.ticker, request.market)
    peak_price = get_peak_price(request.ticker, request.market, request.start_date, request.end_date)
    from tabulate import tabulate
    print(
        tabulate(
            X, headers="keys",
            tablefmt="pretty",  # "github", "fancy_grid" 등도 가능
            showindex=False,
            floatfmt=".3f"
        )
    )

    # 2) 예측
    # 2) 예측 --------------------------------------------------
    if isinstance(model, dict):  # ⓐ 블렌딩 저장 구조
        α = model["alpha"]
        xgb = model["xgb"]
        rf = model["rf"]

        pred_pct = (
                α * xgb.predict(X) +
                (1 - α) * rf.predict(X)
        )[0]  # 첫 샘플
    else:  # ⓑ 단일 파이프라인
        pred_pct = model.predict(X)[0]

    pred_pct = float(np.round(pred_pct, 2))
    adjusted_price = int(round(peak_price * (1 - pred_pct / 100), 2))

    return RisePredictionResponse(
        inputs=request,
        predicted_adjust_pct=pred_pct,
        name=name,
        predicted_adjust_price=adjusted_price
    )
