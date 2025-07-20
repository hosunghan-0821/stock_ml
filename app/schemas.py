from datetime import date

from pydantic import BaseModel, Field, model_validator
from pydantic import ConfigDict


# ---------- 요청 ----------
class PredictionRequest(BaseModel):
    first_wave_return: float = Field(..., alias="1차파동수익률(%)")
    first_wave_volume_multiple: float = Field(..., alias="1차파동거래대금배율")
    material_strength: int = Field(..., alias="재료강도")
    market_condition: int = Field(..., alias="시황")

    model_config = ConfigDict(populate_by_name=True)  # alias 그대로 파싱


# ---------- 응답 ----------
class PredictionResponse(BaseModel):
    inputs: PredictionRequest  # ← 원본 입력값 echo
    predicted_adjust_pct: float = Field(  # ← 예측 조정폭
        ...,
        alias="예상조정폭(%)",
        description="모델이 예측한 조정폭(%)",
    )

    model_config = ConfigDict(populate_by_name=True,  # alias 로 직렬화
                              from_attributes=True)


# ---------- 요청 ----------
class RisePredictionRequest(BaseModel):
    """상승 구간 예측용 요청 스키마 (한글 alias 지원)"""

    ticker: str = Field(
        ..., alias="티커",
        description="종목 티커 (예: NVDA, 005930, BTC‑USDT)"
    )
    start_date: date = Field(
        ..., alias="시작일",
        description="상승 구간 시작일 (YYYY‑MM‑DD)"
    )
    end_date: date = Field(
        ..., alias="종료일",
        description="상승 구간 종료일 (YYYY‑MM‑DD)"
    )
    material_strength: int = Field(
        ..., alias="재료강도", gt=0, le=100,
        description="재료 강도 (정수 1‑10)"
    )
    market: str = Field(
        "KOSPI", alias="시장",
        description="시장(KOSPI/KOSDAQ 등)")

    @model_validator(mode="after")
    def check_dates(self):
        if self.end_date < self.start_date:
            raise ValueError("`end_date` must be on/after `start_date`")
        return self

    # alias 로 파싱·직렬화 모두 허용
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


# ---------- 응답 ----------
class RisePredictionResponse(BaseModel):
    """상승 구간 예측 결과 스키마 (한글 alias 지원)"""

    inputs: RisePredictionRequest = Field(..., alias="입력값_Echo")
    predicted_adjust_pct: float = Field(
        ..., alias="예상조정폭(%)",
        description="모델이 예측한 조정폭(%)"
    )
    predicted_adjust_days: int = Field(
        None,
        alias="예상조정일수",
        description="예측된 조정 지속 거래일 수"
    )
    name: str = Field(
        None,
        alias="종목명",
        description="종목명"
    )

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)
