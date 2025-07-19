from pydantic import BaseModel, Field, ConfigDict

# ---------- 요청 ----------
class PredictionRequest(BaseModel):
    first_wave_return:          float = Field(..., alias="1차파동수익률(%)")
    first_wave_volume_multiple: float = Field(..., alias="1차파동거래대금배율")
    material_strength:          int   = Field(..., alias="재료강도")
    market_condition:           int   = Field(..., alias="시황")

    model_config = ConfigDict(populate_by_name=True)   # alias 그대로 파싱

# ---------- 응답 ----------
class PredictionResponse(BaseModel):
    inputs: PredictionRequest                     # ← 원본 입력값 echo
    predicted_adjust_pct: float = Field(          # ← 예측 조정폭
        ...,
        alias="예상조정폭(%)",
        description="모델이 예측한 조정폭(%)",
    )

    model_config = ConfigDict(populate_by_name=True,  # alias 로 직렬화
                              from_attributes=True)
