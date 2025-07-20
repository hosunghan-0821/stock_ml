from __future__ import annotations
from datetime import datetime, timedelta, date
from typing import Literal

import numpy as np
import pandas as pd

from pykrx import stock
import yfinance as yf

# ───── 기존 유틸 그대로 재사용 (fetch_ohlcv·market_scores 등) ─────
from app.model.make_preprocessing import fetch_ohlcv, market_scores, get_market_cap, get_ticker_name

LOOK_BACK_VOL = 20  # 거래대금 평균 산출 구간


def get_name_from_ticker(ticker: str, market: str) -> str:
    return get_ticker_name(ticker, market)


def get_peak_price(ticker: str, market: str, peak_start: date,peak_end:date) -> float:
    df = fetch_ohlcv(ticker, market, peak_start, peak_end)
    ts_key = pd.Timestamp(peak_end)
    return df.at[ts_key, "고가"]



def first_wave_features(
        *,
        ticker: str,
        market: Literal["KOSPI", "KOSDAQ"],
        date_start: date | str,
        date_peak: date | str,
        material_strength: int,
) -> pd.DataFrame:
    """
    ① 1차 파동 수익률(%)
    ② 1차 파동 거래대금배율
    ③ 재료강도
    ④ 시황 점수 (VIX·RSI 가중)
    ⑤ 상승파동일수 (= 시작→고점 거래일 수)
    ⑥ 시가총액 대비 메인 거래대금(%)

    반환: 입력 스키마와 동일한 컬럼 순서를 가진 **단일‑행 DataFrame**.
    """
    d0 = pd.to_datetime(date_start).normalize()
    d1 = pd.to_datetime(date_peak).normalize()
    if d1 <= d0:
        raise ValueError("date_peak must be after date_start")

    # 1) OHLCV 확보 (시점 버퍼 포함)
    df = fetch_ohlcv(ticker, market, d0 - timedelta(days=LOOK_BACK_VOL), d1)

    # 2) 가격·거래대금 지표
    price_start = df.at[d0, "시가"]
    price_peak = df.at[d1, "고가"]

    first_wave_ret = (price_peak / price_start - 1) * 100  # ①
    vol_mean_1st = df.loc[d0:d1, "거래대금"].mean()
    vol_mean_hist = df.loc[d0 - timedelta(days=LOOK_BACK_VOL):d0 - timedelta(days=1),
                    "거래대금"].mean()
    first_wave_vol_mul = vol_mean_1st / vol_mean_hist  # ②

    # 3) 메인 거래대금/시총 (%)
    main_vol = df.loc[d0:d1, "거래대금"].max()
    mcap = get_market_cap(ticker, d1) or np.nan
    main_vol_vs_mcap_pct = (main_vol / mcap * 100) if mcap else np.nan  # ⑥

    # 4) 시황 점수
    *_unused, market_score = market_scores(d0, d1, market)  # ④

    # 5) 1차 파동 거래일 수
    days_start_to_peak = df.index.get_loc(d1) - df.index.get_loc(d0)  # ⑤

    # 6) DataFrame 구성 (모델 입력용 alias 이름 순서 준수)
    sample = pd.DataFrame([{
        "1차파동수익률(%)": round(first_wave_ret, 2),
        "1차파동거래대금배율": round(first_wave_vol_mul, 2),
        "재료강도": int(material_strength),
        "시황": int(market_score),
        "상승파동일수": int(days_start_to_peak),
        "시가총액 대비 메인 거래대금": round(main_vol_vs_mcap_pct, 2),
    }])

    return sample
