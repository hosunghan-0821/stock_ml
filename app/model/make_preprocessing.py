from typing import Optional

import pandas as pd, numpy as np, yfinance as yf
from pandas_ta import rsi
from pykrx import stock
from datetime import datetime, timedelta
from tqdm import tqdm
import logging, sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True,  # 이미 설정돼 있어도 덮어쓰기 (Py ≥ 3.8)
)

# ── 파라미터 ───────────────────────────────────────────────
LOOK_BACK_VOL = 20  # 평균 거래대금 창
API_WIN_EXTRA = 30  # 버퍼 일수 (rebound 뒤까지 확보)

BASE_DIR = Path(__file__).resolve().parent          # /.../app/model
SRC =  BASE_DIR / "first_wave_input.csv"
# SRC = "debug.csv"
DEST = BASE_DIR / "pullback_dataset.csv"
# DEST = "dubug_dataset.csv"


# ────────────────── 전역(캐시) 시계열 초기화 ──────────────────
VIX_HIST = pd.Series(dtype="float64")  # ^VIX 1y 종가
_INDEX_CACHE: dict[str, pd.Series] = {}  # 심벌별 종가 시계열


def get_market_cap(ticker: str, date: pd.Timestamp) -> Optional[float]:
    """
    해당 날짜(영업일)의 시가총액(원)을 반환.
    • 데이터가 없으면 None 을 돌려줌.
    """
    d = date.strftime("%Y%m%d")
    df = stock.get_market_cap_by_date(d, d, ticker)
    if df.empty:
        return None
    return float(df["시가총액"].iloc[0])


def _download_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    야후파이낸스에서 종가만 받아 Series 반환 (종목 코드·날짜·auto_adjust 고정)
    """
    df = yf.download(symbol, start=start, end=end + pd.Timedelta(days=1),
                     auto_adjust=False, progress=False)["Close"]
    return df.squeeze()  # 1‑열 DF → Series


# ── 최소 구간만 받아서 이어 붙이는 헬퍼 ──────────────────
def _ensure_range(series: Optional[pd.Series],
                  symbol: str,
                  start: pd.Timestamp,
                  end: pd.Timestamp) -> pd.Series:
    if series is None or series.empty:
        dl = yf.download(symbol, start=start,
                         end=end + pd.Timedelta(days=1),
                         auto_adjust=False, progress=False)["Close"].squeeze()
        return dl

    need_front = start < series.index[0]
    need_back = end > series.index[-1]

    if not (need_front or need_back):
        return series  # 이미 범위 충족

    dl_start = start if need_front else series.index[-1] + pd.Timedelta(days=1)
    dl_end = series.index[0] - pd.Timedelta(days=1) if need_front else end

    # ───── 안전 다운로드 & 병합 래퍼 ───────────────────────────────
    dl = yf.download(
        symbol,
        start=dl_start,
        end=dl_end + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False
    )["Close"]

    # ▸ dl 이 스칼라(float)로 떨어지는 경우가 있다
    if np.isscalar(dl):
        dl = pd.Series({dl_start: dl})  # 날짜‑값 한 줄짜리 Series 로 래핑

    # ▸ 기존 series 가 None 이면 빈 Series 로 초기화
    if series is None:
        series = pd.Series(dtype="float64")

    series = (
        pd.concat([series, dl])  # Series + Series → concat
        .sort_index()
        .drop_duplicates(keep="last")
    )
    return series


# ────────────────── MAIN: 시황 점수 ──────────────────────────
# ── 간단 RSI (순수 pandas) ──────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────────────────────
def market_scores(d1: pd.Timestamp, d2: pd.Timestamp, market: str
                  ) -> tuple[int, int, int]:
    """
    market : "KOSPI", "KOSDAQ" 등 (대소문자 무관)
    반환   : (vix_score, rsi_score, market_score)  모두 1~10
    """
    global VIX_HIST, _INDEX_CACHE

    # ─ ① VIX 캐시 보강 ───────────────────────────────────────
    VIX_HIST = _ensure_range(
        VIX_HIST, "^VIX", d1 - pd.Timedelta(days=365), d2
    )

    # ─ ② 지수 심벌 선택 & 캐시 ───────────────────────────────
    market = market.upper()
    symbol = "^KS11" if market == "KOSPI" else "^KQ11"  # 기본 KOSDAQ
    ser = _INDEX_CACHE.get(symbol)
    ser = _ensure_range(
        ser, symbol, d1 - pd.Timedelta(days=60), d2
    )
    _INDEX_CACHE[symbol] = ser  # 캐시 업데이트

    # ─ ③ VIX 점수(역스케일) ─────────────────────────────────
    vix_window = VIX_HIST.loc[d1 : d2 - pd.Timedelta(days=1)]
    vix_mean   = vix_window.mean().item()                 # .item() → 스칼라
    decile     = (VIX_HIST < vix_mean).mean().item()      # 0‑1 스칼라
    vix_score  = 11 - int(np.ceil(decile * 10))
    vix_score  = max(1, min(vix_score, 10))               # 1~10 보장

    # ─ ④ RSI 점수(0~100 → 1~10) ────────────────────────────
    rsi_series = rsi(ser, 14)
    rsi_avg    = (
        rsi_series.loc[d1 : d2 - pd.Timedelta(days=1)]
        .mean()
        .item()
    )
    rsi_score  = int(np.clip(round((rsi_avg - 30) / 7) + 1, 1, 10))

    # ─ ⑤ 가중 평균 (국내 RSI 80 % : 해외 VIX 20 %) ────────────
    market_score = int(np.clip(round(0.2 * vix_score + 0.8 * rsi_score), 1, 10))

    return vix_score, rsi_score, market_score
# ─────────────────────────────────────────────────────────────


def get_ticker_name(ticker: str, market: str) -> str:
    """티커·시장코드로 종목명을 돌려준다."""
    try:
        return stock.get_market_ticker_name(ticker)
    except Exception as e:
        print(f"[경고] {ticker} 종목명 조회 실패: {e}")
        return "N/A"


# ── 도우미: 국내 거래대금 ────────────────────
def fetch_ohlcv(ticker, market, start, end):
    d0, d1 = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    df = stock.get_market_ohlcv_by_date(d0, d1, ticker)
    df.index = pd.to_datetime(df.index).normalize()
    if "거래대금" not in df.columns or df["거래대금"].isnull().all():
        # 평균가 ≈ (시가+고가+저가+종가)/4
        avg_price = df[["시가", "고가", "저가", "종가"]].mean(axis=1)
        df["거래대금"] = avg_price * df["거래량"]

    # pykrx 일부 구간은 거래대금이 'object' 일 수 있으니 float 변환
    df["거래대금"] = df["거래대금"].astype(float)
    return df


# ── 메인 로직 ──────────────────────────────────────────────
def make_preprocessing():
    rows = []
    src = pd.read_csv(SRC, dtype={"ticker": str, "market": str},
                      parse_dates=["date_first_start", "date_first_peak", "date_rebound"])

    for r in tqdm(src.itertuples(index=False), total=len(src)):
        d0, d1, d2 = r.date_first_start, r.date_first_peak, r.date_rebound
        start = d0 - timedelta(days=LOOK_BACK_VOL)
        end = d2 + timedelta(days=API_WIN_EXTRA)
        logging.info(f"START {r.종목명:<6} {d0.date()}→{d2.date()}")

        # ───────────── fetch 단계만 보호 ─────────────
        try:
            df = fetch_ohlcv(r.ticker, r.market, start, end)
        except Exception as e:
            logging.info(f"[{r.ticker}, {r.종목명}] fetch_ohlcv 실패 → 건너뜀")
            continue  # ★ 다음 종목으로
        # 가격 값
        price_first_start = df.at[d0, "시가"]
        price_first_peak = df.at[d1, "고가"]

        low_zone = df.loc[d1:d2].iloc[1:]
        low_idx = low_zone["저가"].idxmin()
        price_pull_low = low_zone.at[low_idx, "저가"]

        price_rebound = df.at[d2, "고가"]
        main_vol = df.loc[d0:d1, "거래대금"].max()
        reb_vol_ratio = df.at[d2, "거래대금"] / main_vol

        # 시가총액
        cap_peak = get_market_cap(r.ticker, d1)  # 1차 고점일 시총
        cap_peak_ek = None if cap_peak is 0 else cap_peak // 100_000_000
        main_vol_vs_mcap_pct = main_vol / cap_peak * 100

        # 1차 파동 거래일 수  =  d0에서 d1까지 인덱스 위치 차이
        days_start_to_peak = df.index.get_loc(d1) - df.index.get_loc(d0)

        # 파생 변수
        first_wave_ret = (price_first_peak / price_first_start - 1) * 100
        first_wave_body = (price_first_peak - price_first_start) / price_first_start
        first_wave_vol_rel = df.loc[d0:d1, "거래대금"].mean() / \
                             df.loc[d0 - timedelta(days=LOOK_BACK_VOL):d0 - timedelta(days=1),
                             "거래대금"].mean()

        vol_recent = df.loc[d0:d1, "거래대금"].mean()
        vol_hist = df.loc[d0 - timedelta(days=LOOK_BACK_VOL):d0 - timedelta(days=1),
                   "거래대금"].mean()

        retr_pct = (price_first_peak - price_pull_low) / price_first_peak * 100
        # 날짜
        days_to_reb = df.index.get_loc(d2) - df.index.get_loc(low_idx)
        days_peak_to_reb = df.index.get_loc(d2) - df.index.get_loc(d1)
        days_peak_to_low = df.index.get_loc(low_idx) - df.index.get_loc(d1)

        reb_ratio = (price_rebound / price_pull_low - 1) * 100  # 15.0 (%)

        vix_score, rsi_score, market_score = market_scores(d0, d1, r.market)

        name = get_ticker_name(r.ticker, r.market)

        rows.append([
            r.ticker,  # 1  티커
            name,  # 2  종목명
            d0.date(),  # 3  1차 시작일
            d1.date(),  # 4  1차 고점일
            low_idx.date(),  # 5  눌림 저점일
            d2.date(),  # 6  반등일
            round(price_rebound, 2),  # 7  반등 종가
            round(first_wave_ret, 2),  # 8  1차 파동 수익률(%)
            round(first_wave_vol_rel, 2),  # 9  1차 파동 거래대금배율
            round(retr_pct, 2),  # 10 조정폭(%)
            days_peak_to_low,  # ★ 고점→저점 거래일
            days_peak_to_reb,  # ← 11  고점→반등 거래일
            days_to_reb,  # 11 저점→반등 경과일
            round(reb_ratio, 2),  # 12 반등(%)
            round(reb_vol_ratio, 2),  # 13 메인대비 반등 거래대금 비율
            market_score,  # 14 시황
            r.재료강도,  # 15 재료강도
            vix_score,  # 16 VIX 점수
            rsi_score,  # 17 RSI 점수
            cap_peak_ek,
            days_start_to_peak,
            main_vol_vs_mcap_pct
        ])

    # ── 저장 ───────────────────────────────────────────────────
    cols = [
        "티커",  # 1
        "종목명",  # 2
        "1차시작일",  # 3
        "1차고점일",  # 4
        "눌림저점일",  # 5
        "반등일",  # 6
        "반등종가",  # 7
        "1차파동수익률(%)",  # 8
        "1차파동거래대금배율",  # 9
        "조정폭(%)",  # 10
        "고점→저점_경과거래일",  # ★ 새 컬럼
        "고점→반등_경과거래일",  # ★ 새 컬럼
        "저점→반등_경과일",  # 11
        "반등(%)",  # 12
        "메인거래대금 대비 반등거래대금비율",  # 13
        "시황",  # 14
        "재료강도",  # 15  ← ★ 쉼표 추가
        "VIX점수(1~10)",  # 16
        "RSI점수(1~10)",  # 17
        "시가총액",  # 18
        "상승파동일수",
        "시가총액 대비 메인 거래대금"
    ]
    pd.DataFrame(rows, columns=cols).to_csv(DEST, index=False)
    print(f"✓ {DEST} 생성 완료")

if __name__ == "__main__":
    make_preprocessing()
