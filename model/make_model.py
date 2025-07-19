# make_model.py
# ───────────────────────────────────────────────────────────
# 1) 데이터 로드 → 2) 전처리 정의 → 3) Random Forest + K‑Fold
# 4) 홀드아웃 평가 → 5) 모델 저장 → 6) 중요도 시각화 → 7) 새 샘플 예측
# ───────────────────────────────────────────────────────────

# ───── 라이브러리
import pandas as pd, numpy as np, joblib, datetime, matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupShuffleSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
RANDOM_SEED  = 42
CSV_PATH     = "pullback_dataset.csv"

# ────────────────── 1) 데이터 로드/검토 ───────────────────
df = pd.read_csv(CSV_PATH, parse_dates=["1차고점일"])
print(f"총 {len(df)}행 | 미리보기:\n{df.head(2)}\n")

print("현재 열 목록:", list(df.columns))


y = df[["조정폭(%)", "고점→저점_경과거래일"]]
X = df.drop(columns=[
    "조정폭(%)", "티커", "종목명",
    "고점→저점_경과거래일", "고점→반등_경과거래일", "저점→반등_경과일"
    ,"VIX점수(1~10)","RSI점수(1~10)","시가총액"
])                                               # 입력 피처
num_cols = ["1차파동수익률(%)","1차파동거래대금배율",
            "재료강도","시황","상승파동일수","시가총액 대비 메인 거래대금"]
cat_cols = []

# ────────────────── 2) 전처리 파이프라인 ────────────────
preprocess = ColumnTransformer([
    ("num", StandardScaler(),             num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ────────────────── 3) Random Forest + K‑Fold ───────────
# rf_pipe = Pipeline([
#     ("prep", preprocess),
#     ("rf"  , RandomForestRegressor(random_state=RANDOM_SEED))
# ])
#
# param_rf = {
#     "rf__n_estimators": [300, 500],
#     "rf__max_depth"  : [None, 10, 20]
# }

# ────────────────── 3) xgboost + K‑Fold ───────────
xgb_pipe = Pipeline([
    ("prep", preprocess),
    ("xgb", XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        tree_method="hist",      # CPU 히스토그램(빠름)
        n_jobs=-1
    ))
])

param_xgb = {
    "xgb__n_estimators"      : [400, 700],
    "xgb__learning_rate"     : [0.05, 0.1],
    "xgb__max_depth"         : [3, 5],
    "xgb__subsample"         : [0.6, 0.8, 1.0],
    "xgb__colsample_bytree"  : [0.6, 0.8, 1.0]
}


kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

xgb_cv = GridSearchCV(
    xgb_pipe, param_xgb,
    cv=kfold, scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1
).fit(X, y)

print(f"[XGB] CV MAE { -xgb_cv.best_score_ :.2f} | best {xgb_cv.best_params_}")
best_model = xgb_cv.best_estimator_

# ────────────────── 4) 홀드아웃 평가 ───────────────────
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups=df["티커"]))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"[Hold‑out] MAE {mae:.2f} | RMSE {rmse:.2f}")

# ────────────────── 5) 모델 저장 ───────────────────────
fname = f"pullback_rf.pkl"
joblib.dump(best_model, fname)
print("모델 저장 완료 →", fname)

# ────────────────── 6) 피처 중요도 시각화 ───────────────
import numpy as np, matplotlib.pyplot as plt

# 1) 파이프라인에서 'prep' 을 제외한 실제 모델 단계 이름 찾기
model_key = [k for k in best_model.named_steps if k != "prep"][0]
native    = best_model.named_steps[model_key]

# 2) 중요도 벡터 가져오기
if hasattr(native, "feature_importances_"):            # 트리 계열
    importances = native.feature_importances_
    title_base  = f"{model_key.upper()} Feature Importance"
elif hasattr(native, "coef_"):                         # 선형계열
    importances = np.abs(native.coef_)                 # 계수 절댓값
    title_base  = f"{model_key.upper()} |coef|"
else:
    raise ValueError("이 모델은 중요도/계수를 지원하지 않습니다.")

# 3) 전처리 후 컬럼 이름
feature_names = best_model.named_steps["prep"].get_feature_names_out()

# 4) Top‑N (실제 피처 수가 10개보다 적으면 자동 조정)
N = min(10, len(importances))
top_idx = np.argsort(importances)[-N:][::-1]

# 5) 그리기
plt.figure(figsize=(7, 4))
plt.barh(range(N), importances[top_idx][::-1])
plt.yticks(range(N), [feature_names[i] for i in top_idx][::-1])
plt.gca().invert_yaxis()                       # 큰 값이 위로 오도록
plt.xlabel("Importance")
plt.title(f"{title_base}  (Top {N})")
plt.tight_layout()
plt.show()

sample = pd.DataFrame([{
    "1차파동수익률(%)"      : 15.3,   # 예: 1차 파동 수익률 15.3 %
    "1차파동거래대금배율"   : 2.1,    # 예: 평균 대비 2.1배
    "재료강도"              : 5,      # 예: 내부 스코어
    "시황"                  : 7,      # 예: 1~10 점수
    "상승파동일수"          : 8,      # 예: 시작→고점 8 거래일
    "시가총액 대비 메인 거래대금" : 1.8   # 예: 시총 대비 1.8 %
}])

# 3) 예측 (출력 shape = (1, 2)  →  [조정폭(%), 고점→저점_경과거래일])
pred = best_model.predict(sample)
adj_pct, days_peak_to_low = pred[0]

print(f"예측 조정폭   : {adj_pct:6.2f} %")
print(f"예측 조정기간 : {days_peak_to_low:5.0f} 거래일")