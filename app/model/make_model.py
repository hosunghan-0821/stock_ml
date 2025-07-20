# make_model.py
# ───────────────────────────────────────────────────────────
# 1) 데이터 로드 → 2) 전처리 정의 → 3) Random Forest + K‑Fold
# 4) 홀드아웃 평가 → 5) 모델 저장 → 6) 중요도 시각화 → 7) 새 샘플 예측
# ───────────────────────────────────────────────────────────
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent          # /.../app/model
SRC =  BASE_DIR / "first_wave_input.csv"
# SRC = "debug.csv"
CSV_PATH = BASE_DIR / "pullback_dataset.csv"
# DEST = "dubug_dataset.csv"


# ────────────────── 1) 데이터 로드/검토 ───────────────────
df = pd.read_csv(CSV_PATH, parse_dates=["1차고점일"])
print(f"총 {len(df)}행 | 미리보기:\n{df.head(2)}\n")

print("현재 열 목록:", list(df.columns))


y = df[["조정폭(%)"]]
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
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ))
])

param_rf = {
    "rf__n_estimators": [400, 600, 800],
    "rf__max_depth"   : [None, 10, 20]
}


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

rf_cv = GridSearchCV(
    rf_pipe, param_rf,
    cv=kfold, scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1
).fit(X, y.values.ravel())


xgb_cv = GridSearchCV(
    xgb_pipe, param_xgb,
    cv=kfold, scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1
).fit(X, y)

print(f"[RF ] CV MAE {-rf_cv.best_score_:.2f} | best {rf_cv.best_params_}")
best_rf  = rf_cv.best_estimator_
best_xgb = xgb_cv.best_estimator_          # 기존 변수명 best_model → best_xgb 로 변경

print(f"[XGB] CV MAE { -xgb_cv.best_score_ :.2f} | best {xgb_cv.best_params_}")
best_model = xgb_cv.best_estimator_

# ────────────────── 4) 홀드아웃 평가 & 블렌딩 ───────────────
# ────────────────── 3) 모델별 CV 성적 로그 ──────────────────
print(f"[RF ] CV MAE {-rf_cv.best_score_ :.2f} | best {rf_cv.best_params_}")
best_rf  = rf_cv.best_estimator_

print(f"[XGB] CV MAE {-xgb_cv.best_score_: .2f} | best {xgb_cv.best_params_}")
best_xgb = xgb_cv.best_estimator_

# ────────────────── 4) 홀드아웃 평가 & 블렌딩 ───────────────
# ① 종목(Group) 기반 1회 Split
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups=df["티커"]))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ② 두 모델 학습 & 개별 성능
best_rf .fit (X_train, y_train.values.ravel())
best_xgb.fit(X_train, y_train.values.ravel())

pred_rf  = best_rf .predict(X_test).ravel()
pred_xgb = best_xgb.predict(X_test).ravel()

mae_rf  = mean_absolute_error(y_test, pred_rf)
mae_xgb = mean_absolute_error(y_test, pred_xgb)
print(f"[Hold‑out] RF  MAE {mae_rf :.2f}")
print(f"[Hold‑out] XGB MAE {mae_xgb:.2f}")

# ③ α 가중치 탐색 (0.05 단위로 촘촘히)
alphas   = np.linspace(0.05, 0.95, 19)
mae_list = [mean_absolute_error(y_test, a*pred_xgb + (1-a)*pred_rf)
            for a in alphas]

alpha_best   = float(alphas[np.argmin(mae_list)])
y_pred_blend = alpha_best*pred_xgb + (1-alpha_best)*pred_rf

mae_blend  = mean_absolute_error(y_test, y_pred_blend)
rmse_blend = np.sqrt(mean_squared_error(y_test, y_pred_blend))
print(f"[Blend α={alpha_best:.2f}] MAE {mae_blend:.2f} | RMSE {rmse_blend:.2f}")

# ────────────────── 5) 모델 저장 ───────────────────────
fname = "pullback_blend.pkl"                       # ← 파일명도 의미에 맞게
blend_obj = {"xgb": best_xgb,
             "rf" : best_rf,
             "alpha": alpha_best}
joblib.dump(blend_obj, fname)
print("블렌딩 모델 저장 완료 →", fname)

# ────────────────── 6) 피처 중요도 (XGB 기준) ───────────────
#   ⊳ 블렌딩 내에서 '설명력'은 보통 XGB가 담당하므로 XGB 중요도 출력
model_for_imp = best_xgb                           # ← XGB만 사용
model_key     = [k for k in model_for_imp.named_steps if k != "prep"][0]
native        = model_for_imp.named_steps[model_key]

if hasattr(native, "feature_importances_"):        # 트리 계열
    importances = native.feature_importances_
    title_base  = f"{model_key.upper()} Feature Importance"
elif hasattr(native, "coef_"):
    importances = np.abs(native.coef_)
    title_base  = f"{model_key.upper()} |coef|"
else:
    raise ValueError("이 모델은 중요도/계수를 지원하지 않습니다.")

feature_names = model_for_imp.named_steps["prep"].get_feature_names_out()

N        = min(10, len(importances))
top_idx  = np.argsort(importances)[-N:][::-1]
feat_nm  = [feature_names[i] for i in top_idx]
feat_sc  = importances[top_idx]

w_feat = max(len(f) for f in feat_nm)
w_rank = len(str(N))
border = "-" * (w_feat + 20)

print(border)
print(f"{title_base}  (Top {N})")
print(f"{'Rank':>{w_rank}}  {'Feature'.ljust(w_feat)}  Importance")
for r, (f, s) in enumerate(zip(feat_nm, feat_sc), 1):
    print(f"{r:>{w_rank}}  {f.ljust(w_feat)}  {s:>10.4f}")
print(border)

# ────────────────── 7) 샘플 예측 (블렌딩) ───────────────────
sample = pd.DataFrame([{
    "1차파동수익률(%)"          : 15.3,
    "1차파동거래대금배율"       : 2.1,
    "재료강도"                  : 5,
    "시황"                      : 7,
    "상승파동일수"              : 8,
    "시가총액 대비 메인 거래대금": 1.8
}])

α = blend_obj["alpha"]
pred = (
    α * blend_obj["xgb"].predict(sample) +
    (1-α) * blend_obj["rf"].predict(sample)
)
print(f"예측 조정폭 : {pred[0]:6.2f} %")