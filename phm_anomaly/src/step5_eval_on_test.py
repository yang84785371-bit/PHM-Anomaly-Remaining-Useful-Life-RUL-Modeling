# src/step5_eval_on_test.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


BASE_DATA = Path("/home/didu/projects/datasets/cmapss")

COLS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)


def load_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLS[: df.shape[1]]
    return df


def make_window_features_test(df: pd.DataFrame, win: int = 30) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("s_")]
    op_cols = [c for c in df.columns if c.startswith("op_")]
    use_cols = op_cols + sensor_cols

    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit", group_keys=False)

    base = df[["unit", "cycle"]].copy()

    rolling_mean = g[use_cols].rolling(win, min_periods=win).mean().reset_index(level=0, drop=True)
    rolling_std  = g[use_cols].rolling(win, min_periods=win).std().reset_index(level=0, drop=True)

    feat_mean = rolling_mean.add_prefix(f"w{win}_mean_")
    feat_std  = rolling_std.add_prefix(f"w{win}_std_")
    feat_trend = (df[use_cols] - rolling_mean).add_prefix(f"w{win}_trend_")

    out = pd.concat([base, feat_mean, feat_std, feat_trend], axis=1)
    out = out.dropna().reset_index(drop=True)  # only full windows
    return out


def eval_model(name: str, model, X_tr, y_tr, X_test_last, y_test_true):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_test_last)
    mae = mean_absolute_error(y_test_true, pred)
    rmse = mean_squared_error(y_test_true, pred) ** 0.5
    return name, mae, rmse


def main():
    # ---- load train window features (already labeled) ----
    train_feat = pd.read_csv("artifacts/train_fd001_w30_features.csv")
    feat_cols = [c for c in train_feat.columns if c not in ["unit", "cycle", "RUL"]]
    X_tr = train_feat[feat_cols].values
    y_tr = train_feat["RUL"].values

    # ---- build test window features from raw test ----
    test_raw = load_cmapss(BASE_DATA / "test_FD001.txt")
    test_feat = make_window_features_test(test_raw, win=30)

    # Keep only last available row per unit (the point we predict RUL for)
    test_last = test_feat.sort_values(["unit", "cycle"]).groupby("unit").tail(1).copy()

    # Align columns to train feature columns (safety)
    X_test_last = test_last[feat_cols].values

    # ---- true RUL for each test unit ----
    rul_true = pd.read_csv(BASE_DATA / "RUL_FD001.txt", header=None, names=["RUL_true"])
    # Ensure same order: unit 1..100
    test_last = test_last.sort_values("unit").reset_index(drop=True)
    y_test_true = rul_true["RUL_true"].values

    assert len(test_last) == len(y_test_true), "Mismatch: test units vs RUL rows"

    results = []

    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    results.append(eval_model("RF", rf, X_tr, y_tr, X_test_last, y_test_true))

    if XGBRegressor is None:
        print("[WARN] xgboost not installed. Run: pip install xgboost")
    else:
        xgb = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        results.append(eval_model("XGB", xgb, X_tr, y_tr, X_test_last, y_test_true))

    out = pd.DataFrame(results, columns=["model", "MAE", "RMSE"]).sort_values("MAE")
    print(out.to_string(index=False))
    Path("artifacts").mkdir(exist_ok=True)
    out.to_csv("artifacts/model_compare_test.csv", index=False)
    print("[OK] saved artifacts/model_compare_test.csv")


if __name__ == "__main__":
    main()
