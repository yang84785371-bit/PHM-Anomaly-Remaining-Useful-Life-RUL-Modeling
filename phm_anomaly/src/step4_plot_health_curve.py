'''

'''
# -- package install --
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# try xgboost
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

# --split the set of train and valid --
def split_by_unit(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = int(len(units) * val_ratio)
    val_units = set(units[:n_val])
    tr = df[~df["unit"].isin(val_units)].copy()
    va = df[df["unit"].isin(val_units)].copy()
    return tr, va

# -- fit the model and use the metric to assess --
def fit_and_eval(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    mae = mean_absolute_error(y_va, pred)
    rmse = mean_squared_error(y_va, pred) ** 0.5
    return mae, rmse


def main():
    # -- load the maked window data --
    feat_path = Path("artifacts/train_fd001_w30_features.csv")
    df = pd.read_csv(feat_path)
    
    # -- remove the useless cols and build the X and Y --
    feat_cols = [c for c in df.columns if c not in ["unit", "cycle", "RUL"]]
    X = df[feat_cols].values
    y = df["RUL"].values

    # ---- split by unit (avoid leakage) ----
    tr_df, va_df = split_by_unit(df, val_ratio=0.2, seed=42)
    X_tr, y_tr = tr_df[feat_cols].values, tr_df["RUL"].values
    X_va, y_va = va_df[feat_cols].values, va_df["RUL"].values

    results = []

    '''
        two part: random forest and XGBoost 
    '''
    # ---- RF ----
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    mae_rf, rmse_rf = fit_and_eval(rf, X_tr, y_tr, X_va, y_va)
    results.append(("RF", mae_rf, rmse_rf))

    # ---- XGBoost ----
    if XGBRegressor is None:
        print("[WARN] xgboost not installed. Run: pip install xgboost")
        xgb = None
    else:
        xgb = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        mae_xgb, rmse_xgb = fit_and_eval(xgb, X_tr, y_tr, X_va, y_va)
        results.append(("XGB", mae_xgb, rmse_xgb))

    # ---- print & save metrics ----
    metrics_df = pd.DataFrame(results, columns=["model", "MAE", "RMSE"]).sort_values("MAE")
    print(metrics_df.to_string(index=False))

    Path("artifacts").mkdir(exist_ok=True)
    metrics_df.to_csv("artifacts/model_compare_trainval.csv", index=False)
    print("[OK] saved artifacts/model_compare_trainval.csv")

    # ---- plot health curve (pick one unit from validation set, to be fair) ----
    unit_id = int(va_df["unit"].unique()[0])
    one = va_df[va_df["unit"] == unit_id].sort_values("cycle").copy()
    X_one = one[feat_cols].values
    # ---- true RUL from failure cycle (train set only) ----
    failure_cycle = one["cycle"].max()
    one["true_rul"] = failure_cycle - one["cycle"]

    # refit models on full training split (tr_df) for plotting
    rf.fit(X_tr, y_tr)
    one["pred_rul_rf"] = rf.predict(X_one)

    eps = 1e-6
    one["health_rf"] = (one["pred_rul_rf"] - one["pred_rul_rf"].min()) / (
        one["pred_rul_rf"].max() - one["pred_rul_rf"].min() + eps
    )
    one["health_true"] = (one["true_rul"] - one["true_rul"].min()) / (
        one["true_rul"].max() - one["true_rul"].min() + eps
    )

    plt.figure()

    # true health
    plt.plot(one["cycle"], one["health_true"], label="True", linewidth=2)

    # RF
    plt.plot(one["cycle"], one["health_rf"], label="RF")

    # XGB
    if xgb is not None:
        if xgb is not None: 
            xgb.fit(X_tr, y_tr)
            one["pred_rul_xgb"] = xgb.predict(X_one) 
            one["health_xgb"] = (one["pred_rul_xgb"] - one["pred_rul_xgb"].min()) / ( one["pred_rul_xgb"].max() - one["pred_rul_xgb"].min() + eps )
        plt.plot(one["cycle"], one["health_xgb"], label="XGB")

    plt.xlabel("cycle")
    plt.ylabel("health score (0~1)")
    plt.title(f"Health curve (unit={unit_id}) - True vs RF vs XGB")
    plt.legend()
    plt.tight_layout()

    out_png = "artifacts/health_curve_unit_compare_with_truth.png"
    plt.savefig(out_png, dpi=200)
    print(f"[OK] saved {out_png}")



if __name__ == "__main__":
    main()
