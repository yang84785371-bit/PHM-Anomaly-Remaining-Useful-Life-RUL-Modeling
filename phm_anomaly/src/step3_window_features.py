'''
    at last step that is step2 ,we prove that this question can be use model to deal with
    at that step ,we use a better method to model that quetion .
'''
# -- import packages --
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# -- raw data path --
BASE_DATA = Path("/home/didu/projects/datasets/cmapss")

# -- col --
COLS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)

# -- load data --
def load_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLS[: df.shape[1]]
    return df

# -- add label——rul -- 
def add_rul_label(train: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train.groupby("unit")["cycle"].max().rename("max_cycle")
    df = train.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df.drop(columns=["max_cycle"])

# build the window feature
def make_window_features(df: pd.DataFrame, win: int = 30) -> pd.DataFrame:
    # dertermine raw data feature
    sensor_cols = [c for c in df.columns if c.startswith("s_")]
    op_cols = [c for c in df.columns if c.startswith("op_")]
    use_cols = op_cols + sensor_cols

    # --
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit", group_keys=False) # make sure would not calculate across units

    feat = df[["unit", "cycle", "RUL"]].copy()

    rolling_mean = g[use_cols].rolling(win, min_periods=win).mean().reset_index(level=0, drop=True) # 取滑动平均值
    rolling_std  = g[use_cols].rolling(win, min_periods=win).std().reset_index(level=0, drop=True) # 取滑动误差

    feat_mean = rolling_mean.add_prefix(f"w{win}_mean_") # 命名 在前面增加一点东西
    feat_std  = rolling_std.add_prefix(f"w{win}_std_") # 命名

    # -- 局部偏离 在窗口内属于什么位置-- 
    cur = df[use_cols]
    feat_trend = (cur - rolling_mean).add_prefix(f"w{win}_trend_")

    out = pd.concat([feat, feat_mean, feat_std, feat_trend], axis=1)

    # 只保留窗口完整的行
    out = out.dropna().reset_index(drop=True)
    return out

def main():
    # -- train data --
    train = load_cmapss(BASE_DATA / "train_FD001.txt")
    train = add_rul_label(train)

    # -- extract the feature --
    feats = make_window_features(train, win=30)
    print("[INFO] window features shape:", feats.shape)

    # -- 保存特征（工程产物） -- 
    Path("artifacts").mkdir(exist_ok=True)
    feats.to_csv("artifacts/train_fd001_w30_features.csv", index=False)
    print("[OK] saved artifacts/train_fd001_w30_features.csv")

    # -- 训练一个模型做对比 -- 
    y = feats["RUL"].values #只留feature
    X = feats.drop(columns=["unit", "cycle", "RUL"]).values # 只留feature

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1) #随机森林
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)

    mae = mean_absolute_error(y_va, pred)
    mse = mean_squared_error(y_va, pred)
    rmse = mse ** 0.5

    print(f"[VAL] MAE={mae:.3f}  RMSE={rmse:.3f}")

if __name__ == "__main__":
    main()
