'''
    this file we use to transform industry time_series problem to a problem of rul regression.
    and we use the rf to be a baseline to prove that problem is worked in ml
'''
# -- import packages --
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# -- raw data --
BASE_DATA = Path("/home/didu/projects/datasets/cmapss")

# -- cols --
COLS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)

# -- load raw data --
def load_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLS[: df.shape[1]]
    return df

# -- calculate the label --
def add_rul_label(train: pd.DataFrame) -> pd.DataFrame:
    # 每个 unit 的最大 cycle 就是失效点
    max_cycle = train.groupby("unit")["cycle"].max().rename("max_cycle") # gain the max cycle of each unit 
    df = train.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"] # gain the remain unit life
    df = df.drop(columns=["max_cycle"]) # delete useless col
    return df

def main():
    # -- load data --
    train = load_cmapss(BASE_DATA / "train_FD001.txt")
    # -- calculate label and add it  --
    train = add_rul_label(train)

    # -- feture cols -- 
    feat_cols = [c for c in train.columns if c.startswith("op_") or c.startswith("s_")]
    X = train[feat_cols].values # make feature value as X
    y = train["RUL"].values # make RUL as Y

    # -- that func is of scikit,just be used to split  train and test --
    # -- here rf is a baseline --
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # -- choose rf as the baseline model --
    model = RandomForestRegressor(
        n_estimators=300, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_va)
    mae = mean_absolute_error(y_va, pred)
    mse = mean_squared_error(y_va, pred)
    rmse = mse ** 0.5

    print("[OK] baseline trained.")
    print(f"[VAL] MAE={mae:.3f}  RMSE={rmse:.3f}")
    print("[INFO] train rows:", train.shape[0], "features:", len(feat_cols))

if __name__ == "__main__":
    main()
