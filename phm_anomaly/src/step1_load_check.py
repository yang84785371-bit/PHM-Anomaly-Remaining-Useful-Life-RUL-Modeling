'''
    this file we use to load and check the data.
    just check, and we would load again and process the raw data at step2 
'''
# -- import some packages --
import pandas as pd
from pathlib import Path

# -- raw data address -
BASE_DATA = Path("/home/didu/projects/datasets/cmapss")

# feature 
COLS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)

# -- load data --
def load_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None) # read csv
    df = df.dropna(axis=1, how="all") # drop na axis=(1row)
    df.columns = COLS[: df.shape[1]]
    return df

def main():
    # --load --
    train = load_cmapss(BASE_DATA / "train_FD001.txt")
    test  = load_cmapss(BASE_DATA / "test_FD001.txt")
    rul   = pd.read_csv(BASE_DATA / "RUL_FD001.txt", header=None, names=["RUL"])
    # -- check --
    print("[TRAIN]", train.shape, "units:", train["unit"].nunique())
    print("[TEST ]", test.shape,  "units:", test["unit"].nunique())
    print("[RUL  ]", rul.shape)

    # -- assert --
    assert train.isna().sum().sum() == 0
    assert test.isna().sum().sum() == 0
    assert rul.shape[0] == test["unit"].nunique()

    print("[OK] data loaded and checked.")

if __name__ == "__main__":
    main()
