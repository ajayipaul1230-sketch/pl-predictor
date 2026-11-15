import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from features import FEATURES

def brier_multi(y, p):
    return ((p - np.eye(3)[y]) ** 2).sum(axis=1).mean()

def main():
    feat = pd.read_parquet(ROOT / "data/processed/features.parquet")
    model = joblib.load(ROOT / "models/pl_model.joblib")["model"]

    for test_season in ["2223","2324","2425"]:
        test = feat[feat["season"] == test_season].dropna(subset=FEATURES)
        y = test["result"].values
        p_model = model.predict_proba(test[FEATURES])
        p_odds  = test[["p_home","p_draw","p_away"]].values
        base = np.tile([ (feat["result"]==i).mean() for i in range(3) ], (len(test),1))

        print(f"--- {test_season} (n={len(test)}) ---")
        for name, p in [("model",p_model),("odds",p_odds),("base",base)]:
            print(f"  {name:5s}  logloss={log_loss(y,p):.4f}  brier={brier_multi(y,p):.4f}")

    test = feat[feat["season"]=="2425"].dropna(subset=FEATURES)
    p_model = model.predict_proba(test[FEATURES])
    y = test["result"].values
    plt.figure(figsize=(7, 6))
    for cls, name in enumerate(["Home","Draw","Away"]):
        frac, mean = calibration_curve((y==cls).astype(int), p_model[:,cls], n_bins=8)
        plt.plot(mean, frac, marker="o", label=name)
    plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.title("Calibration 2024-25")
    plt.xlabel("Predicted"); plt.ylabel("Observed")
    plt.savefig(ROOT / "data/processed/calibration.png")
    print("Saved calibration plot to data/processed/calibration.png")

if __name__ == "__main__":
    main()