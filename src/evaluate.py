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
    # Load the multi-league data and model
    feat = pd.read_parquet(ROOT / "data/processed/features.parquet")
    model = joblib.load(ROOT / "models/multi_league_model.joblib")["model"]

    # We will test on the most recent season (2024-25)
    test = feat[feat["season"] == "2425"].dropna(subset=FEATURES).copy()
    y = test["result"].values
    
    # Make predictions
    p_model = model.predict_proba(test[FEATURES])
    p_odds  = test[["p_home","p_draw","p_away"]].values
    
    # Base rate (predicting the historical average for every match)
    base_probs = np.array([ (feat["result"]==i).mean() for i in range(3) ])
    p_base = np.tile(base_probs, (len(test),1))

    # 1. Calculate and save metrics per league
    metrics = []
    for league in test["league"].unique():
        mask = test["league"] == league
        if mask.sum() < 10: continue # Skip if too few games
        
        y_l = y[mask]
        m_l = p_model[mask]
        o_l = p_odds[mask]
        
        metrics.append({
            "League": league,
            "Matches": len(y_l),
            "Model Log-Loss": round(log_loss(y_l, m_l), 4),
            "Odds Log-Loss": round(log_loss(y_l, o_l), 4),
            "Model Brier": round(brier_multi(y_l, m_l), 4),
            "Odds Brier": round(brier_multi(y_l, o_l), 4),
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(ROOT / "data/processed/metrics.csv", index=False)
    print("Saved metrics to data/processed/metrics.csv")
    print(metrics_df.to_string(index=False))

    # 2. Generate and save calibration plot
    plt.figure(figsize=(8, 6))
    for cls, name in enumerate(["Home Win", "Draw", "Away Win"]):
        frac, mean = calibration_curve((y==cls).astype(int), p_model[:,cls], n_bins=10)
        plt.plot(mean, frac, marker="o", label=name)
        
    plt.plot([0,1],[0,1],"k--", label="Perfectly Calibrated")
    plt.legend()
    plt.title("Calibration Curve - Multi-League (2024-25)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Observed Frequency")
    plt.grid(alpha=0.3)
    
    plt.savefig(ROOT / "data/processed/multi_calibration.png", dpi=150, bbox_inches='tight')
    print("Saved calibration plot to data/processed/multi_calibration.png")

if __name__ == "__main__":
    main()