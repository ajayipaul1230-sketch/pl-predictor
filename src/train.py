import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from features import build_features, FEATURES

def main():
    raw = pd.read_parquet(ROOT / "data/raw/all_leagues.parquet")
    feat = build_features(raw)
    feat.to_parquet(ROOT / "data/processed/features.parquet")

    train = feat.dropna(subset=FEATURES).copy()
    X, y = train[FEATURES], train["result"]

    # Create a preprocessor to one-hot encode the 'league' column
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['league'])],
        remainder='passthrough'
    )

    # Create the full pipeline
    base_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000))
    ])

    # Calibrate the pipeline
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X, y)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, ROOT / "models/multi_league_model.joblib")
    print(f"Trained on {len(train)} matches across all leagues -> models/multi_league_model.joblib")

if __name__ == "__main__":
    main()