import pandas as pd
import numpy as np

FEATURES = [
    "league",  # NEW: Model needs to know the league context
    "elo_diff","form_diff",
    "form_home_gf","form_home_ga",
    "form_away_gf","form_away_ga",
    "rest_home","rest_away",
    "p_home","p_draw","p_away",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True).copy()

    # --- Elo ---
    elo, HFA, K = {}, 65, 20
    def get(t): return elo.setdefault(t, 1500.0)
    elo_home, elo_away = [], []
    for _, r in df.iterrows():
        rh, ra = get(r["HomeTeam"]), get(r["AwayTeam"])
        elo_home.append(rh); elo_away.append(ra)
        exp_h = 1 / (1 + 10 ** ((ra - (rh + HFA)) / 400))
        score_h = 1 if r["FTHG"] > r["FTAG"] else 0.5 if r["FTHG"] == r["FTAG"] else 0
        gd = abs(r["FTHG"] - r["FTAG"])
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        elo[r["HomeTeam"]] = rh + K * mult * (score_h - exp_h)
        elo[r["AwayTeam"]] = ra + K * mult * ((1 - score_h) - (1 - exp_h))
    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    # --- Rolling form ---
    home = df[["Date","HomeTeam","FTHG","FTAG"]].copy()
    home.columns = ["Date","team","gf","ga"]; home["is_home"] = True
    home["match_idx"] = df.index
    away = df[["Date","AwayTeam","FTAG","FTHG"]].copy()
    away.columns = ["Date","team","gf","ga"]; away["is_home"] = False
    away["match_idx"] = df.index

    long = pd.concat([home, away]).sort_values(["team","Date"]).reset_index(drop=True)
    long["pts"] = np.where(long["gf"] > long["ga"], 3,
                    np.where(long["gf"] == long["ga"], 1, 0))
    for c in ["pts","gf","ga"]:
        long[f"form_{c}_5"] = long.groupby("team")[c].transform(
            lambda s: s.shift().rolling(5, min_periods=3).mean()
        )

    hf = long[long["is_home"]].set_index("match_idx")
    af = long[~long["is_home"]].set_index("match_idx")
    df["form_home_ppg"] = hf["form_pts_5"]
    df["form_home_gf"]  = hf["form_gf_5"]
    df["form_home_ga"]  = hf["form_ga_5"]
    df["form_away_ppg"] = af["form_pts_5"]
    df["form_away_gf"]  = af["form_gf_5"]
    df["form_away_ga"]  = af["form_ga_5"]

    df["rest_home"] = df.groupby("HomeTeam")["Date"].diff().dt.days
    df["rest_away"] = df.groupby("AwayTeam")["Date"].diff().dt.days
    df["form_diff"] = df["form_home_ppg"] - df["form_away_ppg"]
    return df