import pandas as pd
import pathlib
import warnings

# Ignore the pandas performance warnings to keep the terminal clean
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/all_leagues.parquet"

SEASONS = ["1617","1718","1819","1920","2021","2122","2223","2324","2425"]
MAIN_LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga"}
EXTRA_LEAGUES = {"DNK": "Superliga", "NOR": "Eliteserien"}

def process_df(df, season, league_name):
    # Standardize extra league column names
    rename_map = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR"}
    df = df.rename(columns=rename_map)
    
    # Drop problematic columns
    cols_to_drop = ["Season", "Season_Year", "Time"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    df["season"] = str(season)
    df["league"] = league_name
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    
    # Force goals to be numbers
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    
    # Drop rows with missing dates or scores
    df = df.dropna(subset=["Date", "FTHG", "FTAG"]).copy()
    
    # Standardize odds and FORCE them to be numbers
    for col, fallbacks in {
        "odds_h": ["PSCH", "PSH", "AvgH", "B365H"],
        "odds_d": ["PSCD", "PSD", "AvgD", "B365D"],
        "odds_a": ["PSCA", "PSA", "AvgA", "B365A"]
    }.items():
        for fb in fallbacks:
            if fb in df.columns:
                df[col] = pd.to_numeric(df[fb], errors="coerce")
                break
                
    return df

def main():
    frames = []
    
    # 1. Download Main Leagues
    for code, name in MAIN_LEAGUES.items():
        for s in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv"
            try:
                df = pd.read_csv(url, encoding="latin-1")
                frames.append(process_df(df, s, name))
                print(f"Downloaded {name} {s}")
            except Exception as e:
                pass # Some seasons might be missing

    # 2. Download Extra Leagues (Nordic)
    for code, name in EXTRA_LEAGUES.items():
        url = f"https://www.football-data.co.uk/new/{code}.csv"
        try:
            df = pd.read_csv(url, encoding="latin-1")
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["Season_Year"] = df["Date"].dt.year.astype(str)
            
            for s in SEASONS:
                start_year = "20" + s[:2]
                end_year = "20" + s[2:]
                mask = (df["Season_Year"] == start_year) | (df["Season_Year"] == end_year)
                if mask.any():
                    frames.append(process_df(df[mask], s, name))
            print(f"Downloaded {name} history")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

    raw = pd.concat(frames, ignore_index=True)
    
    # Calculate Implied Probabilities
    inv = 1/raw["odds_h"] + 1/raw["odds_d"] + 1/raw["odds_a"]
    raw["p_home"] = (1/raw["odds_h"]) / inv
    raw["p_draw"] = (1/raw["odds_d"]) / inv
    raw["p_away"] = (1/raw["odds_a"]) / inv
    
    raw["result"] = raw["FTR"].map({"H":0,"D":1,"A":2})
    
    # Drop rows missing odds or results
    raw = raw.dropna(subset=["p_home", "p_draw", "p_away", "result"])
    raw = raw.sort_values("Date").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(OUT)
    print(f"SUCCESS! Wrote {len(raw)} total matches across all leagues to {OUT}")

if __name__ == "__main__":
    main()