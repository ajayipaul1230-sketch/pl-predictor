import os
import pathlib
import pandas as pd
import requests
from io import StringIO
from sqlalchemy import create_engine, text
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)

def get_engine():
    """Connects to Supabase if available, otherwise uses local SQLite."""
    try:
        db_url = st.secrets.get("DATABASE_URL")
    except Exception:
        db_url = None
        
    if not db_url:
        db_url = os.getenv("DATABASE_URL")
        
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    else:
        # Fallback to local SQLite file
        return create_engine(f"sqlite:///{DB_DIR}/predictions.db")

def init_db():
    engine = get_engine()
    is_postgres = engine.name == "postgresql"
    id_type = "SERIAL PRIMARY KEY" if is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
    
    with engine.begin() as conn:
        conn.execute(text(f'''
            CREATE TABLE IF NOT EXISTS predictions (
                id {id_type},
                match_date TEXT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                p_home REAL,
                p_draw REAL,
                p_away REAL,
                actual_result TEXT,
                brier_score REAL,
                UNIQUE(home_team, away_team, match_date)
            )
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS team_mappings (
                api_name TEXT PRIMARY KEY,
                csv_name TEXT
            )
        '''))

def save_predictions(df):
    init_db()
    engine = get_engine()
    is_postgres = engine.name == "postgresql"
    
    insert_cmd = "INSERT INTO predictions (match_date, league, home_team, away_team, p_home, p_draw, p_away) VALUES (:match_date, :league, :home_team, :away_team, :p_home, :p_draw, :p_away)"
    if is_postgres:
        insert_cmd += " ON CONFLICT (home_team, away_team, match_date) DO NOTHING"
    else:
        insert_cmd = insert_cmd.replace("INSERT INTO", "INSERT OR IGNORE INTO")
        
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text(insert_cmd), {
                "match_date": row["Date"], "league": row["league"], 
                "home_team": row["HomeTeam"], "away_team": row["AwayTeam"],
                "p_home": row["p_home"], "p_draw": row["p_draw"], "p_away": row["p_away"]
            })

def save_team_mapping(api_name, csv_name):
    init_db()
    engine = get_engine()
    is_postgres = engine.name == "postgresql"
    cmd = '''
        INSERT INTO team_mappings (api_name, csv_name) VALUES (:api_name, :csv_name)
        ON CONFLICT (api_name) DO UPDATE SET csv_name = EXCLUDED.csv_name
    ''' if is_postgres else "INSERT OR REPLACE INTO team_mappings (api_name, csv_name) VALUES (:api_name, :csv_name)"
        
    with engine.begin() as conn:
        conn.execute(text(cmd), {"api_name": api_name, "csv_name": csv_name})

def get_team_mappings():
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT api_name, csv_name FROM team_mappings")).fetchall()
    return {row[0]: row[1] for row in result}

def sync_results():
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        pending = conn.execute(text("SELECT id, match_date, league, home_team, away_team FROM predictions WHERE actual_result IS NULL")).fetchall()
    
    if not pending: return 0
        
    league_codes = {"Premier League": "E0", "La Liga": "SP1", "Serie A": "I1", "Bundesliga": "D1", "Superliga": "DNK", "Eliteserien": "NOR"}
    updated_count = 0
    
    for league_name, code in league_codes.items():
        url = f"https://www.football-data.co.uk/new/{code}.csv" if code in ["DNK", "NOR"] else f"https://www.football-data.co.uk/mmz4281/2425/{code}.csv"
        try:
            df = pd.read_csv(StringIO(requests.get(url).text), encoding="latin-1")
            df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "Res": "FTR"})
            with engine.begin() as conn:
                for row in pending:
                    db_id, match_date, l_name, h_team, a_team = row
                    if l_name != league_name: continue
                    match = df[(df["HomeTeam"] == h_team) & (df["AwayTeam"] == a_team)]
                    if not match.empty and pd.notna(match.iloc[0]["FTR"]):
                        ftr = match.iloc[0]["FTR"]
                        p_row = conn.execute(text("SELECT p_home, p_draw, p_away FROM predictions WHERE id=:id"), {"id": db_id}).fetchone()
                        p_home, p_draw, p_away = p_row
                        y_true = [1 if ftr=="H" else 0, 1 if ftr=="D" else 0, 1 if ftr=="A" else 0]
                        brier = ((p_home - y_true[0])**2 + (p_draw - y_true[1])**2 + (p_away - y_true[2])**2)
                        conn.execute(text("UPDATE predictions SET actual_result=:ftr, brier_score=:brier WHERE id=:id"), {"ftr": ftr, "brier": brier, "id": db_id})
                        updated_count += 1
        except Exception:
            pass
    return updated_count

def get_all_predictions():
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM predictions ORDER BY match_date DESC"), conn)
    return df