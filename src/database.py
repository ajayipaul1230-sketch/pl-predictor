import sqlite3
import pathlib
import pandas as pd
import requests
import os
from io import StringIO

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data"
DB_PATH = DB_DIR / "predictions.db"

def get_connection():
    """Ensures the directory exists and returns a database connection."""
    # This forces the creation of the 'data' folder if it's missing on the cloud
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    ''')
    conn.commit()
    conn.close()

def init_mapping_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS team_mappings (
            api_name TEXT PRIMARY KEY,
            csv_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_predictions(df):
    init_db()
    conn = get_connection()
    records = []
    for _, row in df.iterrows():
        records.append((
            row["Date"], row["league"], row["HomeTeam"], row["AwayTeam"],
            row["p_home"], row["p_draw"], row["p_away"]
        ))
    c = conn.cursor()
    c.executemany('''
        INSERT OR IGNORE INTO predictions 
        (match_date, league, home_team, away_team, p_home, p_draw, p_away)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', records)
    conn.commit()
    conn.close()

def save_team_mapping(api_name, csv_name):
    init_mapping_table()
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO team_mappings (api_name, csv_name) VALUES (?, ?)", (api_name, csv_name))
    conn.commit()
    conn.close()

def get_team_mappings():
    init_mapping_table()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT api_name, csv_name FROM team_mappings")
    rows = c.fetchall()
    conn.close()
    return dict(rows)

def sync_results():
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, match_date, league, home_team, away_team FROM predictions WHERE actual_result IS NULL")
    pending = c.fetchall()
    
    if not pending:
        conn.close()
        return 0
        
    league_codes = {
        "Premier League": "E0", "La Liga": "SP1", "Serie A": "I1",
        "Bundesliga": "D1", "Superliga": "DNK", "Eliteserien": "NOR"
    }
    
    updated_count = 0
    for league_name, code in league_codes.items():
        if code in ["DNK", "NOR"]:
            url = f"https://www.football-data.co.uk/new/{code}.csv"
        else:
            url = f"https://www.football-data.co.uk/mmz4281/2425/{code}.csv"
            
        try:
            df = requests.get(url).text
            df = pd.read_csv(StringIO(df), encoding="latin-1")
            df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "Res": "FTR"})
            
            for row in pending:
                db_id, match_date, l_name, h_team, a_team = row
                if l_name != league_name: continue
                
                match = df[(df["HomeTeam"] == h_team) & (df["AwayTeam"] == a_team)]
                if not match.empty and pd.notna(match.iloc[0]["FTR"]):
                    ftr = match.iloc[0]["FTR"]
                    
                    c.execute("SELECT p_home, p_draw, p_away FROM predictions WHERE id=?", (db_id,))
                    p_home, p_draw, p_away = c.fetchone()
                    
                    y_true = [1 if ftr=="H" else 0, 1 if ftr=="D" else 0, 1 if ftr=="A" else 0]
                    brier = ((p_home - y_true[0])**2 + (p_draw - y_true[1])**2 + (p_away - y_true[2])**2)
                    
                    c.execute('''
                        UPDATE predictions 
                        SET actual_result = ?, brier_score = ?
                        WHERE id = ?
                    ''', (ftr, brier, db_id))
                    updated_count += 1
        except Exception as e:
            pass
            
    conn.commit()
    conn.close()
    return updated_count

def get_all_predictions():
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY match_date DESC", conn)
    conn.close()
    return df