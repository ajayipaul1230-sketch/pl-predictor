import os, pathlib, requests, pandas as pd
from dotenv import load_dotenv
import streamlit as st
from features import FEATURES
from database import get_team_mappings, save_predictions

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv()

LEAGUE_API_MAP = {
    "Premier League": "soccer_epl", "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a", "Bundesliga": "soccer_germany_bundesliga",
    "Superliga": "soccer_denmark_superliga", "Eliteserien": "soccer_norway_eliteserien"
}

TEAM_NAME_MAP = {
    # Premier League
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth", 
    "Brentford": "Brentford", "Brighton and Hove Albion": "Brighton", "Chelsea": "Chelsea", 
    "Crystal Palace": "Crystal Palace", "Everton": "Everton", "Fulham": "Fulham", 
    "Ipswich Town": "Ipswich", "Leicester City": "Leicester", "Liverpool": "Liverpool", 
    "Manchester City": "Man City", "Manchester United": "Man United", "Newcastle United": "Newcastle", 
    "Nottingham Forest": "Nott'm Forest", "Southampton": "Southampton", "Tottenham Hotspur": "Tottenham", 
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",

    # La Liga
    "Deportivo Alaves": "Alaves", "Athletic Bilbao": "Ath Bilbao", "Atletico Madrid": "Ath Madrid", 
    "Barcelona": "Barcelona", "Real Betis": "Betis", "Celta Vigo": "Celta", "Espanyol": "Espanyol", 
    "Getafe": "Getafe", "Girona": "Girona", "Las Palmas": "Las Palmas", "Leganes": "Leganes", 
    "RCD Mallorca": "Mallorca", "CA Osasuna": "Osasuna", "Rayo Vallecano": "Vallecano", 
    "Real Madrid": "Real Madrid", "Real Sociedad": "Sociedad", "Sevilla FC": "Sevilla", 
    "Valencia CF": "Valencia", "Real Valladolid": "Valladolid", "Villarreal CF": "Villarreal",

    # Serie A
    "Atalanta BC": "Atalanta", "Bologna FC": "Bologna", "Cagliari": "Cagliari", "Como": "Como", 
    "Empoli": "Empoli", "ACF Fiorentina": "Fiorentina", "Genoa": "Genoa", "Hellas Verona": "Verona", 
    "Inter Milan": "Inter", "Juventus FC": "Juventus", "SS Lazio": "Lazio", "Lecce": "Lecce", 
    "AC Milan": "Milan", "Monza": "Monza", "SSC Napoli": "Napoli", "Parma": "Parma", 
    "AS Roma": "Roma", "Torino FC": "Torino", "Udinese Calcio": "Udinese", "Venezia": "Venezia",

    # Bundesliga
    "FC Augsburg": "Augsburg", "Bayer Leverkusen": "Leverkusen", "Bayern Munich": "Bayern Munich", 
    "VfL Bochum": "Bochum", "Borussia Dortmund": "Dortmund", "Eintracht Frankfurt": "Ein Frankfurt", 
    "SC Freiburg": "Freiburg", "1. FC Heidenheim": "Heidenheim", "TSG Hoffenheim": "Hoffenheim", 
    "Holstein Kiel": "Holstein Kiel", "RB Leipzig": "RB Leipzig", "Borussia Monchengladbach": "M'gladbach", 
    "Mainz 05": "Mainz", "FC St. Pauli": "St Pauli", "VfB Stuttgart": "Stuttgart", 
    "1. FC Union Berlin": "Union Berlin", "Werder Bremen": "Werder Bremen", "VfL Wolfsburg": "Wolfsburg",

    # Superliga (Denmark)
    "Aalborg BK": "Aalborg", "AGF Aarhus": "Aarhus", "Brondby IF": "Brondby", 
    "FC Copenhagen": "Copenhagen", "Lyngby BK": "Lyngby", "FC Midtjylland": "Midtjylland", 
    "FC Nordsjaelland": "Nordsjaelland", "Randers FC": "Randers", "Silkeborg IF": "Silkeborg", 
    "SonderjyskE": "SonderjyskE", "Vejle BK": "Vejle", "Viborg FF": "Viborg",

    # Eliteserien (Norway)
    "Bodo/Glimt": "Bodo/Glimt", "SK Brann": "Brann", "Fredrikstad": "Fredrikstad", 
    "HamKam": "HamKam", "FK Haugesund": "Haugesund", "KFUM Oslo": "KFUM", 
    "Kristiansund BK": "Kristiansund", "Lillestrom": "Lillestrom", "Molde FK": "Molde", 
    "Odds BK": "Odd", "Rosenborg BK": "Rosenborg", "Sandefjord": "Sandefjord", 
    "Sarpsborg 08": "Sarpsborg", "Stromsgodset": "Stromsgodset", "Tromso": "Tromso", 
    "Viking FK": "Viking"
}
# Add these to your TEAM_NAME_MAP:
    "Atlético Madrid": "Ath Madrid",
    "Alavés": "Alaves",
    "Espanyol": "Espanyol",
    "Bodø/Glimt": "Bodo/Glimt",
    "Vålerenga": "Valerenga",
    "FSV Mainz 05": "Mainz",
    "1. FC Köln": "FC Koln",     # If Koln is in your data
    "Deportivo La Coruña": "La Coruna", 
    # If they are in your data

def get_implied_probs(h, d, a):
    inv = 1/h + 1/d + 1/a
    return (1/h)/inv, (1/d)/inv, (1/a)/inv

def get_latest_team_stats(team, feat_df):
    home = feat_df[feat_df["HomeTeam"] == team].tail(1)
    away = feat_df[feat_df["AwayTeam"] == team].tail(1)
    if home.empty and away.empty: return None
    if not home.empty and (away.empty or home["Date"].iloc[0] > away["Date"].iloc[0]):
        return {"elo": home["elo_home"].iloc[0], "ppg": home["form_home_ppg"].iloc[0], "gf": home["form_home_gf"].iloc[0], "ga": home["form_home_ga"].iloc[0]}
    else:
        return {"elo": away["elo_away"].iloc[0], "ppg": away["form_away_ppg"].iloc[0], "gf": away["form_away_gf"].iloc[0], "ga": away["form_away_ga"].iloc[0]}

def fetch_live_predictions(model, feat_df):
    # Safe API key loading for both local and cloud
    try: api_key = st.secrets.get("ODDS_API_KEY")
    except Exception: api_key = os.getenv("ODDS_API_KEY")

    if not api_key:
        return pd.DataFrame({"Error": ["API Key not found. Check .env or Streamlit Secrets."]}), []

    db_mappings = get_team_mappings()
    master_map = {**TEAM_NAME_MAP, **db_mappings}
    all_predictions = []
    unmapped_teams = set()

    for league_name, sport_key in LEAGUE_API_MAP.items():
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=uk&markets=h2h"
        try: res = requests.get(url, timeout=10)
        except Exception: continue
        
        if res.status_code != 200: continue
        data = res.json()
        if not data: continue

        for match in data:
            home_raw, away_raw = match["home_team"], match["away_team"]
            home_team, away_team = master_map.get(home_raw, home_raw), master_map.get(away_raw, away_raw)
            h_stats, a_stats = get_latest_team_stats(home_team, feat_df), get_latest_team_stats(away_team, feat_df)
            
            if not h_stats: unmapped_teams.add((home_raw, league_name)); continue
            if not a_stats: unmapped_teams.add((away_raw, league_name)); continue

            h_odds, d_odds, a_odds = [], [], []
            for bookmaker in match["bookmakers"]:
                for market in bookmaker["markets"]:
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_raw: h_odds.append(outcome["price"])
                            elif outcome["name"] == away_raw: a_odds.append(outcome["price"])
                            elif outcome["name"] == "Draw": d_odds.append(outcome["price"])
            
            if not h_odds or not d_odds or not a_odds: continue

            p_home, p_draw, p_away = get_implied_probs(sum(h_odds)/len(h_odds), sum(d_odds)/len(d_odds), sum(a_odds)/len(a_odds))
            row = {
                "league": league_name, "Date": pd.to_datetime(match["commence_time"]).strftime("%Y-%m-%d %H:%M"),
                "HomeTeam": home_team, "AwayTeam": away_team,
                "elo_diff": h_stats["elo"] - a_stats["elo"], "form_diff": h_stats["ppg"] - a_stats["ppg"],
                "form_home_gf": h_stats["gf"], "form_home_ga": h_stats["ga"], "form_away_gf": a_stats["gf"], "form_away_ga": a_stats["ga"],
                "rest_home": 7, "rest_away": 7, "p_home": p_home, "p_draw": p_draw, "p_away": p_away
            }
            
            probs = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
            row["Home Win %"], row["Draw %"], row["Away Win %"] = round(probs[0]*100, 1), round(probs[1]*100, 1), round(probs[2]*100, 1)
            all_predictions.append(row)

    if not all_predictions: return pd.DataFrame({"Info": ["No upcoming matches found."]}), list(unmapped_teams)
        
    df = pd.DataFrame(all_predictions)
    save_predictions(df)
    df = df.sort_values(["league", "Date"]).reset_index(drop=True)
    return df[["league", "Date", "HomeTeam", "AwayTeam", "Home Win %", "Draw %", "Away Win %", "p_home", "p_draw", "p_away"]], list(unmapped_teams)