import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd
import joblib
import streamlit as st
from features import FEATURES
from live_predictor import fetch_live_predictions

st.set_page_config(page_title="Multi-League Predictor", page_icon="⚽", layout="wide")
st.title("⚽ European Football Probability Predictor")

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_parquet(ROOT / "data/processed/features.parquet")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/multi_league_model.joblib")

feat = load_data()
bundle = load_model()
model = bundle["model"]

tab1, tab2, tab3 = st.tabs(["🔥 Live Fixtures", "🎮 Matchup Simulator", "📊 Model Analytics"])

# --- TAB 1: LIVE FIXTURES ---
with tab1:
    st.header("Upcoming Matches Across Europe")
    st.write("These are real upcoming fixtures with live odds pulled from The Odds API.")
    
    if st.button("Refresh Live Data"):
        st.cache_data.clear()
        
    with st.spinner("Fetching live odds across all leagues and running model..."):
        live_df = fetch_live_predictions(model, feat)
        
    if "Error" in live_df.columns or "Info" in live_df.columns:
        st.warning(live_df.iloc[0, 0])
    else:
        # Display results grouped by league
        for league in live_df["league"].unique():
            st.subheader(league)
            league_data = live_df[live_df["league"] == league].drop(columns=["league"])
            st.dataframe(league_data, use_container_width=True, hide_index=True)

# --- TAB 2: SIMULATOR --- (Unchanged from previous step)
with tab2:
    st.header("Matchup Simulator")
    
    all_leagues = sorted(feat["league"].unique())
    selected_league = st.selectbox("Select League", all_leagues)
    
    league_df = feat[feat["league"] == selected_league]
    all_teams = sorted(league_df["HomeTeam"].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Select Home Team", all_teams, index=0)
    with col2:
        away_team = st.selectbox("Select Away Team", all_teams, index=1 if len(all_teams)>1 else 0)

    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        home_stats = league_df[league_df["HomeTeam"] == home_team].iloc[-1]
        away_stats = league_df[league_df["AwayTeam"] == away_team].iloc[-1]

        sim_features = pd.DataFrame([{
            "league": selected_league,
            "elo_diff": home_stats["elo_home"] - away_stats["elo_away"],
            "form_diff": home_stats["form_home_ppg"] - away_stats["form_away_ppg"],
            "form_home_gf": home_stats["form_home_gf"], "form_home_ga": home_stats["form_home_ga"],
            "form_away_gf": away_stats["form_away_gf"], "form_away_ga": away_stats["form_away_ga"],
            "rest_home": 7, "rest_away": 7,
            "p_home": 0.45, "p_draw": 0.25, "p_away": 0.30
        }])

        probs = model.predict_proba(sim_features[FEATURES])[0]

        st.divider()
        st.subheader(f"{home_team} vs {away_team} ({selected_league})")

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric(f"{home_team} Win", f"{probs[0]*100:.1f}%")
        pcol2.metric("Draw", f"{probs[1]*100:.1f}%")
        pcol3.metric(f"{away_team} Win", f"{probs[2]*100:.1f}%")

# --- TAB 3: MODEL ANALYTICS --- (Unchanged)
with tab3:
    st.header("Model Analytics")
    st.write("This model calculates Elo and Form across multiple European leagues.")
    st.write("Because it uses a `league` categorical feature, it learns that a goal in the Bundesliga has a different baseline probability than a goal in Eliteserien.")
    # Optional: You can add the calibration plot here later