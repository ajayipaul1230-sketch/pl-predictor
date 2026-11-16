import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd
import joblib
import streamlit as st
from features import FEATURES
from live_predictor import fetch_live_predictions
from database import sync_results, get_all_predictions, save_team_mapping

st.set_page_config(page_title="Multi-League Predictor", page_icon="⚽", layout="wide")
st.title("⚽ European Football Probability Predictor")

@st.cache_data(ttl=3600)
def load_data(): return pd.read_parquet(ROOT / "data/processed/features.parquet")

@st.cache_resource
def load_model(): return joblib.load(ROOT / "models/multi_league_model.joblib")

feat = load_data()
model = load_model()["model"]

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Live Fixtures", "🎮 Matchup Simulator", "📊 Model Analytics", "📈 Prediction Tracker"])

# --- TAB 1: LIVE FIXTURES ---
with tab1:
    st.header("Upcoming Matches Across Europe")
    
    if st.button("Refresh Live Data"):
        st.cache_data.clear()
        
    with st.spinner("Fetching live odds..."):
        live_df, unmapped = fetch_live_predictions(model, feat)
        
    if unmapped:
        st.warning(f"⚠️ Found {len(unmapped)} unknown team(s) in the API. Please map them.")
        all_teams = sorted(feat["HomeTeam"].unique())
        for api_name, league in unmapped:
            with st.expander(f"Map '{api_name}' ({league})"):
                selected_team = st.selectbox(f"Select the correct name for '{api_name}'", ["-- Select --"] + all_teams, key=api_name)
                if st.button(f"Save mapping for {api_name}"):
                    if selected_team != "-- Select --":
                        save_team_mapping(api_name, selected_team)
                        st.success(f"Mapped '{api_name}' to '{selected_team}'! Refreshing...")
                        st.rerun()
        
    if "Error" in live_df.columns or "Info" in live_df.columns:
        st.info(live_df.iloc[0, 0])
    else:
        for league in live_df["league"].unique():
            st.subheader(league)
            st.dataframe(live_df[live_df["league"] == league].drop(columns=["league", "p_home", "p_draw", "p_away"]), use_container_width=True, hide_index=True)

# --- TAB 2: SIMULATOR ---
with tab2:
    st.header("Matchup Simulator")
    all_leagues = sorted(feat["league"].unique())
    selected_league = st.selectbox("Select League", all_leagues)
    league_df = feat[feat["league"] == selected_league]
    all_teams = sorted(league_df["HomeTeam"].unique())
    
    col1, col2 = st.columns(2)
    with col1: home_team = st.selectbox("Select Home Team", all_teams, index=0)
    with col2: away_team = st.selectbox("Select Away Team", all_teams, index=1 if len(all_teams)>1 else 0)

    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        h_stat = league_df[league_df["HomeTeam"] == home_team].iloc[-1]
        a_stat = league_df[league_df["AwayTeam"] == away_team].iloc[-1]
        sim_features = pd.DataFrame([{
            "league": selected_league, "elo_diff": h_stat["elo_home"] - a_stat["elo_away"],
            "form_diff": h_stat["form_home_ppg"] - a_stat["form_away_ppg"], "form_home_gf": h_stat["form_home_gf"], 
            "form_home_ga": h_stat["form_home_ga"], "form_away_gf": a_stat["form_away_gf"], "form_away_ga": a_stat["form_away_ga"],
            "rest_home": 7, "rest_away": 7, "p_home": 0.45, "p_draw": 0.25, "p_away": 0.30
        }])
        probs = model.predict_proba(sim_features[FEATURES])[0]
        st.divider()
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric(f"{home_team} Win", f"{probs[0]*100:.1f}%")
        pcol2.metric("Draw", f"{probs[1]*100:.1f}%")
        pcol3.metric(f"{away_team} Win", f"{probs[2]*100:.1f}%")

# --- TAB 3: MODEL ANALYTICS ---
with tab3:
    st.header("Model Analytics")
    img_path = ROOT / "data/processed/multi_calibration.png"
    if img_path.exists(): st.image(str(img_path), caption="Calibration Curve", width=800)
    metrics_path = ROOT / "data/processed/metrics.csv"
    if metrics_path.exists(): st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)

# --- TAB 4: PREDICTION TRACKER ---
with tab4:
    st.header("Prediction Tracker")
    if st.button("🔄 Sync Match Results"):
        with st.spinner("Checking for results..."):
            updated = sync_results()
            if updated > 0: st.success(f"Updated {updated} matches!")
            else: st.info("No new results found.")
                
    df_history = get_all_predictions()
    if not df_history.empty:
        finished = df_history.dropna(subset=["actual_result"])
        if not finished.empty: st.metric("Overall Average Brier Score", f"{finished['brier_score'].mean():.4f}")
            
        display_df = df_history.copy()
        for col in ["p_home", "p_draw", "p_away"]:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df[["match_date", "league", "home_team", "away_team", "p_home", "p_draw", "p_away", "actual_result", "brier_score"]], use_container_width=True, hide_index=True)