def fetch_live_predictions(model, feat_df):
    api_key = st.secrets.get("ODDS_API_KEY")
    if not api_key:
        return pd.DataFrame({"Error": ["API Key not found in .env file."]}), []

    # Load custom mappings from DB and merge with hardcoded ones
    db_mappings = get_team_mappings()
    master_map = {**TEAM_NAME_MAP, **db_mappings}

    all_predictions = []
    unmapped_teams = set() # Keep track of teams we don't know

    for league_name, sport_key in LEAGUE_API_MAP.items():
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=uk&markets=h2h"
        res = requests.get(url)
        
        if res.status_code != 200: continue
        data = res.json()
        if not data: continue

        for match in data:
            home_raw = match["home_team"]
            away_raw = match["away_team"]
            
            # Use the master map (hardcoded + database)
            home_team = master_map.get(home_raw, home_raw)
            away_team = master_map.get(away_raw, away_raw)

            h_stats = get_latest_team_stats(home_team, feat_df)
            a_stats = get_latest_team_stats(away_team, feat_df)
            
            # If we don't have data, log the unknown raw name and skip
            if not h_stats:
                unmapped_teams.add((home_raw, league_name))
                continue
            if not a_stats:
                unmapped_teams.add((away_raw, league_name))
                continue

            h_odds, d_odds, a_odds = [], [], []
            for bookmaker in match["bookmakers"]:
                for market in bookmaker["markets"]:
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_raw: h_odds.append(outcome["price"])
                            elif outcome["name"] == away_raw: a_odds.append(outcome["price"])
                            elif outcome["name"] == "Draw": d_odds.append(outcome["price"])
            
            if not h_odds or not d_odds or not a_odds: continue

            p_home, p_draw, p_away = get_implied_probs(
                sum(h_odds)/len(h_odds), sum(d_odds)/len(d_odds), sum(a_odds)/len(a_odds)
            )

            row = {
                "league": league_name, 
                "Date": pd.to_datetime(match["commence_time"]).strftime("%Y-%m-%d %H:%M"),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "elo_diff": h_stats["elo"] - a_stats["elo"],
                "form_diff": h_stats["ppg"] - a_stats["ppg"],
                "form_home_gf": h_stats["gf"], "form_home_ga": h_stats["ga"],
                "form_away_gf": a_stats["gf"], "form_away_ga": a_stats["ga"],
                "rest_home": 7, "rest_away": 7, 
                "p_home": p_home, "p_draw": p_draw, "p_away": p_away
            }
            
            probs = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
            row["Home Win %"] = round(probs[0]*100, 1)
            row["Draw %"] = round(probs[1]*100, 1)
            row["Away Win %"] = round(probs[2]*100, 1)
            
            all_predictions.append(row)

    if not all_predictions:
        df = pd.DataFrame({"Info": ["No upcoming matches found."]})
    else:
        df = pd.DataFrame(all_predictions)
        save_predictions(df) # Save to DB
        df = df.sort_values(["league", "Date"]).reset_index(drop=True)
        df = df[["league", "Date", "HomeTeam", "AwayTeam", "Home Win %", "Draw %", "Away Win %", "p_home", "p_draw", "p_away"]]

    return df, list(unmapped_teams)