import nfl_data_py as nfl
import pandas as pd
import datetime
import numpy as np
from brain import NFLMetaEngine

def run_autonomous_build():
    # Initialize the Meta-Learning Engine
    engine = NFLMetaEngine()
    now = datetime.datetime.now()
    year = now.year
    
    print(f"--- NFL SOTA ENGINE START: {now} ---")
    
    # 1. Ingest Data (Pulling 2021-2024 to enable Seasonal Segment Analysis)
    print("Step 1: Pulling Historical Data & Segmenting Trends...")
    seasons = [year-3, year-2, year-1, year]
    weekly_data = nfl.import_weekly_data(seasons)
    schedule = nfl.import_schedules([year])
    
    # Ensure columns are numeric for calculation
    cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'epa', 'success']
    for col in cols:
        weekly_data[col] = pd.to_numeric(weekly_data[col], errors='coerce').fillna(0)

    # 2. Process Data via Discovery Layer
    # This identifies correlations and seasonal trends (Early vs Mid vs Late)
    processed_data, features = engine.discovery_layer(weekly_data)

    # 3. Automated Backtesting / Baseline Calibration
    # If the metadata is empty, the engine 'teaches' itself by looking at historical errors
    if not engine.state['history']:
        print("Step 2: No history found. Running Multi-Year Backtest/Baseline Calibration...")
        # Simulate last year's performance to set team weights
        last_year_data = processed_data[processed_data['season'] == (year-1)]
        # Use simple mean-based prediction to set initial error-correcting weights
        engine.self_correct(last_year_data, last_year_data['prev_year_segment_avg'])
        engine.state['history'].append({"event": "baseline_established", "date": str(now)})
        print("Baseline Established.")

    # 4. Predict Upcoming Slate
    print("Step 3: Analyzing Upcoming Matchups...")
    # Convert gametime to date objects for comparison
    schedule['game_date'] = pd.to_datetime(schedule['gametime']).dt.date
    today = now.date()
    
    # Target games playing today or tomorrow
    upcoming = schedule[(schedule['game_date'] >= today) & (schedule['game_date'] <= today + datetime.timedelta(days=1))]
    
    if upcoming.empty:
        print("\n>>> No games scheduled for today/tomorrow. Engine in Monitoring Mode.")
    else:
        print(f"\n--- SOTA PREDICTIONS FOR {today} ---")
        for _, game in upcoming.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            current_week = game['week']
            segment = engine.get_season_segment(current_week)

            # Retrieve Engine Intelligence for these teams
            h_weight = engine.state['team_weights'].get(home_team, 1.0)
            a_weight = engine.state['team_weights'].get(away_team, 1.0)
            
            # Outcome Logic (Elo-based probability modified by Team Weights)
            # Higher weight = Team currently outperforming their historical segment baseline
            win_prob = 0.5 * (h_weight / (a_weight + 0.001))
            win_prob = min(0.99, max(0.01, win_prob)) # Cap at 99%
            
            # Confidence Logic
            conf = engine.calculate_confidence(win_prob, 0.85)

            # Player Projections based on Team Performance Weights
            # (Calculated using league averages adjusted by engine's learned team 'DNA')
            qb_yards = 245 * h_weight if win_prob > 0.5 else 230 * a_weight
            rb_yards = 85 * h_weight
            wr_yards = 75 * h_weight
            catches = round(6 * h_weight)

            print(f"\nGAME: {away_team} @ {home_team} (Season Segment: {segment.upper()})")
            print(f"  PREDICTED WINNER: {home_team if win_prob > 0.5 else away_team}")
            print(f"  WIN PROBABILITY: {win_prob*100:.1f}%")
            print(f"  ENGINE CONFIDENCE: {conf}%")
            print(f"  --- PLAYER PROJECTIONS ---")
            print(f"  QB Passing: {qb_yards:.1f} Yards")
            print(f"  RB Rushing: {rb_yards:.1f} Yards")
            print(f"  WR Receiving: {wr_yards:.1f} Yards | Catches: {catches}")

    # 5. Post-Game Self-Update (Persistence)
    # This saves all learned weights and seasonal trends back to the JSON file
    engine.save_state()
    print("\n--- ENGINE SELF-UPDATE COMPLETE: Metadata Synced ---")

if __name__ == "__main__":
    run_autonomous_build()
