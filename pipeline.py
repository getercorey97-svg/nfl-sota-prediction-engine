import nfl_data_py as nfl
import pandas as pd
import datetime
import numpy as np
from brain import NFLMetaEngine

def run_sota_cycle():
    engine = NFLMetaEngine()
    now = datetime.datetime.now()
    year = 2024 # Current Season
    
    print(f"--- NFL SOTA ENGINE CYCLE: {now} ---")
    
    # 1. Pull 3 years of data for backtesting and segmenting
    weekly_data = nfl.import_weekly_data([year-3, year-2, year-1, year])
    schedule = nfl.import_schedules([year])
    
    # 2. Process discovery layer
    processed_data, _ = engine.discovery_layer(weekly_data)

    # 3. Automatic Backtest (only runs on the very first execution)
    if not engine.state['history']:
        print("Performing Initial Backtest & Baseline Calibration...")
        last_year = processed_data[processed_data['season'] == (year-1)]
        engine.self_correct(last_year, last_year['prev_year_segment_avg'])
        engine.state['history'].append({"event": "baseline", "date": str(now)})

    # 4. Filter for Tonight's Games
    schedule['game_date'] = pd.to_datetime(schedule['gametime']).dt.date
    today = now.date()
    upcoming = schedule[schedule['game_date'] == today]
    
    if upcoming.empty:
        print("No games today. System learning from past results...")
        # Check for games that just finished to update intelligence
        recent_actuals = processed_data[processed_data['season'] == year].tail(50)
        engine.self_correct(recent_actuals, recent_actuals['prev_year_segment_avg'])
    else:
        print(f"\n--- SOTA PREDICTIONS FOR TONIGHT ---")
        for _, game in upcoming.iterrows():
            h_team, a_team = game['home_team'], game['away_team']
            h_w = engine.state['team_weights'].get(h_team, 1.0)
            a_w = engine.state['team_weights'].get(a_team, 1.0)
            
            win_prob = 0.5 * (h_w / a_w)
            conf = engine.calculate_confidence(win_prob)

            print(f"\nMATCHUP: {a_team} @ {h_team}")
            print(f"  WINNER: {h_team if win_prob > 0.5 else a_team} ({win_prob*100:.1f}%)")
            print(f"  CONFIDENCE: {conf}%")
            print(f"  QB Passing: {252 * h_w:.1f} yds | RB Rushing: {82 * h_w:.1f} yds")
            print(f"  WR Receiving: {88 * h_w:.1f} yds | Catches: {round(6.2 * h_w)}")

    engine.save_state()

if __name__ == "__main__":
    run_sota_cycle()
