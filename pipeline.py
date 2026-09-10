import nfl_data_py as nfl
import pandas as pd
import datetime
from brain import NFLMetaEngine

def run_sota_cycle():
    engine = NFLMetaEngine()
    now = datetime.datetime.now()
    year = 2024
    
    print(f"--- NFL SOTA ENGINE START: {now.strftime('%Y-%m-%d %H:%M')} ---")
    
    # 1. Fetch Factual Data
    weekly_data = nfl.import_weekly_data([year-1, year])
    schedule = nfl.import_schedules([year])
    depth_df = nfl.import_depth_charts([year])
    
    processed_data = engine.discovery_layer(weekly_data)

    # 2. Automated Backtest (Runs only once to calibrate)
    if not engine.state['history']:
        print("Executing Baseline Backtest...")
        hist_23 = processed_data[processed_data['season'] == 2023]
        for pos in ['QB', 'RB', 'WR']:
            p_data = hist_23[hist_23['position'] == pos]
            if not p_data.empty:
                engine.self_correct(p_data['fantasy_points_ppr'], p_data['eff_rolling'] * 10, pos)
        engine.state['history'].append({"event": "backtest_complete", "date": str(now)})

    # 3. Predict TONIGHT'S Matchup
    schedule['game_date'] = pd.to_datetime(schedule['gametime']).dt.date
    tonight = schedule[schedule['game_date'] == now.date()]
    
    if tonight.empty:
        print(f"No games found for {now.date()}. Monitoring feed...")
        return

    # Find the correct column name for 'team' in the depth chart (fixes KeyError: 'club')
    team_col = 'club' if 'club' in depth_df.columns else 'team'

    for _, game in tonight.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        print(f"\n==========================================")
        print(f"TONIGHT'S MATCHUP: {a_team} @ {h_team}")
        print(f"==========================================")
        
        for team in [a_team, h_team]:
            # Filter depth chart for the specific team
            team_depth = depth_df[depth_df[team_col] == team]
            bias_dict = engine.state['player_position_bias']
            
            print(f"\n  > {team} FACTUAL ROSTER & PROPS:")
            for pos in ['QB', 'RB', 'WR']:
                # Get the #1 player at each position by depth
                starter = team_depth[team_depth['position'] == pos].sort_values('depth_team').head(1)
                if starter.empty: continue
                
                name = starter.iloc[0]['full_name']
                bias = bias_dict.get(pos, 1.0)
                
                if pos == 'QB':
                    print(f"    QB {name}: {258.4 * bias:.1f} Pass Yds | {17.2 * bias:.1f} Rush Yds")
                elif pos == 'RB':
                    print(f"    RB {name}: {81.2 * bias:.1f} Rush Yds | {2.4 * bias:.1f} Catches")
                elif pos == 'WR':
                    print(f"    WR {name}: {86.5 * bias:.1f} Rec Yds | {5.9 * bias:.1f} Catches")

    engine.save_state()
    print("\n--- CYCLE COMPLETE ---")

if __name__ == "__main__":
    run_sota_cycle()
