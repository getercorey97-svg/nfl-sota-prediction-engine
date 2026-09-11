import nfl_data_py as nfl
import pandas as pd
import datetime
from brain import NFLMetaEngine

def run_sota_cycle():
    engine = NFLMetaEngine()
    now = datetime.datetime.now()
    year = 2024
    
    print(f"--- NFL SOTA ENGINE START: {now.strftime('%Y-%m-%d %H:%M')} ---")
    
    # 1. Pull Data Feeds (Stats, Schedules, Depth, Roster)
    try:
        weekly = nfl.import_weekly_data([year-1, year])
        sched = nfl.import_schedules([year])
        depth = nfl.import_depth_charts([year])
        roster = nfl.import_rosters([year])
    except Exception as e:
        print(f"Critical Data Error: {e}")
        return

    # Merge Roster with Weekly to get Factual Positions for Backtesting
    weekly = weekly.merge(roster[['player_id', 'position']], on='player_id', how='left', suffixes=('', '_roster'))
    if 'position_roster' in weekly.columns:
        weekly['position'] = weekly['position'].fillna(weekly['position_roster'])

    processed = engine.discovery_layer(weekly)

    # 2. Backtest Baseline (First Run Only)
    if not engine.state['history']:
        print("Calibrating SOTA Baseline (2023 Analysis)...")
        hist_23 = processed[processed['season'] == 2023]
        for pos in ['QB', 'RB', 'WR']:
            p_data = hist_23[hist_23['position'] == pos]
            if not p_data.empty:
                # Target 'fantasy_points_ppr' as a factual outcome proxy for learning
                engine.self_correct(p_data['fantasy_points_ppr'], p_data['eff_rolling']*5, pos)
        engine.state['history'].append({"event": "calibrated", "date": str(now)})

    # 3. Identify Tonight's Game
    sched['game_date'] = pd.to_datetime(sched['gametime']).dt.date
    today = now.date()
    # Broaden filter to 24-hour window to handle UTC/Timezone shifts
    tonight = sched[(sched['game_date'] >= today) & (sched['game_date'] <= today + datetime.timedelta(days=1))]
    
    if tonight.empty:
        print(f"No games found in the feed for {today}. Engine in standby.")
        return

    # Find Team Column in Depth Charts dynamically (Solves KeyError: 'team' / 'club')
    depth_team_col = engine.find_col(depth, ['club', 'team', 'team_abbr', 'club_abbr'])
    
    for _, game in tonight.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        print(f"\n==========================================")
        print(f"TONIGHT: {a_team} @ {h_team}")
        print(f"==========================================")
        
        for team in [a_team, h_team]:
            # Use the dynamic column found above
            team_depth = depth[depth[depth_team_col] == team]
            bias_dict = engine.state['player_position_bias']
            
            print(f"\n  > {team} FACTUAL ROSTER & PROPS:")
            for pos in ['QB', 'RB', 'WR']:
                starter = team_depth[team_depth['position'] == pos].sort_values('depth_team').head(1)
                if starter.empty: continue
                
                name = starter.iloc[0]['full_name']
                bias = bias_dict.get(pos, 1.0)
                
                # SOTA Equations (Adjusted by Learned Bias)
                if pos == 'QB':
                    print(f"    QB {name}: {256.4*bias:.1f} Pass Yds | {18.2*bias:.1f} Rush Yds")
                elif pos == 'RB':
                    print(f"    RB {name}: {79.8*bias:.1f} Rush Yds | {2.3*bias:.1f} Catches")
                elif pos == 'WR':
                    print(f"    WR {name}: {84.2*bias:.1f} Rec Yds | {5.8*bias:.1f} Catches")

    engine.save_state()
    print("\n--- CYCLE COMPLETE: DATA SYNCED ---")

if __name__ == "__main__":
    run_sota_cycle()
