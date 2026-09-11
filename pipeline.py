import nfl_data_py as nfl
import pandas as pd
import datetime
from brain import NFLMetaEngine

def safe_load(func_list, *args, **kwargs):
    for func in func_list:
        if hasattr(nfl, func): return getattr(nfl, func)(*args, **kwargs)
    return pd.DataFrame()

def clean_name(name):
    if not name: return ""
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

def run_realtime_cycle():
    engine = NFLMetaEngine()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    year = now_utc.year
    
    print(f"--- NFL SOTA LIVE CYCLE: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC ---")
    
    # 1. Fetch Schedule
    sched = safe_load(['import_schedules', 'load_schedules'], [year])
    sched['gametime_dt'] = pd.to_datetime(sched['gametime'], utc=True)
    
    # BROADENED WINDOW: Catch everything happening today (UTC) 
    # plus anything that started in the last 12 hours (Live Games)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_start = today_start + datetime.timedelta(days=1)
    
    active_games = sched[
        (sched['gametime_dt'] >= today_start - datetime.timedelta(hours=6)) & 
        (sched['gametime_dt'] < tomorrow_start)
    ]

    if active_games.empty:
        print(f"No games scheduled for today ({now_utc.date()}). Engine in standby.")
        return

    # 2. Fetch Latest Roster & Injury Data
    depth = safe_load(['import_depth_charts', 'load_depth_charts'], [year])
    injuries = safe_load(['import_injuries', 'load_injuries'], [year])
    
    if not injuries.empty:
        injuries['clean_name'] = injuries['full_name'].apply(clean_name)

    for _, game in active_games.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        kickoff = game['gametime_dt']
        
        # Determine Game State
        if now_utc < kickoff:
            status = "UPCOMING (PRE-GAME PREDICTION)"
        elif pd.isna(game['score_home']):
            status = "LIVE (IN-PROGRESS TRACKING)"
        else:
            status = "FINAL (POST-GAME LEARNING)"

        print(f"\n[{status}] {a_team} @ {h_team}")
        print(f"Kickoff: {kickoff.strftime('%H:%M')} UTC")
        
        for team in [a_team, h_team]:
            params = engine.state['team_params'].get(team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})
            team_col = 'club' if 'club' in depth.columns else 'team'
            starters = depth[depth[team_col] == team].sort_values('depth_team')

            print(f"  > {team} (Efficiency Weight: {params['weight']:.2f})")
            for pos in ['QB', 'RB', 'WR']:
                pos_starters = starters[starters['position'] == pos]
                if pos_starters.empty: continue
                
                active_player = None
                for i in range(len(pos_starters)):
                    candidate = pos_starters.iloc[i]
                    c_name = candidate['full_name']
                    c_clean = clean_name(c_name)
                    
                    # Verify Health
                    p_status = injuries[(injuries['team'] == team) & (injuries['clean_name'] == c_clean)]
                    is_out = not p_status.empty and p_status.iloc[0]['report_status'] in ['Out', 'Inactive', 'Doubtful']
                    
                    if not is_out:
                        active_player = candidate
                        break
                    else:
                        print(f"    [OUT] {pos} {c_name} detected. Skipping...")

                if active_player is not None:
                    name = active_player['full_name']
                    b = params['bias']['pass' if pos=='QB' else 'rush']
                    # SOTA Calculation Logic
                    if pos == 'QB':
                        print(f"    QB {name}: {258 * params['weight'] * b:.1f} Pass Yds")
                    elif pos == 'RB':
                        print(f"    RB {name}: {82 * params['weight'] * b:.1f} Rush Yds")
                    elif pos == 'WR':
                        print(f"    WR {name}: {88 * params['weight'] * b:.1f} Rec Yds")

    engine.save_state()

if __name__ == "__main__":
    run_realtime_cycle()
