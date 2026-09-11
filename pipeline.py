import nfl_data_py as nfl
import pandas as pd
import datetime
from brain import NFLMetaEngine

def safe_load(func_list, *args, **kwargs):
    for func in func_list:
        if hasattr(nfl, func): return getattr(nfl, func)(*args, **kwargs)
    return pd.DataFrame()

def clean_name(name):
    """Normalizes names to ensure accurate injury/depth-chart matching."""
    if not name: return ""
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

def run_realtime_cycle():
    engine = NFLMetaEngine()
    # Fix: Ensure both datetimes are UTC-aware to prevent TypeError crashes
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    year = now_utc.year
    
    print(f"--- NFL SOTA LIVE CYCLE: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC ---")
    
    sched = safe_load(['import_schedules', 'load_schedules'], [year])
    # Fix: Force UTC awareness on schedule data
    sched['gametime_dt'] = pd.to_datetime(sched['gametime'], utc=True)
    
    active_games = sched[
        (sched['gametime_dt'] >= now_utc - datetime.timedelta(hours=6)) & 
        (sched['gametime_dt'] <= now_utc + datetime.timedelta(minutes=150))
    ]

    if active_games.empty:
        print("No games in the immediate 2.5hr window. Standby mode.")
        return

    depth = safe_load(['import_depth_charts', 'load_depth_charts'], [year])
    injuries = safe_load(['import_injuries', 'load_injuries'], [year])
    
    # Normalize injury names for robust matching
    if not injuries.empty:
        injuries['clean_name'] = injuries['full_name'].apply(clean_name)

    for _, game in active_games.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        kickoff = game['gametime_dt']
        
        is_pregame = now_utc < kickoff
        print(f"\n[{'PRE-GAME' if is_pregame else 'LIVE/POST'}] {a_team} @ {h_team}")
        
        for team in [a_team, h_team]:
            params = engine.state['team_params'].get(team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})
            team_col = 'club' if 'club' in depth.columns else 'team'
            starters = depth[depth[team_col] == team].sort_values('depth_team')

            print(f"  > {team} (Intelligence Weight: {params['weight']:.2f})")
            for pos in ['QB', 'RB', 'WR']:
                # Filter depth for position
                pos_starters = starters[starters['position'] == pos]
                if pos_starters.empty: continue
                
                # Logic to find the first HEALTHY starter
                active_player = None
                for i in range(len(pos_starters)):
                    candidate = pos_starters.iloc[i]
                    c_name = candidate['full_name']
                    c_clean = clean_name(c_name)
                    
                    # Check injury status
                    p_status = injuries[(injuries['team'] == team) & (injuries['clean_name'] == c_clean)]
                    
                    status_text = "Healthy"
                    is_out = False
                    if not p_status.empty:
                        status_text = p_status.iloc[0]['report_status']
                        if status_text in ['Out', 'Inactive', 'Doubtful']:
                            is_out = True
                    
                    if not is_out:
                        active_player = candidate
                        break
                    else:
                        print(f"    [SCRATCHED] {pos} {c_name} is {status_text}. Moving to next on depth chart...")

                if active_player is not None:
                    name = active_player['full_name']
                    b = params['bias']['pass' if pos=='QB' else 'rush']
                    if pos == 'QB':
                        print(f"    QB {name}: {258 * params['weight'] * b:.1f} Pass Yds")
                    elif pos == 'RB':
                        print(f"    RB {name}: {82 * params['weight'] * b:.1f} Rush Yds")
                    elif pos == 'WR':
                        print(f"    WR {name}: {88 * params['weight'] * b:.1f} Rec Yds")

    engine.save_state()

if __name__ == "__main__":
    run_realtime_cycle()
