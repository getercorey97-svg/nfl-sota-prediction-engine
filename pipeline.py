import nfl_data_py as nfl
import pandas as pd
import datetime
from brain import NFLMetaEngine

def safe_load(func_list, *args, **kwargs):
    for func in func_list:
        if hasattr(nfl, func):
            try:
                data = getattr(nfl, func)(*args, **kwargs)
                if isinstance(data, pd.DataFrame): return data
            except: continue
    return pd.DataFrame()

def find_col(df, options):
    """Finds the existing column from a list of possibilities."""
    for opt in options:
        if opt in df.columns: return opt
    return None

def clean_name(name):
    if not name: return ""
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

def run_realtime_cycle():
    engine = NFLMetaEngine()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    year = now_utc.year
    
    print(f"--- NFL SOTA LIVE CYCLE: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC ---")
    
    sched = safe_load(['import_schedules', 'load_schedules'], [year])
    if sched.empty: return
    
    sched['gametime_dt'] = pd.to_datetime(sched['gametime'], utc=True)
    
    # Analyze games within a 36-hour window
    window_start = now_utc - datetime.timedelta(hours=24)
    window_end = now_utc + datetime.timedelta(hours=12)
    active_games = sched[(sched['gametime_dt'] >= window_start) & (sched['gametime_dt'] <= window_end)]

    if active_games.empty:
        print("No games in the current window. Standby.")
        return

    depth = safe_load(['import_depth_charts', 'load_depth_charts'], [year])
    injuries = safe_load(['import_injuries', 'load_injuries'], [year])
    weekly_stats = safe_load(['import_weekly_data'], [year])

    # Dynamic Column Mapping
    score_col = find_col(sched, ['home_score', 'score_home', 'total_home_score'])
    name_col = find_col(weekly_stats, ['player_display_name', 'player_name', 'player'])
    team_col_depth = find_col(depth, ['club', 'team', 'team_abbr'])
    team_col_weekly = find_col(weekly_stats, ['recent_team', 'team', 'team_abbr'])

    for _, game in active_games.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        # Fix: Robust score checking
        is_final = score_col and not pd.isna(game[score_col])
        status = "FINAL (LEARNING)" if is_final else "UPCOMING/LIVE (PREDICTING)"

        print(f"\n[{status}] {a_team} @ {h_team}")

        for team in [a_team, h_team]:
            params = engine.state['team_params'].get(team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})
            starters = depth[depth[team_col_depth] == team].sort_values('depth_team')

            print(f"  > {team} (Weight: {params['weight']:.2f})")
            for pos in ['QB', 'RB', 'WR']:
                pos_starters = starters[starters['position'] == pos]
                if pos_starters.empty: continue
                
                active_player = None
                for i in range(len(pos_starters)):
                    candidate = pos_starters.iloc[i]
                    c_clean = clean_name(candidate['full_name'])
                    p_injury = injuries[(injuries['team'] == team) & (injuries['full_name'].apply(clean_name) == c_clean)]
                    if p_injury.empty or p_injury.iloc[0]['report_status'] not in ['Out', 'Inactive']:
                        active_player = candidate
                        break

                if active_player is not None:
                    name = active_player['full_name']
                    b = params['bias']['pass' if pos=='QB' else 'rush']
                    pred_yds = (258 if pos=='QB' else 82 if pos=='RB' else 88) * params['weight'] * b
                    
                    if is_final and not weekly_stats.empty:
                        # LEARNING MODE: Pull actuals and update team DNA
                        actual = weekly_stats[(weekly_stats[name_col].apply(clean_name) == clean_name(name)) & (weekly_stats[team_col_weekly] == team)]
                        if not actual.empty:
                            cat = 'pass' if pos=='QB' else 'rush'
                            actual_yds = actual.iloc[0]['passing_yards' if pos=='QB' else 'rushing_yards' if pos=='RB' else 'receiving_yards']
                            print(f"    {pos} {name}: Pred {pred_yds:.1f} | Actual {actual_yds:.1f}")
                            engine.self_correct(team, actual_yds, pred_yds, cat)
                    else:
                        # PREDICTION MODE
                        label = "Pass Yds" if pos=='QB' else "Rush Yds" if pos=='RB' else "Rec Yds"
                        print(f"    {pos} {name}: {pred_yds:.1f} {label}")

    engine.save_state()

if __name__ == "__main__":
    run_realtime_cycle()
