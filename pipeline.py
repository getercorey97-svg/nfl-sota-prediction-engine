import nfl_data_py as nfl
import pandas as pd
import datetime
from brain import NFLMetaEngine

def safe_load(func_list, *args, **kwargs):
    """Safely loads data from multiple possible function names."""
    for func in func_list:
        if hasattr(nfl, func):
            try:
                data = getattr(nfl, func)(*args, **kwargs)
                if isinstance(data, pd.DataFrame) and not data.empty: return data
            except: continue
    return pd.DataFrame()

def find_col(df, options):
    """Dynamically finds the correct column name from a list of possibilities."""
    for opt in options:
        if opt in df.columns: return opt
    return None

def clean_name(name):
    """Normalizes names to ensure accurate cross-referencing."""
    if not name: return ""
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

def run_realtime_cycle():
    engine = NFLMetaEngine()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    year = now_utc.year
    
    print(f"--- NFL SOTA ENGINE START: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC ---")
    
    # 1. Pull Core Factual Data
    sched = safe_load(['import_schedules', 'load_schedules'], [year])
    if sched.empty:
        print("Schedule data unavailable. Check year/API status.")
        return
        
    sched['gametime_dt'] = pd.to_datetime(sched['gametime'], utc=True)
    
    # Look for games in a 24-hour window
    window_start = now_utc - datetime.timedelta(hours=18)
    window_end = now_utc + datetime.timedelta(hours=6)
    active_games = sched[(sched['gametime_dt'] >= window_start) & (sched['gametime_dt'] <= window_end)]

    if active_games.empty:
        print("No games currently scheduled for today.")
        return

    # Fetch Supporting Data
    depth = safe_load(['import_depth_charts', 'load_depth_charts'], [year])
    injuries = safe_load(['import_injuries', 'load_injuries'], [year])
    weekly_stats = safe_load(['import_weekly_data', 'load_weekly_data'], [year-1, year])

    # Dynamic Column Mapping (Fixes KeyError crashes)
    score_col = find_col(sched, ['home_score', 'score_home', 'total_home_score'])
    name_col = find_col(weekly_stats, ['player_display_name', 'player_name', 'full_name'])
    team_col_depth = find_col(depth, ['club', 'team', 'team_abbr'])
    depth_rank_col = find_col(depth, ['depth_team', 'depth', 'depth_order', 'rank']) # Fix for depth_team error
    team_col_weekly = find_col(weekly_stats, ['recent_team', 'team', 'team_abbr'])

    for _, game in active_games.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        is_final = score_col and not pd.isna(game[score_col])
        status = "FINAL (LEARNING)" if is_final else "UPCOMING/LIVE (PREDICTING)"

        print(f"\n[{status}] {a_team} @ {h_team}")

        for team in [a_team, h_team]:
            params = engine.state['team_params'].get(team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})
            
            # Identify Starters with fallback sorting
            if team_col_depth and not depth.empty:
                team_roster = depth[depth[team_col_depth] == team]
                if depth_rank_col:
                    starters = team_roster.sort_values(depth_rank_col)
                else:
                    starters = team_roster
            else:
                starters = pd.DataFrame()

            print(f"  > {team} (Intelligence Weight: {params['weight']:.2f})")
            
            for pos in ['QB', 'RB', 'WR']:
                if starters.empty: 
                    print(f"    {pos} Data Missing for this team.")
                    continue
                    
                pos_starters = starters[starters['position'] == pos]
                active_player = None
                
                # Check health for the top of the depth chart
                for i in range(len(pos_starters)):
                    candidate = pos_starters.iloc[i]
                    name_to_check = candidate['full_name']
                    c_clean = clean_name(name_to_check)
                    
                    # Roster/Injury Verification
                    p_injury = injuries[(injuries['team'] == team) & (injuries['full_name'].apply(clean_name) == c_clean)]
                    if p_injury.empty or p_injury.iloc[0]['report_status'] not in ['Out', 'Inactive']:
                        active_player = candidate
                        break
                    else:
                        print(f"    [OUT] {pos} {name_to_check} is scratched.")

                if active_player is not None:
                    p_name = active_player['full_name']
                    cat = 'pass' if pos=='QB' else 'rush'
                    b = params['bias'][cat]
                    pred_yds = (258 if pos=='QB' else 82 if pos=='RB' else 88) * params['weight'] * b
                    
                    if is_final and not weekly_stats.empty:
                        # ACTUALS LEARNING MODE
                        actual = weekly_stats[(weekly_stats[name_col].apply(clean_name) == clean_name(p_name)) & (weekly_stats[team_col_weekly] == team)]
                        if not actual.empty:
                            actual_yds = actual.iloc[0]['passing_yards' if pos=='QB' else 'rushing_yards' if pos=='RB' else 'receiving_yards']
                            print(f"    {pos} {p_name}: Predicted {pred_yds:.1f} | Actual {actual_yds:.1f}")
                            engine.self_correct(team, actual_yds, pred_yds, cat)
                    else:
                        # PREDICTION MODE
                        label = "Pass Yds" if pos=='QB' else "Rush Yds" if pos=='RB' else "Rec Yds"
                        print(f"    {pos} {p_name}: {pred_yds:.1f} {label}")

    engine.save_state()
    print("\n--- CYCLE COMPLETE: ALL INTELLIGENCE SYNCED ---")

if __name__ == "__main__":
    run_realtime_cycle()
