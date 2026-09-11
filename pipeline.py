import nfl_data_py as nfl
import pandas as pd
import datetime
from brain import NFLMetaEngine

def safe_load(func_list, *args, **kwargs):
    for func in func_list:
        if hasattr(nfl, func): return getattr(nfl, func)(*args, **kwargs)
    return pd.DataFrame()

def run_realtime_cycle():
    engine = NFLMetaEngine()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    year = now_utc.year
    
    print(f"--- NFL SOTA LIVE CYCLE: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC ---")
    
    # 1. Fetch Schedule and Identify 'Active Windows'
    sched = safe_load(['import_schedules', 'load_schedules'], [year])
    sched['gametime_dt'] = pd.to_datetime(sched['gametime'])
    
    # Identify games starting in the next 120 minutes or recently finished
    active_games = sched[
        (sched['gametime_dt'] >= now_utc - datetime.timedelta(hours=4)) & 
        (sched['gametime_dt'] <= now_utc + datetime.timedelta(minutes=120))
    ]

    if active_games.empty:
        print("No games in the immediate 2hr window. Standby mode.")
        return

    # 2. Fetch Latest Rosters, Depth Charts, and Injuries
    depth = safe_load(['import_depth_charts', 'load_depth_charts'], [year])
    injuries = safe_load(['import_injuries', 'load_injuries'], [year])
    # Filter injuries for current week
    current_week = active_games.iloc[0]['week']
    injuries = injuries[injuries['week'] == current_week]

    for _, game in active_games.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        kickoff = game['gametime_dt']
        
        # Check if we are in the 'Pre-Game Roster Verification' window (90 mins before)
        is_pregame = now_utc < kickoff
        status_label = "PRE-GAME PREDICTION (LIVE ROSTER CHECK)" if is_pregame else "POST-GAME LEARNING"

        print(f"\n[{status_label}] {a_team} @ {h_team}")
        
        for team in [a_team, h_team]:
            team_params = engine.state['team_params'].get(team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})
            
            # Find Depth Chart Starters
            team_col = 'club' if 'club' in depth.columns else 'team'
            starters = depth[depth[team_col] == team].sort_values('depth_team')

            print(f"  > {team} (Team Weight: {team_params['weight']:.2f})")
            for pos in ['QB', 'RB', 'WR']:
                player = starters[starters['position'] == pos].head(1)
                if player.empty: continue
                
                name = player.iloc[0]['full_name']
                # CROSS-CHECK: Is this player on the INACTIVE/OUT list?
                p_injury = injuries[(injuries['team'] == team) & (injuries['full_name'] == name)]
                
                if not p_injury.empty:
                    status = p_injury.iloc[0]['report_status']
                    if status in ['Out', 'Inactive', 'Doubtful']:
                        print(f"    [SCRATCHED] {pos} {name} is {status}. Checking next in depth chart...")
                        player = starters[starters['position'] == pos].iloc[1:2] # Take Backup
                        if player.empty: continue
                        name = player.iloc[0]['full_name']
                        print(f"    [NEW STARTER] {pos} {name} moving to QB1/RB1/WR1.")

                # Calculate factual props using team's custom bias
                p_bias = team_params['bias']['pass' if pos=='QB' else 'rush']
                if pos == 'QB':
                    print(f"    QB {name}: {258 * team_params['weight'] * p_bias:.1f} Pass Yds")
                elif pos == 'RB':
                    print(f"    RB {name}: {82 * team_params['weight'] * p_bias:.1f} Rush Yds")
                elif pos == 'WR':
                    print(f"    WR {name}: {88 * team_params['weight'] * p_bias:.1f} Rec Yds")

    engine.save_state()

if __name__ == "__main__":
    run_realtime_cycle()
