import nfl_data_py as nfl
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

def find_col(df, options):
    """Dynamically finds the correct column name from a list of possibilities."""
    for opt in options:
        if opt in df.columns:
            return opt
    return None

def clean_name(name):
    """Normalizes names to ensure accurate cross-referencing."""
    if not name:
        return ""
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

def seed_engine_5_year(years=[2021, 2022, 2023, 2024, 2025]):
    """
    Seeds the engine using 5 years of weighted historical data.
    Calculates Base Weights, Playcalling Biases, and a Motivation/Seasonality Index.
    """
    print(f"🌱 Initiating 5-Year Engine Seeding for seasons: {years}...")
    
    print("📥 Fetching schedules and weekly stats (this will take a moment)...")
    sched = nfl.import_schedules(years)
    weekly = nfl.import_weekly_data(years)
    
    # Standardize columns to lowercase
    sched.columns = sched.columns.str.lower()
    weekly.columns = weekly.columns.str.lower()
    
    # Drop games without scores
    sched = sched.dropna(subset=['home_score', 'away_score'])
    
    # Find the team column in weekly data (could be 'recent_team', 'team', etc.)
    team_col_weekly = find_col(weekly, ['recent_team', 'team', 'team_abbr'])
    if not team_col_weekly:
        raise ValueError("Could not find team column in weekly data")
    
    team_data = {}
    
    # 1. Process Data Year-by-Year with Exponential Decay
    for year in years:
        yr_weight = (year - min(years)) + 1 
        yr_sched = sched[sched['season'] == year]
        yr_weekly = weekly[weekly['season'] == year]
        
        lg_ppg = (yr_sched['home_score'].mean() + yr_sched['away_score'].mean()) / 2
        lg_pass = yr_weekly.groupby(team_col_weekly)['passing_yards'].sum().mean()
        lg_rush = yr_weekly.groupby(team_col_weekly)['rushing_yards'].sum().mean()
        
        for team in yr_sched['home_team'].unique():
            if team not in team_data:
                team_data[team] = {
                    'weighted_ppg_ratio': 0, 'weighted_pass_ratio': 0, 'weighted_rush_ratio': 0,
                    'early_ppg': 0, 'late_ppg': 0, 'total_weight': 0
                }
            
            t_games = yr_sched[(yr_sched['home_team'] == team) | (yr_sched['away_team'] == team)]
            if t_games.empty: continue
            
            t_pts = t_games.apply(lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1)
            t_ppg = t_pts.mean()
            
            t_wk = yr_weekly[yr_weekly[team_col_weekly] == team]
            t_pass = t_wk['passing_yards'].sum()
            t_rush = t_wk['rushing_yards'].sum()
            
            # Early vs Late Season splits for Motivation
            early_games = t_games[t_games['week'] <= 13]
            late_games = t_games[t_games['week'] >= 14]
            
            e_pts = early_games.apply(lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1).mean()
            l_pts = late_games.apply(lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1).mean()
            
            team_data[team]['weighted_ppg_ratio'] += (t_ppg / lg_ppg) * yr_weight
            team_data[team]['weighted_pass_ratio'] += (t_pass / max(lg_pass, 1)) * yr_weight
            team_data[team]['weighted_rush_ratio'] += (t_rush / max(lg_rush, 1)) * yr_weight
            
            # Handle NaN for early/late splits (if no games in that split)
            team_data[team]['early_ppg'] += (np.nan_to_num(e_pts) * yr_weight)
            team_data[team]['late_ppg'] += (np.nan_to_num(l_pts) * yr_weight)
            team_data[team]['total_weight'] += yr_weight

    # 2. Finalize Engine Brain State
    meta_path = 'engine_metadata.json'
    state = {"team_params": {}, "model_params": {
        "learning_rate": 0.05, "dixon_coles_rho": 0.13, "qb_wr_correlation": 0.45, "fatigue_decay": 0.02
    }}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                state = json.load(f)
        except: pass

    if "team_params" not in state: state["team_params"] = {}

    print("\n📊 5-Year Seeded Intelligence & Motivation Indices:")
    for t, d in team_data.items():
        tw = d['total_weight']
        if tw == 0: continue
        
        weight = max(0.75, min(1.25, d['weighted_ppg_ratio'] / tw))
        pass_bias = max(0.80, min(1.20, d['weighted_pass_ratio'] / tw))
        rush_bias = max(0.80, min(1.20, d['weighted_rush_ratio'] / tw))
        
        avg_early = d['early_ppg'] / tw
        avg_late = d['late_ppg'] / tw
        motivation_index = max(0.85, min(1.15, avg_late / max(avg_early, 10.0)))
        
        if t not in state['team_params']:
            state['team_params'][t] = {"lr": 0.05, "bias": {}}
            
        state['team_params'][t]['weight'] = round(weight, 3)
        state['team_params'][t]['bias']['pass'] = round(pass_bias, 3)
        state['team_params'][t]['bias']['rush'] = round(rush_bias, 3)
        state['team_params'][t]['motivation_index'] = round(motivation_index, 3)
        
        trend = "🔥 Surges Late" if motivation_index > 1.02 else ("🧊 Fades Late" if motivation_index < 0.98 else "⚖️ Consistent")
        print(f"  {t.ljust(4)}: Wgt {weight:.2f} | P-Bias {pass_bias:.2f} | R-Bias {rush_bias:.2f} | Motivation: {motivation_index:.2f} ({trend})")

    with open(meta_path, 'w') as f:
        json.dump(state, f, indent=4)
        
    print("\n✅ 5-Year Seeding & Motivation Mapping Complete. The engine brain is primed.")

if __name__ == "__main__":
    seed_engine_5_year()