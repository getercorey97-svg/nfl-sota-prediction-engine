import nfl_data_py as nfl
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

def safe_load(func_list, years_list):
    """Safely loads data year-by-year to prevent a single 404 error from crashing the batch."""
    all_data = []
    for y in years_list:
        print(f"  Attempting to load data for {y}...")
        for func in func_list:
            if hasattr(nfl, func):
                try:
                    data = getattr(nfl, func)([y])
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        all_data.append(data)
                        break
                except Exception:
                    print(f"    ⚠️ Warning: {y} data not found (HTTP 404). Skipping...")
                    continue
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def seed_engine_5_year(years=[2021, 2022, 2023, 2024, 2025, 2026]):
    """
    Seeds the engine using historical data.
    Skips missing years gracefully to prevent crashes.
    """
    print(f"🌱 Initiating Engine Seeding...")

    print("\n📥 Fetching schedules (year-by-year to prevent crashes)...")
    sched = safe_load(['import_schedules', 'load_schedules'], years)

    print("\n📥 Fetching weekly data (year-by-year to prevent crashes)...")
    weekly = safe_load(['import_weekly_data', 'load_weekly_data'], years)

    if sched.empty or weekly.empty:
        print("❌ Fatal Error: Could not load sufficient historical data to seed.")
        return

    sched = sched.dropna(subset=['home_score', 'away_score'])
    if 'season' not in sched.columns:
        print("❌ Error: 'season' column missing from schedule data.")
        return
    actual_years = sorted(list(sched['season'].unique()))
    print(f"\n✅ Successfully loaded valid data for seasons: {actual_years}")

    team_data = {}

    # 1. Process Data Year-by-Year with Exponential Decay
    for year in actual_years:
        yr_weight = (year - min(actual_years)) + 1
        yr_sched = sched[sched['season'] == year]
        yr_weekly = weekly[weekly['season'] == year]

        if yr_sched.empty:
            print(f"⚠️ No games found for season {year}. Skipping...")
            continue

        lg_ppg = (yr_sched['home_score'].mean() + yr_sched['away_score'].mean()) / 2
        if 'recent_team' not in yr_weekly.columns or 'passing_yards' not in yr_weekly.columns or 'rushing_yards' not in yr_weekly.columns:
            print(f"⚠️ Required columns missing in weekly data for season {year}. Skipping...")
            continue
        lg_pass = yr_weekly.groupby('recent_team')['passing_yards'].sum().mean()
        lg_rush = yr_weekly.groupby('recent_team')['rushing_yards'].sum().mean()

        for team in yr_sched['home_team'].unique():
            if team not in team_data:
                team_data[team] = {
                    'weighted_ppg_ratio': 0.0, 'weighted_pass_ratio': 0.0, 'weighted_rush_ratio': 0.0,
                    'early_ppg': 0.0, 'late_ppg': 0.0, 'total_weight': 0.0
                }

            t_games = yr_sched[(yr_sched['home_team'] == team) | (yr_sched['away_team'] == team)]
            if t_games.empty: continue

            t_pts = t_games.apply(lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1)
            t_ppg = t_pts.mean()

            t_wk = yr_weekly[yr_weekly['recent_team'] == team]
            t_pass = t_wk['passing_yards'].sum()
            t_rush = t_wk['rushing_yards'].sum()

            early_games = t_games[t_games['week'] <= 13]
            late_games = t_games[t_games['week'] >= 14]

            e_pts = early_games.apply(lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1).mean()
            l_pts = late_games.apply(lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1).mean()

            team_data[team]['weighted_ppg_ratio'] += (t_ppg / max(lg_ppg, 1)) * yr_weight
            team_data[team]['weighted_pass_ratio'] += (t_pass / max(lg_pass, 1)) * yr_weight
            team_data[team]['weighted_rush_ratio'] += (t_rush / max(lg_rush, 1)) * yr_weight
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
        except:
            pass

    if "team_params" not in state:
        state["team_params"] = {}

    print("\n📊 Seeded Intelligence & Motivation Indices:")
    for t, d in team_data.items():
        tw = d['total_weight']
        if tw == 0:
            continue

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

    print("\n✅ Seeding & Motivation Mapping Complete. The engine brain is primed.")

if __name__ == "__main__":
    seed_engine_5_year()
