import nfl_data_py as nfl
import pandas as pd
import numpy as np
import datetime
import os
import json
from brain import NFLMetaEngine
from scipy.stats import poisson, norm
import warnings
warnings.filterwarnings('ignore')

def safe_load(func_list, years_list):
    """Safely loads data year-by-year to bypass HTTP 404s on missing seasons."""
    all_data = []
    for y in years_list:
        for func in func_list:
            if hasattr(nfl, func):
                try:
                    data = getattr(nfl, func)([y])
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        all_data.append(data)
                        break
                except Exception:
                    continue
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def find_col(df, options):
    for opt in options:
        if opt in df.columns:
            return opt
    return None

def clean_name(name):
    if not name:
        return ""
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

def dixon_coles_adjustment(home_goals, away_goals, rho, mu_x, mu_y):
    if home_goals == 0 and away_goals == 0:
        return 1 - rho * mu_x * mu_y
    elif home_goals == 0 and away_goals == 1:
        return 1 + rho * mu_x
    elif home_goals == 1 and away_goals == 0:
        return 1 + rho * mu_y
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0

class BacktestCalibrator:
    def __init__(self, years=[2023, 2024, 2025, 2026]):
        self.years = years
        self.engine = NFLMetaEngine()

        self.metrics = {
            "games_played": 0,
            "winner_correct": 0,
            "spread_covers": 0,
            "total_over_under_correct": 0,
            "team_over_under_correct": 0,
            "qb_errors": [],
            "rb_errors": [],
            "wr_errors": []
        }

    def apply_drift_reduction(self, team):
        if team not in self.engine.state['team_params']:
            return
        params = self.engine.state['team_params'][team]

        # Bring weights slightly back toward baseline 1.0
        params['weight'] += (1.0 - params['weight']) * 0.02
        params['bias']['pass'] += (1.0 - params['bias'].get('pass', 1.0)) * 0.02
        params['bias']['rush'] += (1.0 - params['bias'].get('rush', 1.0)) * 0.02

    def run_calibration(self):
        print(f"🚀 INITIALIZING BACKTEST CALIBRATION FOR SEASONS: {self.years}")

        print("📥 Loading historical data (bypassing 404 missing years)...")
        sched = safe_load(['import_schedules', 'load_schedules'], self.years)
        weekly = safe_load(['import_weekly_data', 'load_weekly_data'], self.years)

        if sched.empty or weekly.empty:
            print("❌ Failed to load schedule or weekly data. Aborting.")
            return

        sched.columns = sched.columns.str.lower()
        weekly.columns = weekly.columns.str.lower()

        if 'gametime' not in sched.columns:
            print("❌ 'gametime' column missing from schedule data.")
            return
        sched['gametime_dt'] = pd.to_datetime(sched['gametime'], utc=True)
        sched = sched.sort_values('gametime_dt')

        score_col = find_col(sched, ['home_score', 'score_home'])
        if score_col is None:
            print("❌ No valid score column found in schedule data.")
            return
        sched = sched.dropna(subset=[score_col, 'away_score'])

        name_col = find_col(weekly, ['player_display_name', 'player_name'])
        team_col_weekly = find_col(weekly, ['recent_team', 'team', 'team_abbr'])

        for index, game in sched.iterrows():
            h_team = game['home_team']
            a_team = game['away_team']

            actual_home_score = game[score_col]
            actual_away_score = game['away_score']
            vegas_spread = game.get('spread_line', 0)
            vegas_total = game.get('total_line', 45.0)

            h_params = self.engine.state['team_params'].get(h_team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})
            a_params = self.engine.state['team_params'].get(a_team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})

            h_mot = h_params.get('motivation_index', 1.0) if game.get('week', 1) >= 14 else 1.0
            a_mot = a_params.get('motivation_index', 1.0) if game.get('week', 1) >= 14 else 1.0

            mu_home = 22.5 * h_params['weight'] * 1.05 * h_mot
            mu_away = 22.5 * a_params['weight'] * 0.95 * a_mot

            pred_h_qb = 258 * h_params['weight'] * h_params['bias'].get('pass', 1.0) * h_mot
            pred_a_qb = 258 * a_params['weight'] * a_params['bias'].get('pass', 1.0) * a_mot
            pred_h_rb = 82 * h_params['weight'] * h_params['bias'].get('rush', 1.0) * h_mot
            pred_a_rb = 82 * a_params['weight'] * a_params['bias'].get('rush', 1.0) * a_mot

            predicted_home_margin = mu_home - mu_away
            predicted_total = mu_home + mu_away

            pred_home_win = mu_home > mu_away
            actual_home_win = actual_home_score > actual_away_score

            self.metrics["games_played"] += 1

            if pred_home_win == actual_home_win:
                self.metrics["winner_correct"] += 1

            if not pd.isna(vegas_total):
                pred_over = predicted_total > vegas_total
                actual_over = (actual_home_score + actual_away_score) > vegas_total
                if pred_over == actual_over:
                    self.metrics["total_over_under_correct"] += 1

            if not pd.isna(vegas_spread):
                pred_cover = predicted_home_margin > (vegas_spread * -1)
                actual_cover = (actual_home_score - actual_away_score) > (vegas_spread * -1)
                if pred_cover == actual_cover:
                    self.metrics["spread_covers"] += 1

            if not pd.isna(vegas_total) and not pd.isna(vegas_spread):
                implied_home = (vegas_total / 2) + (vegas_spread * -0.5)
                if (mu_home > implied_home) == (actual_home_score > implied_home):
                    self.metrics["team_over_under_correct"] += 1

            if team_col_weekly:
                h_qbs = weekly[(weekly[team_col_weekly] == h_team) & (weekly['week'] == game['week'])]
                if not h_qbs.empty and 'passing_yards' in h_qbs.columns:
                    actual_h_qb = h_qbs['passing_yards'].max()
                    if not pd.isna(actual_h_qb):
                        self.metrics["qb_errors"].append(abs(actual_h_qb - pred_h_qb))
                        self.engine.self_correct(h_team, actual_h_qb, pred_h_qb, 'pass')

                a_rbs = weekly[(weekly[team_col_weekly] == a_team) & (weekly['week'] == game['week'])]
                if not a_rbs.empty and 'rushing_yards' in a_rbs.columns:
                    actual_a_rb = a_rbs['rushing_yards'].max()
                    if not pd.isna(actual_a_rb):
                        self.metrics["rb_errors"].append(abs(actual_a_rb - pred_a_rb))
                        self.engine.self_correct(a_team, actual_a_rb, pred_a_rb, 'rush')

            self.apply_drift_reduction(h_team)
            self.apply_drift_reduction(a_team)

            if self.metrics["games_played"] % 50 == 0:
                print(f"Processed {self.metrics['games_played']} games... Calibrating...")

        self.engine.save_state()
        self.print_report()

    def print_report(self):
        games = max(self.metrics["games_played"], 1)
        print("\n" + "="*50)
        print("🏆 SOTA ENGINE CALIBRATION & BACKTEST REPORT")
        print("="*50)
        print(f"Total Games Processed: {games}")
        print(f"Model Winner Accuracy:   {self.metrics['winner_correct'] / games:.2%}")
        print(f"Model Spread Accuracy:   {self.metrics['spread_covers'] / games:.2%}")
        print(f"Total Points O/U Acc:    {self.metrics['total_over_under_correct'] / games:.2%}")
        print(f"Team Totals O/U Acc:   {self.metrics['team_over_under_correct'] / games:.2%}")

        print("\n📈 PLAYER CALIBRATION METRICS (Mean Absolute Error):")
        qb_mae = np.mean(self.metrics['qb_errors']) if self.metrics['qb_errors'] else 0
        rb_mae = np.mean(self.metrics['rb_errors']) if self.metrics['rb_errors'] else 0
        print(f"QB Pass Yds Expected Error: ±{qb_mae:.1f} yds")
        print(f"RB Rush Yds Expected Error: ±{rb_mae:.1f} yds")

        print("\n⚖️ DRIFT REDUCTION CHECK (Current Top 3 Team Weights):")
        if 'team_params' in self.engine.state:
            sorted_teams = sorted(self.engine.state['team_params'].items(), key=lambda x: x[1].get('weight', 1.0), reverse=True)
            for t, params in sorted_teams[:3]:
                print(f"   {t}: {params.get('weight', 1.0):.3f} (Pass Bias: {params.get('bias', {}).get('pass', 1.0):.2f})")

        print("\n✅ CALIBRATION COMPLETE. State saved to engine_metadata.json.")

if __name__ == "__main__":
    calibrator = BacktestCalibrator(years=[2023, 2024, 2025, 2026])
    calibrator.run_calibration()
