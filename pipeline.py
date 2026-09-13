import nfl_data_py as nfl
import pandas as pd
import numpy as np
import datetime
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

def dixon_coles_adjustment(home_goals, away_goals, rho, mu_x, mu_y):
    """Dixon-Coles adjustment for low-score dependencies in Poisson model."""
    if home_goals == 0 and away_goals == 0:
        return 1 - rho * mu_x * mu_y
    elif home_goals == 0 and away_goals == 1:
        return 1 + rho * mu_x
    elif home_goals == 1 and away_goals == 0:
        return 1 + rho * mu_y
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0

def gaussian_copula_correlation(u, v, rho):
    """Gaussian Copula density for modeling QB-WR yard correlation."""
    z1 = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    z2 = norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
    denom = 2 * np.pi * np.sqrt(1 - rho**2)
    exponent = -(z1**2 - 2 * rho * z1 * z2 + z2**2) / (2 * (1 - rho**2))
    copula_density = np.exp(exponent) / denom
    return copula_density / (norm.pdf(z1) * norm.pdf(z2))

def geter_context_factor(team, schedule_row, weekly_stats, team_col_weekly, name_col, engine_state):
    """
    The Geter Principle 2.0: Maps physical fatigue alongside Time-of-Season Motivation.
    """
    context_multiplier = 1.0

    # 1. TRAVEL FATIGUE
    tz_map = {
        'BUF': -5, 'MIA': -5, 'NE': -5, 'NYJ': -5, 'BAL': -5, 'CIN': -5, 'CLE': -5, 'PIT': -5,
        'HOU': -6, 'IND': -5, 'JAX': -5, 'TEN': -6, 'DEN': -7, 'KC': -6, 'LV': -8, 'LAC': -8,
        'DAL': -6, 'NYG': -5, 'PHI': -5, 'WAS': -5, 'CHI': -6, 'DET': -5, 'GB': -6, 'MIN': -6,
        'ATL': -5, 'CAR': -5, 'NO': -6, 'TB': -5, 'ARI': -7, 'LAR': -8, 'SF': -8, 'SEA': -8
    }
    home_tz = tz_map.get(schedule_row['home_team'], -5)
    away_tz = tz_map.get(schedule_row['away_team'], -5)
    if abs(home_tz - away_tz) >= 3:
        context_multiplier *= 0.97

    # 2. TIME OF SEASON & MOTIVATION MAPPING
    current_week = schedule_row.get('week', 1)

    # Only apply motivation adjustments late in the season (Week 14 onwards)
    if current_week >= 14:
        # Fetch the historical 5-year motivation index calculated in the seed
        team_motivation = engine_state['team_params'].get(team, {}).get('motivation_index', 1.0)

        # Scale the motivation impact based on how late it is in the season.
        # Week 14 applies 20% of the trait, Week 18 applies 100% of the trait.
        lateness_scale = min(1.0, (current_week - 13) / 5.0)

        # Apply the weighted motivation to the context multiplier
        active_motivation = 1.0 + ((team_motivation - 1.0) * lateness_scale)
        context_multiplier *= active_motivation

    return max(0.80, min(1.20, context_multiplier))

def run_realtime_cycle():
    engine = NFLMetaEngine()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    year = now_utc.year

    print(f"--- NFL SOTA ENGINE START: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC ---")

    sched = safe_load(['import_schedules', 'load_schedules'], [year])
    if sched.empty:
        print("Schedule data unavailable. Check year/API status.")
        return

    sched.columns = sched.columns.str.lower()
    sched['gametime_dt'] = pd.to_datetime(sched['gametime'], utc=True)

    window_start = now_utc - datetime.timedelta(hours=18)
    window_end = now_utc + datetime.timedelta(hours=6)
    active_games = sched[(sched['gametime_dt'] >= window_start) & (sched['gametime_dt'] <= window_end)]

    if active_games.empty:
        print("No games currently scheduled for today.")
        return

    # Using the updated safe_load to fetch data individually by year
    depth = safe_load(['import_depth_charts', 'load_depth_charts'], [year])
    injuries = safe_load(['import_injuries', 'load_injuries'], [year])
    weekly_stats = safe_load(['import_weekly_data', 'load_weekly_data'], [year - 1, year])
    seasonal_stats = safe_load(['import_seasonal_data', 'load_seasonal_data'], [year - 1, year])

    for df in [depth, injuries, weekly_stats, seasonal_stats]:
        if not df.empty:
            df.columns = df.columns.str.lower()

    score_col = find_col(sched, ['home_score', 'score_home', 'total_home_score'])
    name_col = find_col(weekly_stats, ['player_display_name', 'player_name', 'full_name'])
    team_col_depth = find_col(depth, ['club', 'team', 'team_abbr'])

    # SCHEMA FIX: 'pos_abb', 'pos_name', 'pos_rank' added for the 2026 schema changes
    depth_rank_col = find_col(depth, ['depth_team', 'depth', 'depth_order', 'rank', 'pos_rank'])
    team_col_weekly = find_col(weekly_stats, ['recent_team', 'team', 'team_abbr'])

    dc_rho = engine.state.get('model_params', {}).get('dixon_coles_rho', 0.13)
    qb_wr_rho = engine.state.get('model_params', {}).get('qb_wr_correlation', 0.45)

    for _, game in active_games.iterrows():
        h_team, a_team = game['home_team'], game['away_team']
        is_final = score_col and not pd.isna(game[score_col])
        status = "FINAL (LEARNING)" if is_final else "UPCOMING/LIVE (PREDICTING)"

        print(f"\n[{status}] {a_team} @ {h_team}")

        for team in [a_team, h_team]:
            params = engine.state['team_params'].get(team, {"weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}})
            if team_col_depth and not depth.empty:
                team_roster = depth[depth[team_col_depth] == team]
                if depth_rank_col:
                    starters = team_roster.sort_values(depth_rank_col)
                else:
                    starters = team_roster
            else:
                starters = pd.DataFrame()

            print(f"  > {team} (Intelligence Weight: {params['weight']:.2f})")

            # Applying the new Context/Motvation logic
            fatigue = get_realtime_context_factor(team, game, weekly_stats, team_col_weekly, name_col, engine.state) \
                if 'get_realtime_context_factor' in globals() else geter_context_factor(team, game, weekly_stats, team_col_weekly, name_col, engine.state)

            qb_pred = None
            wr_preds = []

            for pos in ['QB', 'RB', 'WR', 'TE']:
                if starters.empty:
                    print(f"    {pos} Data Missing for this team.")
                    continue

                # SCHEMA FIX: 'pos_abb', 'pos_name' added
                pos_col = find_col(starters, ['position', 'pos', 'position_group', 'pos_abb', 'pos_name'])
                if pos_col is None:
                    print(f"    WARNING: No position column found in depth chart for {team}. Skipping {pos}.")
                    continue

                pos_starters = starters[starters[pos_col] == pos]
                active_player = None

                for i in range(len(pos_starters)):
                    candidate = pos_starters.iloc[i]
                    name_col_candidate = find_col(pd.DataFrame([candidate]), ['full_name', 'player_name', 'name'])
                    name_to_check = candidate.get(name_col_candidate, '') if name_col_candidate else ''
                    c_clean = clean_name(name_to_check)

                    if not injuries.empty:
                        injury_name_col = find_col(injuries, ['full_name', 'player_name', 'name'])
                        injury_team_col = find_col(injuries, ['team', 'club', 'team_abbr'])
                        if injury_name_col and injury_team_col:
                            p_injury = injuries[(injuries[injury_team_col] == team) & (injuries[injury_name_col].apply(clean_name) == c_clean)]
                            
                            # CRITICAL FIX: Wrap the get() result in str() to prevent NoneType attribute errors
                            if p_injury.empty or str(p_injury.iloc[0].get('report_status', '')).lower() not in ['out', 'inactive']:
                                active_player = candidate
                                break
                            else:
                                print(f"    [OUT] {pos} {name_to_check} is scratched.")
                        else:
                            active_player = candidate
                            break
                    else:
                        active_player = candidate
                        break

                if active_player is not None:
                    name_col_active = find_col(pd.DataFrame([active_player]), ['full_name', 'player_name', 'name'])
                    p_name = active_player.get(name_col_active, 'Unknown') if name_col_active else 'Unknown'
                    cat = 'pass' if pos in ['QB', 'WR', 'TE'] else 'rush'
                    b = params['bias'].get(cat, 1.0)

                    base_yds = {'QB': 258, 'RB': 82, 'WR': 88, 'TE': 55}[pos]
                    pred_yds = base_yds * params['weight'] * b * fatigue

                    if pos == 'QB':
                        qb_pred = pred_yds
                    elif pos in ['WR', 'TE']:
                        wr_preds.append((p_name, pred_yds))

                    if is_final and not weekly_stats.empty:
                        actual = weekly_stats[(weekly_stats[name_col].apply(clean_name) == clean_name(p_name)) & (weekly_stats[team_col_weekly] == team)]
                        if not actual.empty:
                            stat_map = {'QB': 'passing_yards', 'RB': 'rushing_yards', 'WR': 'receiving_yards', 'TE': 'receiving_yards'}
                            actual_yds = actual.iloc[0].get(stat_map[pos], 0)
                            print(f"    {pos} {p_name}: Predicted {pred_yds:.1f} | Actual {actual_yds:.1f} (Context: {fatigue:.2f})")
                            engine.self_correct(team, actual_yds, pred_yds, cat)
                    else:
                        label = {"QB": "Pass Yds", "RB": "Rush Yds", "WR": "Rec Yds", "TE": "Rec Yds"}[pos]

                        if pos in ['WR', 'TE'] and qb_pred is not None:
                            qb_mu, qb_sigma = np.log(max(qb_pred, 1)), 0.35
                            wr_mu, wr_sigma = np.log(max(pred_yds, 1)), 0.45
                            u = norm.cdf((np.log(max(pred_yds, 1)) - wr_mu) / wr_sigma)
                            v = norm.cdf((np.log(max(qb_pred, 1)) - qb_mu) / qb_sigma)
                            copula_adj = gaussian_copula_correlation(u, v, qb_wr_rho)
                            pred_yds *= (0.8 + 0.4 * copula_adj)

                        print(f"    {pos} {p_name}: {pred_yds:.1f} {label} (Context: {fatigue:.2f})")

            if not is_final:
                off_weight = params['weight']
                home_adv = 1.05 if team == h_team else 0.95
                mu = 22.5 * off_weight * home_adv * fatigue

                if 'team_mus' not in locals():
                    team_mus = {}
                team_mus[team] = mu

        if not is_final and 'team_mus' in locals() and len(team_mus) == 2:
            mu_home = team_mus.get(h_team, 22.5)
            mu_away = team_mus.get(a_team, 22.5)

            max_score = 50
            home_win = away_win = tie = 0.0
            for hg in range(max_score):
                for ag in range(max_score):
                    p_h = poisson.pmf(hg, mu_home)
                    p_a = poisson.pmf(ag, mu_away)
                    dc_adj = dixon_coles_adjustment(hg, ag, dc_rho, mu_home, mu_away)
                    prob = p_h * p_a * dc_adj
                    if hg > ag:
                        home_win += prob
                    elif ag > hg:
                        away_win += prob
                    else:
                        tie += prob

            print(f"  >>> Game Prediction: {h_team} {mu_home:.1f} - {mu_away:.1f} {a_team}")
            print(f"  >>> Win Prob: {h_team} {home_win:.1%} | {a_team} {away_win:.1%} | Tie {tie:.1%}")

            del team_mus

    engine.save_state()
    print("\n--- CYCLE COMPLETE: ALL INTELLIGENCE SYNCED ---")

if __name__ == "__main__":
    run_realtime_cycle()
