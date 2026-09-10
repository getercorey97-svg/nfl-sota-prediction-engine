import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from sklearn.metrics import mean_absolute_error

class NFLMetaEngine:
    def __init__(self):
        self.meta_path = 'engine_metadata.json'
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                return json.load(f)
        return {
            "team_weights": {},
            "history": [],
            "seasonal_trends": {},
            "model_params": {"learning_rate": 0.05}
        }

    def get_season_segment(self, week):
        if week <= 6: return 'early'
        if week <= 12: return 'mid'
        return 'late'

    def discovery_layer(self, df):
        """Robustly analyzes correlations across years and seasonal segments."""
        df = df.copy()
        df['season_segment'] = df['week'].apply(self.get_season_segment)
        
        # SOTA FIX: Dynamically find EPA columns or use Fantasy Points as a proxy
        epa_cols = [c for c in df.columns if 'epa' in c.lower()]
        if epa_cols:
            df['total_epa'] = df[epa_cols].sum(axis=1)
        else:
            # Fallback to PPR points if EPA is missing in that specific data slice
            df['total_epa'] = df.get('fantasy_points_ppr', 0)

        # Calculate Segment-Based Historical Averages
        segment_stats = df.groupby(['player_id', 'season', 'season_segment'])['passing_yards'].mean().reset_index()
        segment_stats.rename(columns={'passing_yards': 'segment_avg_yards'}, inplace=True)
        
        # Cross-year correlation: How did they do in this segment last year?
        segment_stats['prev_year_segment_avg'] = segment_stats.groupby(['player_id', 'season_segment'])['segment_avg_yards'].shift(1)
        
        df = df.merge(segment_stats[['player_id', 'season', 'season_segment', 'prev_year_segment_avg']], 
                     on=['player_id', 'season', 'season_segment'], how='left')

        # Use 'total_epa' instead of a hardcoded 'epa' column
        df['efficiency_rating'] = pd.to_numeric(df['total_epa'], errors='coerce').fillna(0)
        features = ['efficiency_rating', 'prev_year_segment_avg']
        
        return df.fillna(0), features

    def self_correct(self, actuals, predictions):
        """Self-Updating Logic: Adjusts team DNA based on prediction error."""
        # Weekly data uses 'recent_team' as the column name
        team_col = 'recent_team' if 'recent_team' in actuals.columns else 'team'
        
        for team in actuals[team_col].unique():
            team_mask = actuals[team_col] == team
            if not any(team_mask): continue
            
            actual_yards = actuals[team_mask]['passing_yards']
            error = mean_absolute_error(actual_yards, predictions[team_mask])
            current_weight = self.state['team_weights'].get(team, 1.0)
            
            if error > 15:
                adjustment = 0.03
                if actual_yards.mean() < predictions[team_mask].mean():
                    self.state['team_weights'][team] = max(0.5, current_weight - adjustment)
                else:
                    self.state['team_weights'][team] = min(1.5, current_weight + adjustment)
        self.save_state()

    def calculate_confidence(self, win_prob):
        base = 72.0
        variance_bonus = abs(win_prob - 0.5) * 40
        return round(min(98.0, base + variance_bonus), 2)

    def save_state(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.state, f, indent=4)
