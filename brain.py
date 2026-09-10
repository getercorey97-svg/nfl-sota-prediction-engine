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
        """Analyzes correlations across years and seasonal segments."""
        df = df.copy()
        df['season_segment'] = df['week'].apply(self.get_season_segment)
        
        # Calculate Segment-Based Historical Averages
        segment_stats = df.groupby(['player_id', 'season', 'season_segment'])['passing_yards'].mean().reset_index()
        segment_stats.rename(columns={'passing_yards': 'segment_avg_yards'}, inplace=True)
        
        # Cross-year correlation: How did they do in this segment last year?
        segment_stats['prev_year_segment_avg'] = segment_stats.groupby(['player_id', 'season_segment'])['segment_avg_yards'].shift(1)
        
        df = df.merge(segment_stats[['player_id', 'season', 'season_segment', 'prev_year_segment_avg']], 
                     on=['player_id', 'season', 'season_segment'], how='left')

        # Features for SOTA Prediction
        df['epa_per_play'] = pd.to_numeric(df['epa'], errors='coerce')
        features = ['epa_per_play', 'prev_year_segment_avg']
        
        return df.fillna(0), features

    def self_correct(self, actuals, predictions):
        """Self-Updating Logic: Adjusts team DNA based on prediction error."""
        for team in actuals['recent_team'].unique():
            team_mask = actuals['recent_team'] == team
            if not any(team_mask): continue
            
            error = mean_absolute_error(actuals[team_mask]['passing_yards'], predictions[team_mask])
            current_weight = self.state['team_weights'].get(team, 1.0)
            
            # Evolution: If the engine was too high or low, adjust weights
            if error > 15:
                adjustment = 0.03
                if actuals[team_mask]['passing_yards'].mean() < predictions[team_mask].mean():
                    self.state['team_weights'][team] = max(0.5, current_weight - adjustment)
                else:
                    self.state['team_weights'][team] = min(1.5, current_weight + adjustment)
        self.save_state()

    def calculate_confidence(self, win_prob):
        base = 72.0
        # More extreme win probabilities (high or low) increase engine confidence
        variance_bonus = abs(win_prob - 0.5) * 40
        return round(min(98.0, base + variance_bonus), 2)

    def save_state(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.state, f, indent=4)
