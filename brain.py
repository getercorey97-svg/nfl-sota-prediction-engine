import pandas as pd
import numpy as np
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
            "player_position_bias": {"QB": 1.0, "RB": 1.0, "WR": 1.0},
            "history": []
        }

    def discovery_layer(self, df):
        """Robustly finds efficiency metrics even if column names shift."""
        df = df.copy()
        
        # Find any column containing 'epa' (case-insensitive)
        epa_cols = [c for c in df.columns if 'epa' in c.lower()]
        df['total_eff'] = df[epa_cols].sum(axis=1) if epa_cols else df.get('fantasy_points_ppr', 0)
        
        # Safe Rolling Average
        df = df.sort_values(['player_id', 'season', 'week'])
        df['eff_rolling'] = df.groupby('player_id')['total_eff'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
        return df.fillna(0)

    def self_correct(self, actuals, predictions, position):
        """Learns and updates its own math based on the outcome of games."""
        if len(actuals) == 0: return
        error = mean_absolute_error(actuals, predictions)
        current_bias = self.state['player_position_bias'].get(position, 1.0)
        
        if error > 8: # High sensitivity for state-of-the-art accuracy
            adjustment = 0.03
            if actuals.mean() < predictions.mean():
                self.state['player_position_bias'][position] = max(0.6, current_bias - adjustment)
            else:
                self.state['player_position_bias'][position] = min(1.4, current_bias + adjustment)
        self.save_state()

    def save_state(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.state, f, indent=4)
