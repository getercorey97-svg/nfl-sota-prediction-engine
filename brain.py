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
        """SOTA Fix: Aggregates split EPA/Success columns to prevent KeyErrors."""
        df = df.copy()
        
        # Aggregate all EPA columns (passing_epa, rushing_epa, etc.)
        epa_cols = [c for c in df.columns if 'epa' in c.lower()]
        df['total_epa'] = df[epa_cols].sum(axis=1) if epa_cols else 0
        
        # Aggregate Success columns
        succ_cols = [c for c in df.columns if 'success' in c.lower()]
        df['total_success'] = df[succ_cols].mean(axis=1) if succ_cols else 0

        # Calculate Rolling Averages safely
        df = df.sort_values(['player_id', 'season', 'week'])
        df['eff_rolling'] = df.groupby('player_id')['total_epa'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        return df.fillna(0)

    def self_correct(self, actuals, predictions, position):
        """Self-Learning: Adjusts math based on factual outcomes."""
        if len(actuals) == 0: return
        error = mean_absolute_error(actuals, predictions)
        current_bias = self.state['player_position_bias'].get(position, 1.0)
        
        if error > 10:
            adjustment = 0.03
            if actuals.mean() < predictions.mean():
                self.state['player_position_bias'][position] = max(0.7, current_bias - adjustment)
            else:
                self.state['player_position_bias'][position] = min(1.3, current_bias + adjustment)
        self.save_state()

    def save_state(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.state, f, indent=4)
