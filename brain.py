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

    def find_col(self, df, possibilities):
        """Finds the first matching column name from a list of possibilities."""
        for p in possibilities:
            if p in df.columns:
                return p
        return None

    def discovery_layer(self, df):
        """Robustly aggregates efficiency metrics to prevent crashes."""
        df = df.copy()
        # Dynamically find efficiency/EPA columns
        eff_cols = [c for c in df.columns if any(x in c.lower() for x in ['epa', 'success', 'points'])]
        if eff_cols:
            df['total_eff'] = df[eff_cols].mean(axis=1)
        else:
            df['total_eff'] = df.get('yards_gained', 0)

        # Calculate Rolling Success
        df = df.sort_values(['player_id', 'season', 'week'])
        df['eff_rolling'] = df.groupby('player_id')['total_eff'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        return df.fillna(0)

    def self_correct(self, actuals, predictions, position):
        """The core learning mechanism: Adjusts internal bias based on error."""
        if len(actuals) == 0: return
        error = mean_absolute_error(actuals, predictions)
        current_bias = self.state['player_position_bias'].get(position, 1.0)
        
        # SOTA Logic: If error is high, adjust the 'DNA' for that position
        if error > 5: 
            adj = 0.02
            if actuals.mean() < predictions.mean():
                self.state['player_position_bias'][position] = max(0.6, current_bias - adj)
            else:
                self.state['player_position_bias'][position] = min(1.4, current_bias + adj)
        self.save_state()

    def save_state(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.state, f, indent=4)
