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
        default_teams = ['ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC','LV','LAC','LAR','MIA','MIN','NE','NO','NYG','NYJ','PHI','PIT','SF','SEA','TB','TEN','WAS']
        default_state = {
            "team_params": {t: {"lr": 0.05, "weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}} for t in default_teams},
            "history": []
        }
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r') as f:
                    loaded = json.load(f)
                # Ensure all keys exist
                if "team_params" not in loaded: loaded["team_params"] = default_state["team_params"]
                if "history" not in loaded: loaded["history"] = []
                return loaded
            except:
                return default_state
        return default_state

    def self_correct(self, team, actual_val, pred_val, category='pass'):
        params = self.state['team_params'].get(team)
        if not params: return
        error = actual_val - pred_val
        lr = params['lr']
        if abs(error) > 2:
            direction = 1 if error > 0 else -1
            params['weight'] += (direction * lr * 0.1)
            params['bias'][category] += (direction * lr * 0.05)
            params['weight'] = max(0.5, min(1.5, params['weight']))
            params['bias'][category] = max(0.5, min(1.5, params['bias'][category]))
        self.save_state()

    def save_state(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.state, f, indent=4)
