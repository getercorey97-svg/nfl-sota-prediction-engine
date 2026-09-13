import pandas as pd
import numpy as np
import json
import os
from scipy.stats import norm, poisson
from sklearn.metrics import mean_absolute_error

class NFLMetaEngine:
    def __init__(self):
        self.meta_path = 'engine_metadata.json'
        self.state = self.load_state()

    def load_state(self):
        default_teams = ['ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC','LV','LAC','LAR','MIA','MIN','NE','NO','NYG','NYJ','PHI','PIT','SF','SEA','TB','TEN','WAS']
        default_state = {
            "team_params": {t: {"lr": 0.05, "weight": 1.0, "bias": {"pass": 1.0, "rush": 1.0}} for t in default_teams},
            "history": [],
            "model_params": {
                "learning_rate": 0.05,
                "dixon_coles_rho": 0.13,
                "qb_wr_correlation": 0.45,
                "fatigue_decay": 0.02
            }
        }
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r') as f:
                    loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key in default_state:
                    if key not in loaded:
                        loaded[key] = default_state[key]
                # Ensure all teams exist
                for t in default_teams:
                    if t not in loaded["team_params"]:
                        loaded["team_params"][t] = default_state["team_params"][t]
                return loaded
            except Exception as e:
                print(f"Error loading state: {e}")
                return default_state
        return default_state

    def self_correct(self, team, actual_val, pred_val, category='pass'):
        """Updates team-specific DNA based on performance errors with adaptive learning rate."""
        params = self.state['team_params'].get(team)
        if not params: return
        
        error = actual_val - pred_val
        lr = params['lr']
        
        # Adaptive learning rate based on error magnitude
        if abs(error) > 2:
            direction = 1 if error > 0 else -1
            # Scale update by relative error
            rel_error = abs(error) / max(pred_val, 1)
            update_magnitude = lr * min(rel_error, 0.3)  # Cap at 30% relative error
            
            params['weight'] += direction * update_magnitude * 0.1
            params['bias'][category] += direction * update_magnitude * 0.05
            
            # Maintain stability constraints
            params['weight'] = max(0.5, min(1.5, params['weight']))
            params['bias'][category] = max(0.5, min(1.5, params['bias'][category]))
            
            # Decay learning rate slightly after large corrections
            params['lr'] = max(0.01, params['lr'] * 0.99)
        elif abs(error) < 5:
            # Small errors: slightly increase LR for faster adaptation
            params['lr'] = min(0.1, params['lr'] * 1.001)
        
        # Record in history
        self.state['history'].append({
            "team": team,
            "category": category,
            "predicted": float(pred_val),
            "actual": float(actual_val),
            "error": float(error),
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Keep history bounded
        if len(self.state['history']) > 10000:
            self.state['history'] = self.state['history'][-5000:]
            
        self.save_state()

    def save_state(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.state, f, indent=4, default=str)

    def estimate_dixon_coles_rho(self):
        """Estimate Dixon-Coles rho from historical scoring data."""
        # This would use historical game scores to MLE estimate rho
        # Placeholder returns current value
        return self.state.get('model_params', {}).get('dixon_coles_rho', 0.13)
    
    def estimate_qb_wr_correlation(self):
        """Estimate QB-WR correlation from historical player stats."""
        return self.state.get('model_params', {}).get('qb_wr_correlation', 0.45)