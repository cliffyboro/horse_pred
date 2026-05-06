import sqlite3
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Paths
MODELS_DIR = Path("models")
DB_PATH = "E:/horse_pred/database/horse_racing.db"

class RacecardEnsemble:
    def __init__(self):
        print("Initializing Ensemble...")
        # Load Scaler and Models
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.models = {
            'xgb': joblib.load(MODELS_DIR / "xgb_model.pkl"),
            'rf':  joblib.load(MODELS_DIR / "rf_model.pkl"),
            'et':  joblib.load(MODELS_DIR / "et_model.pkl"),
            'lgb': joblib.load(MODELS_DIR / "lgb_model.pkl")
        }
        self.conn = sqlite3.connect(DB_PATH)
        
        # Features must match your training_matrix.csv order exactly
        self.num_features = [
            'wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc', 
            'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5', 
            'days_since_last', 'last_pos', 'class_diff', 'avg_class_3',
            'gear_change', 'or_change', 'true_wgt', 'log_prize'
        ]
        self.cat_cols = ['course', 'going', 'dist', 'class']
        self.all_features = self.num_features + self.cat_cols

    def get_db_history(self, horse_id, trainer_id, jockey_id):
        """Fetches the 'Slight Details' from your SQL features tables."""
        # 1. Get Form History
        form_query = """
            SELECT avg_btn_3, avg_btn_5, last_pos, avg_class_3, hg, official_rating
            FROM features_form f
            JOIN historic_results r ON f.res_id = r.id
            WHERE r.horse_id = ? ORDER BY r.date DESC LIMIT 1
        """
        form = pd.read_sql(form_query, self.conn, params=(horse_id,))
        
        # 2. Get Synergy (Trainer/Jockey Combo)
        syn_query = "SELECT combo_runs, combo_win_pc FROM features_synergy_lookup WHERE t_id = ? AND j_id = ?"
        syn = pd.read_sql(syn_query, self.conn, params=(trainer_id, jockey_id))
        
        return form, syn

    def predict_race(self, race_data):
        """Processes a single race from your JSON and returns ranked runners."""
        race_runners = []
        
        # Race-level info
        prize_raw = race_data.get('prize', '0').replace('€','').replace('£','').replace(',','')
        log_prize = np.log1p(float(prize_raw))
        
        for runner in race_data.get('runners', []):
            if runner.get('non_runner'): continue
            
            # Fetch SQL context
            form, syn = self.get_db_history(runner['horse_id'], runner['trainer_id'], runner['jockey_id'])
            
            # Build the row (using same logic as your micro_signals script)
            hist_form = form.iloc[0] if not form.empty else {}
            
            row = {
                'wgt_lbs': runner.get('lbs', 0),
                'draw': runner.get('draw', 0),
                'combo_runs': syn['combo_runs'].iloc[0] if not syn.empty else 0,
                'combo_win_pc': syn['combo_win_pc'].iloc[0] if not syn.empty else 0.0,
                'h_course_runs': 0, # Placeholder
                'h_course_wins': 0,
                'avg_btn_3': hist_form.get('avg_btn_3', 5.0),
                'avg_btn_5': hist_form.get('avg_btn_5', 5.0),
                'days_since_last': runner.get('last_run', 30),
                'last_pos': hist_form.get('last_pos', 5),
                'class_diff': 0, 
                'avg_class_3': hist_form.get('avg_class_3', 4),
                'gear_change': 1 if runner.get('headgear') != hist_form.get('hg') else 0,
                'or_change': (runner.get('ofr') or 0) - (hist_form.get('official_rating') or 0),
                'true_wgt': runner.get('lbs', 0) - (runner.get('jockey_allowance') or 0),
                'log_prize': log_prize,
                # Categoricals (Need manual mapping or LabelEncoding)
                'course': race_data.get('course_id', 0),
                'going': 0, # Map 'Yielding' -> index
                'dist': race_data.get('distance_f', 0),
                'class': race_data.get('race_class', 0)
            }
            race_runners.append({'name': runner['name'], 'num': runner['number'], 'features': row})

        if not race_runners: return []

        # Convert to DF for scaling and prediction
        pdf = pd.DataFrame([r['features'] for r in race_runners])
        
        # IMPORTANT: Scale numeric features just like in training
        pdf[self.num_features] = self.scaler.transform(pdf[self.num_features])
        
        # Generate Ensemble Predictions
        X = pdf[self.all_features].values.astype(np.float32)
        
        # Probabilities
        xgb_p = self.models['xgb'].predict_proba(X)[:, 1]
        lgb_p = self.models['lgb'].predict_proba(X)[:, 1]
        rf_p  = self.models['rf'].predict_proba(X)[:, 1]
        et_p  = self.models['et'].predict_proba(X)[:, 1]
        
        # Mean Ensemble Score
        final_scores = (xgb_p + lgb_p + rf_p + et_p) / 4
        
        for i, runner in enumerate(race_runners):
            runner['score'] = final_scores[i]
            
        return sorted(race_runners, key=lambda x: x['score'], reverse=True)