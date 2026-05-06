import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# 1. SILENCE LIGHTGBM/SKLEARN WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIG ---
DB_PATH = r"E:\horse_pred\database\horse_racing.db"
MODELS_DIR = Path("models")
RACECARDS_DIR = Path("racecards")

class EnsemblePredictor:
    def __init__(self):
        print("🚀 Loading Ensemble Models & Scaler...")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.models = {
            'xgb': joblib.load(MODELS_DIR / "xgb_model.pkl"),
            'rf':  joblib.load(MODELS_DIR / "rf_model.pkl"),
            'et':  joblib.load(MODELS_DIR / "et_model.pkl"),
            'lgb': joblib.load(MODELS_DIR / "lgb_model.pkl")
        }
        self.conn = sqlite3.connect(DB_PATH)
        
        self.num_features = [
            'wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc',
            'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5',
            'days_since_last', 'last_pos', 'class_diff', 'avg_class_3',
            'gear_change', 'or_change', 'true_wgt', 'log_prize'
        ]
        self.cat_cols = ['course', 'going', 'dist', 'class']
        self.all_features = self.num_features + self.cat_cols

    def get_horse_context(self, horse_name):
        query = f"""
            SELECT 
                f.avg_btn_3, f.avg_btn_5, f.last_pos,
                s.combo_runs, s.combo_win_pc,
                c.h_course_runs, c.h_course_wins,
                cl.class_diff, cl.avg_class_3,
                m.gear_change, m.or_change, m.true_wgt, m.log_prize,
                r.hg, r.official_rating
            FROM historic_results r
            LEFT JOIN features_form f    ON r.id = f.res_id
            LEFT JOIN features_synergy s ON r.id = s.res_id
            LEFT JOIN features_course c  ON r.id = c.res_id
            LEFT JOIN features_class cl  ON r.id = cl.res_id
            LEFT JOIN features_micro m   ON r.id = m.id
            WHERE r.horse = "{horse_name.replace("'", "''")}"
            ORDER BY r.date DESC LIMIT 1
        """
        df = pd.read_sql(query, self.conn)
        return df.iloc[0].to_dict() if not df.empty else None

    def predict_race(self, race_json):
        runners_data = []
        raw_prize = str(race_json.get('prize', '0')).replace('€','').replace('£','').replace(',','')
        log_p = np.log1p(float(raw_prize) if raw_prize else 0)

        for r in race_json.get('runners', []):
            if r.get('non_runner'): continue
            db_data = self.get_horse_context(r['name'])
            
            row = {
                'wgt_lbs': r.get('lbs', 0),
                'draw': r.get('draw', 0),
                'combo_runs': db_data.get('combo_runs', 0) if db_data else 0,
                'combo_win_pc': db_data.get('combo_win_pc', 0.0) if db_data else 0.0,
                'h_course_runs': db_data.get('h_course_runs', 0) if db_data else 0,
                'h_course_wins': db_data.get('h_course_wins', 0) if db_data else 0,
                'avg_btn_3': db_data.get('avg_btn_3', 6.0) if db_data else 6.0,
                'avg_btn_5': db_data.get('avg_btn_5', 6.0) if db_data else 6.0,
                'days_since_last': r.get('last_run', 30),
                'last_pos': db_data.get('last_pos', 5) if db_data else 5,
                'class_diff': db_data.get('class_diff', 0) if db_data else 0,
                'avg_class_3': db_data.get('avg_class_3', 4.0) if db_data else 4.0,
                'gear_change': 1 if db_data and r.get('headgear') != db_data.get('hg') else 0,
                'or_change': (r.get('ofr') or 0) - (db_data.get('official_rating') or 0) if db_data else 0,
                'true_wgt': r.get('lbs', 0) - (r.get('jockey_allowance') or 0),
                'log_prize': log_p,
                'course': race_json.get('course_id', 0),
                'going': 0, 
                'dist': race_json.get('distance_f', 0),
                'class': race_json.get('race_class', 0) or 4
            }
            runners_data.append({'name': r['name'], 'num': r['number'], 'features': row})

        if not runners_data: return []
        
        df = pd.DataFrame([rd['features'] for rd in runners_data])
        df[self.num_features] = self.scaler.transform(df[self.num_features])
        X = df[self.all_features].values.astype(np.float32)
        
        # Ensemble predictions
        p_xgb = self.models['xgb'].predict_proba(X)[:, 1]
        p_lgb = self.models['lgb'].predict_proba(df[self.all_features])[:, 1]
        p_rf  = self.models['rf'].predict_proba(X)[:, 1]
        p_et  = self.models['et'].predict_proba(X)[:, 1]
        
        final_scores = (p_xgb * 0.40) + (p_lgb * 0.30) + (p_rf * 0.15) + (p_et * 0.15)
        
        for i, rd in enumerate(runners_data):
            rd['score'] = final_scores[i]
            
        return sorted(runners_data, key=lambda x: x['score'], reverse=True)

def main():
    # 1. Setup Predictor
    predictor = EnsemblePredictor()

    # 2. Load Racecards
    files = sorted(RACECARDS_DIR.glob("*.json"), reverse=True)
    if not files:
        print("❌ No racecards found.")
        return
    
    with open(files[0], encoding='utf-8') as f:
        data = json.load(f)

    selection_map = {}
    counter = 1
    
    print("\n" + "="*80)
    print(f"📊 HORSE RACING ENSEMBLE PREDICTOR | {files[0].stem}")
    print("="*80)

    # 3. Build Selection Map (Inside Main)
    for region, courses in data.items():
        prefix = "G" if region.upper() == "GB" else "H"
        for course, races in courses.items():
            print(f"\n🏟️  {course.upper()}")
            for time, race in sorted(races.items()):
                code = f"{prefix}{counter}"
                selection_map[code] = (course, time, race) # Store as tuple
                print(f"  [{code}] {time} - {race.get('race_name')[:45]}")
                counter += 1

    # 4. User Interaction Loop
    while True:
        cmd = input("\nEnter Race Code (e.g. H1), 'ALL', or 'Q' to quit: ").strip().upper()
        if cmd == 'Q': break
        
        # This now safely uses the selection_map defined above
        codes = list(selection_map.keys()) if cmd == 'ALL' else [c.strip() for c in cmd.split(',')]
        
        for code in codes:
            if code in selection_map:
                course_name, time, race = selection_map[code]
                
                print(f"\n{'='*65}")
                print(f" 🏆 {course_name.upper()} {time} | {race.get('race_name')}")
                print(f"{'='*65}")
                print(f"{'RANK':<5} | {'#':<3} | {'HORSE':<28} | {'SCORE':<10}")
                print("-" * 65)
                
                results = predictor.predict_race(race)
                for i, res in enumerate(results[:5], 1):
                    print(f"{i:<5} | {res['num']:<3} | {res['name']:<28} | {res['score']:.4f}")
            else:
                print(f"⚠️  Code {code} not found.")

if __name__ == "__main__":
    main()