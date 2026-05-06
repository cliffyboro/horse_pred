import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
import joblib
import warnings
from pathlib import Path

# 1. SILENCE TECHNICAL CLUTTER
warnings.filterwarnings("ignore", category=UserWarning)

# --- ABSOLUTE PATH CONFIG ---
# This ensures the script finds your files regardless of where you launch it from
BASE_DIR = Path(r"E:\horse_pred")
DB_PATH = BASE_DIR / "database" / "horse_racing.db"
MODELS_DIR = BASE_DIR / "models"
RACECARDS_DIR = BASE_DIR / "racecards"

# 2. CACHED PREDICTOR ENGINE
@st.cache_resource
class EnsemblePredictor:
    def __init__(self):
        # Use absolute paths for loading
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.models = {
            'xgb': joblib.load(MODELS_DIR / "xgb_model.pkl"),
            'rf':  joblib.load(MODELS_DIR / "rf_model.pkl"),
            'et':  joblib.load(MODELS_DIR / "et_model.pkl"),
            'lgb': joblib.load(MODELS_DIR / "lgb_model.pkl")
        }
        
        # FIX: Force XGBoost to CPU to prevent CUDA memory crashes
        if hasattr(self.models['xgb'], "set_params"):
            self.models['xgb'].set_params(device="cpu")
            
        self.num_features = [
            'wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc',
            'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5',
            'days_since_last', 'last_pos', 'class_diff', 'avg_class_3',
            'gear_change', 'or_change', 'true_wgt', 'log_prize'
        ]
        self.cat_cols = ['course', 'going', 'dist', 'class']
        self.all_features = self.num_features + self.cat_cols

    def get_horse_context(self, horse_name):
        conn = sqlite3.connect(DB_PATH)
        query = f"""
            SELECT f.avg_btn_3, f.avg_btn_5, f.last_pos, s.combo_runs, s.combo_win_pc,
                   c.h_course_runs, c.h_course_wins, cl.class_diff, cl.avg_class_3,
                   m.gear_change, m.or_change, m.true_wgt, m.log_prize, r.hg, r.official_rating
            FROM historic_results r
            LEFT JOIN features_form f    ON r.id = f.res_id
            LEFT JOIN features_synergy s ON r.id = s.res_id
            LEFT JOIN features_course c  ON r.id = c.res_id
            LEFT JOIN features_class cl  ON r.id = cl.res_id
            LEFT JOIN features_micro m   ON r.id = m.id
            WHERE r.horse = ? ORDER BY r.date DESC LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(horse_name,))
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else None

    def predict_race(self, race_json):
        runners_data = []
        raw_prize = str(race_json.get('prize', '0')).replace('€','').replace('£','').replace(',','')
        log_p = np.log1p(float(raw_prize) if raw_prize else 0)

        for r in race_json.get('runners', []):
            if r.get('non_runner'): continue
            db_data = self.get_horse_context(r['name'])
            
            feat = {
                'wgt_lbs': r.get('lbs', 0), 'draw': r.get('draw', 0),
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
                'log_prize': log_p, 'course': race_json.get('course_id', 0), 'going': 0, 
                'dist': race_json.get('distance_f', 0), 'class': race_json.get('race_class', 0) or 4
            }
            runners_data.append({'#': r['number'], 'Horse': r['name'], 'features': feat})

        if not runners_data: return pd.DataFrame()
        
        df_feat = pd.DataFrame([rd['features'] for rd in runners_data])
        df_feat[self.num_features] = self.scaler.transform(df_feat[self.num_features])
        X = df_feat[self.all_features].values.astype(np.float32)
        
        # Ensemble Logic (Passing DataFrame to LGBM to avoid feature name warnings)
        p_xgb = self.models['xgb'].predict_proba(X)[:, 1]
        p_lgb = self.models['lgb'].predict_proba(df_feat[self.all_features])[:, 1]
        p_rf  = self.models['rf'].predict_proba(X)[:, 1]
        p_et  = self.models['et'].predict_proba(X)[:, 1]
        
        scores = (p_xgb * 0.40) + (p_lgb * 0.30) + (p_rf * 0.15) + (p_et * 0.15)
        
        res_df = pd.DataFrame(runners_data)[['#', 'Horse']]
        res_df['Model Score'] = scores
        return res_df.sort_values('Model Score', ascending=False)

# 3. STREAMLIT UI SETUP
st.set_page_config(page_title="AI Race Predictor", layout="wide")
predictor = EnsemblePredictor()

st.title("🏇 AI Ensemble Predictor")

# Load Racecards
files = sorted(RACECARDS_DIR.glob("*.json"), reverse=True)
if not files:
    st.error("No racecard files found in /racecards")
    st.stop()

with open(files[0], encoding='utf-8') as f:
    racecard_data = json.load(f)

# Sidebar Navigation
st.sidebar.header("Race Selection")
all_races = []
for region, courses in racecard_data.items():
    for course, races in courses.items():
        for time, race in races.items():
            all_races.append({
                'label': f"{course.upper()} {time}",
                'data': race,
                'course': course,
                'time': time
            })

selected_race_label = st.sidebar.selectbox("Select a Race", [r['label'] for r in all_races])
selected_race = next(r for r in all_races if r['label'] == selected_race_label)

# 4. EXECUTE PREDICTION & DISPLAY
st.subheader(f"📊 {selected_race['label']} - {selected_race['data'].get('race_name')}")

with st.spinner('Analyzing runners...'):
    results = predictor.predict_race(selected_race['data'])

if not results.empty:
    # Highlights
    top_horse = results.iloc[0]
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.metric("Top Pick", top_horse['Horse'], f"{top_horse['Model Score']:.4f}")
        if top_horse['Model Score'] > 0.15:
            st.success("🔥 High Confidence Play")
            
    with c2:
        # Table Styling
        def color_high_scores(val):
            color = '#d4edda' if val >= 0.15 else '#fff3cd' if val >= 0.11 else '#f8d7da'
            return f'background-color: {color}'

        st.dataframe(
            results.style.map(color_high_scores, subset=['Model Score'])
            .format({'Model Score': '{:.4f}'}),
            width="stretch"
        )
else:
    st.warning("No runner data found for this race.")