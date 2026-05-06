import os
import requests
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

# ==========================================
# 1. DYNAMIC PATHING & DIRECTORY SETUP
# ==========================================
# This detects if we are on Streamlit Cloud or your local PC
if os.environ.get("STREAMLIT_RUNTIME_ENV"):
    # Cloud Path: /mount/src/horse_pred/
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_FILENAME = "horse_racing_lite.db"
else:
    # Local PC Path
    BASE_DIR = Path(r"E:\horse_pred")
    DB_FILENAME = "horse_racing.db"

MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / DB_FILENAME

# Ensure folders exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. MODEL DOWNLOADER (GOOGLE DRIVE)
# ==========================================
def download_from_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: f.write(chunk)

# Mapping of your Google Drive files
FILES_TO_DOWNLOAD = {
    "et_model.pkl": "1Y_wuhjrsHqZwATZ8vdrUkKEZSnagoFT3",
    "lgb_model.pkl": "1Rk3FPsYAJJyb1HawNMmHP9noOUa2bxmU",
    "rf_model.pkl": "12Jsb161nTdzLVLa5As7X-yucHAtbKfh7",
    "scaler.pkl": "1jlUETq8rh_9NEwzyJlHQxfI9E19wtMyl",
    "xgb_model.pkl": "1oO2fJVltWM3bX8A0A7R2QIXxjjBT1y6G"
}

# Run the downloader before the app starts
def run_initial_setup():
    for filename, drive_id in FILES_TO_DOWNLOAD.items():
        local_path = MODELS_DIR / filename
        if not local_path.exists():
            with st.spinner(f"📥 Downloading {filename}..."):
                download_from_drive(drive_id, local_path)

run_initial_setup()

# ==========================================
# 3. PREDICTOR CLASS
# ==========================================
@st.cache_resource
class EnsemblePredictor:
    def __init__(self):
        # Using the dynamically created MODELS_DIR
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.models = {
            'xgb': joblib.load(MODELS_DIR / "xgb_model.pkl"),
            'rf':  joblib.load(MODELS_DIR / "rf_model.pkl"),
            'et':  joblib.load(MODELS_DIR / "et_model.pkl"),
            'lgb': joblib.load(MODELS_DIR / "lgb_model.pkl")
        }

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Race Predictor", layout="wide")
predictor = EnsemblePredictor()

st.title("🏇 AI Ensemble Predictor")
st.info(f"Connected to: {DB_FILENAME}")

# ... (rest of your GUI/prediction code below) ...