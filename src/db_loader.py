import pandas as pd
import sqlite3
import re
import os
import sys

# Add the root directory to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import UK_IRE_COURSES, DB_PATH, RAW_DATA_PATH

def clean_weight(wgt_str):
    """Converts '9-7' or '10-0' format to total lbs (int)."""
    try:
        if pd.isna(wgt_str) or wgt_str == '':
            return None
        parts = str(wgt_str).split('-')
        if len(parts) == 2:
            return (int(parts[0]) * 14) + int(parts[1])
        return int(wgt_str)
    except:
        return None

def ingest_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create the table with your requested schema + extra features from JSON
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historic_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TIMESTAMP,
            course TEXT,
            race_id INTEGER,
            off TEXT,
            race_name TEXT,
            type TEXT,
            class TEXT,
            pattern TEXT,
            rating_band TEXT,
            age_band TEXT,
            sex_rest TEXT,
            dist TEXT,
            going TEXT,
            ran INTEGER,
            num INTEGER,
            pos INTEGER,
            draw INTEGER,
            ovr_btn NUMERIC,
            btn NUMERIC,
            horse TEXT,
            age INTEGER,
            sex TEXT,
            wgt_lbs INTEGER,
            hg TEXT,
            time TEXT,
            sp TEXT,
            jockey TEXT,
            trainer TEXT,
            prize INTEGER,
            official_rating INTEGER,
            rpr INTEGER,
            ts INTEGER,
            sire TEXT,
            dam TEXT,
            damsire TEXT,
            owner TEXT,
            comment TEXT,
            trainer_rtf INTEGER,
            jockey_allowance INTEGER
        )
    ''')
    conn.commit()

    print(f"Starting ingestion from {RAW_DATA_PATH}...")
    
    chunk_size = 100000
    total_inserted = 0

    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=chunk_size, low_memory=False):
        # 1. Filter for UK/Ireland
        chunk['course'] = chunk['course'].astype(str).str.strip()
        filtered = chunk[chunk['course'].isin(UK_IRE_COURSES)].copy()
        
        if filtered.empty:
            continue

        # 2. Basic Cleaning
        filtered['date'] = pd.to_datetime(filtered['date'], errors='coerce')
        filtered['wgt_lbs'] = filtered['wgt'].apply(clean_weight)
        
        # Map [or] to official_rating to avoid SQL syntax issues with brackets
        filtered = filtered.rename(columns={'[or]': 'official_rating'})

        # Add placeholders for JSON-only features so schema matches
        if 'trainer_rtf' not in filtered.columns:
            filtered['trainer_rtf'] = None
        if 'jockey_allowance' not in filtered.columns:
            filtered['jockey_allowance'] = None

        # Select only the columns that match our DB schema
        db_columns = [
            'date', 'course', 'race_id', 'off', 'race_name', 'type', 'class', 
            'pattern', 'rating_band', 'age_band', 'sex_rest', 'dist', 'going', 
            'ran', 'num', 'pos', 'draw', 'ovr_btn', 'btn', 'horse', 'age', 
            'sex', 'wgt_lbs', 'hg', 'time', 'sp', 'jockey', 'trainer', 'prize', 
            'official_rating', 'rpr', 'ts', 'sire', 'dam', 'damsire', 'owner', 
            'comment', 'trainer_rtf', 'jockey_allowance'
        ]
        
        # Filter dataframe to only these columns (handle missing ones gracefully)
        to_insert = filtered[[c for c in db_columns if c in filtered.columns]]
        
        to_insert.to_sql('historic_results', conn, if_exists='append', index=False)
        
        total_inserted += len(to_insert)
        print(f"Inserted {total_inserted:,} rows...")

    conn.close()
    print("\nDone! Your SQLite database is ready at database/horse_racing.db")

if __name__ == "__main__":
    ingest_data()