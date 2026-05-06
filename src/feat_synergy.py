import sqlite3
import pandas as pd
import os
import sys
from collections import deque

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def generate_synergy_features():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Create the Features Table
    conn.execute("DROP TABLE IF EXISTS features_synergy")
    conn.execute('''
        CREATE TABLE features_synergy (
            res_id INTEGER PRIMARY KEY,
            combo_runs INTEGER,
            combo_wins INTEGER,
            combo_win_pc REAL,
            combo_place_pc REAL,
            FOREIGN KEY(res_id) REFERENCES historic_results(id)
        )
    ''')

    print("Loading data for synergy calculation...")
    df = pd.read_sql("""
        SELECT id, date, trainer, jockey, pos 
        FROM historic_results 
        ORDER BY date ASC
    """, conn)
    
    df['date'] = pd.to_datetime(df['date'])
    
    # FIX: Safely convert pos to numeric, turning 'DSQ', 'PU', etc., into NaN
    pos_numeric = pd.to_numeric(df['pos'], errors='coerce')
    df['is_win'] = (pos_numeric == 1).astype(int)
    df['is_place'] = (pos_numeric <= 3).astype(int)

    # Dictionary to track history: { (trainer, jockey): {'q': deque, 'w': sum_wins, 'p': sum_places} }
    history = {}
    feature_rows = []

    print("Calculating rolling 365-day synergy (Optimized O(n))...")
    
    for i, row in df.iterrows():
        key = (row['trainer'], row['jockey'])
        curr_date = row['date']
        curr_win = row['is_win']
        curr_place = row['is_place']
        
        if key not in history:
            # First time for this combo
            feature_rows.append((row['id'], 0, 0, 0.0, 0.0))
            history[key] = {
                'q': deque([(curr_date, curr_win, curr_place)]),
                'w': curr_win,
                'p': curr_place
            }
        else:
            # 1. Slide window: remove entries older than 365 days
            cutoff = curr_date - pd.Timedelta(days=365)
            while history[key]['q'] and history[key]['q'][0][0] < cutoff:
                _, old_w, old_p = history[key]['q'].popleft()
                history[key]['w'] -= old_w
                history[key]['p'] -= old_p
            
            # 2. Calculate stats from history BEFORE today's result
            runs = len(history[key]['q'])
            if runs > 0:
                wins = history[key]['w']
                places = history[key]['p']
                win_pc = round(wins / runs, 3)
                place_pc = round(places / runs, 3)
                feature_rows.append((row['id'], runs, wins, win_pc, place_pc))
            else:
                feature_rows.append((row['id'], 0, 0, 0.0, 0.0))
            
            # 3. Add today's result to history for future races
            history[key]['q'].append((curr_date, curr_win, curr_place))
            history[key]['w'] += curr_win
            history[key]['p'] += curr_place

        if i % 200000 == 0 and i > 0:
            print(f"Processed {i:,} / {len(df):,} rows...")

    # 3. Bulk insert
    print("Saving features to database...")
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO features_synergy (res_id, combo_runs, combo_wins, combo_win_pc, combo_place_pc)
        VALUES (?, ?, ?, ?, ?)
    """, feature_rows)
    
    conn.commit()
    # Add an index to make joining fast later
    conn.execute("CREATE INDEX idx_synergy_res_id ON features_synergy(res_id)")
    conn.close()
    print("Done! Synergy features created successfully.")

if __name__ == "__main__":
    generate_synergy_features()