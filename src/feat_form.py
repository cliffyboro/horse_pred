import sqlite3
import pandas as pd
import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def generate_form_features():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Create the Features Table
    conn.execute("DROP TABLE IF EXISTS features_form")
    conn.execute('''
        CREATE TABLE features_form (
            res_id INTEGER PRIMARY KEY,
            avg_btn_3 INTEGER,
            avg_btn_5 INTEGER,
            days_since_last INTEGER,
            last_pos INTEGER,
            FOREIGN KEY(res_id) REFERENCES historic_results(id)
        )
    ''')

    print("Loading data for Form calculation...")
    df = pd.read_sql("""
        SELECT id, date, horse, pos, btn 
        FROM historic_results 
        ORDER BY date ASC
    """, conn)
    
    df['date'] = pd.to_datetime(df['date'])
    # Ensure btn is numeric for math
    df['btn'] = pd.to_numeric(df['btn'], errors='coerce').fillna(20.0)
    # Ensure pos is numeric for 'last_pos' feature
    df['pos_num'] = pd.to_numeric(df['pos'], errors='coerce').fillna(20.0)

    # Dictionary: { horse_name: deque of (btn, date, pos) }
    history = {}
    feature_rows = []

    print("Calculating Rolling Form and Recency...")
    
    for i, row in df.iterrows():
        h = row['horse']
        curr_date = row['date']
        
        if h not in history:
            # Debut runner - no form
            feature_rows.append((row['id'], None, None, None, None))
            history[h] = deque([(row['btn'], curr_date, row['pos_num'])], maxlen=5)
        else:
            # 1. Calculate stats BEFORE adding today's result
            recent = list(history[h])
            last_run_date = recent[-1][1]
            last_pos = recent[-1][2]
            
            days_since = (curr_date - last_run_date).days
            
            # Avg btn last 3
            btns_3 = [x[0] for x in recent[-3:]]
            avg_3 = round(sum(btns_3) / len(btns_3), 2)
            
            # Avg btn last 5
            btns_5 = [x[0] for x in recent]
            avg_5 = round(sum(btns_5) / len(btns_5), 2)
            
            feature_rows.append((row['id'], avg_3, avg_5, days_since, last_pos))
            
            # 2. Add today's result to history
            history[h].append((row['btn'], curr_date, row['pos_num']))

        if i % 200000 == 0 and i > 0:
            print(f"Processed {i:,} / {len(df):,} rows...")

    print("Saving features to database...")
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO features_form (res_id, avg_btn_3, avg_btn_5, days_since_last, last_pos)
        VALUES (?, ?, ?, ?, ?)
    """, feature_rows)
    
    conn.commit()
    conn.execute("CREATE INDEX idx_form_res_id ON features_form(res_id)")
    conn.close()
    print("Done! Form features created.")

if __name__ == "__main__":
    generate_form_features()