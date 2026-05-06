import sqlite3
import pandas as pd
import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def generate_course_features():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Create the Features Table
    conn.execute("DROP TABLE IF EXISTS features_course")
    conn.execute('''
        CREATE TABLE features_course (
            res_id INTEGER PRIMARY KEY,
            h_course_runs INTEGER,
            h_course_wins INTEGER,
            h_course_place_pc REAL,
            FOREIGN KEY(res_id) REFERENCES historic_results(id)
        )
    ''')

    print("Loading data for course suitability calculation...")
    df = pd.read_sql("""
        SELECT id, date, horse, course, pos 
        FROM historic_results 
        ORDER BY date ASC
    """, conn)
    
    df['date'] = pd.to_datetime(df['date'])
    pos_numeric = pd.to_numeric(df['pos'], errors='coerce')
    df['is_win'] = (pos_numeric == 1).astype(int)
    df['is_place'] = (pos_numeric <= 3).astype(int)

    # Dictionary: { (horse, course): {'w': total_wins, 'p': total_places, 'r': total_runs} }
    # Note: We don't need a rolling window here because course suitability 
    # is usually a lifetime trait for a horse.
    history = {}
    feature_rows = []

    print("Calculating Horse-Course Specialist metrics...")
    
    for i, row in df.iterrows():
        key = (row['horse'], row['course'])
        
        if key not in history:
            # First time horse has seen this track
            feature_rows.append((row['id'], 0, 0, 0.0))
            history[key] = {'w': row['is_win'], 'p': row['is_place'], 'r': 1}
        else:
            # Stats BEFORE today
            runs = history[key]['r']
            wins = history[key]['w']
            places = history[key]['p']
            place_pc = round(places / runs, 3)
            
            feature_rows.append((row['id'], runs, wins, place_pc))
            
            # Update history with today's result for future races
            history[key]['w'] += row['is_win']
            history[key]['p'] += row['is_place']
            history[key]['r'] += 1

        if i % 200000 == 0 and i > 0:
            print(f"Processed {i:,} / {len(df):,} rows...")

    print("Saving features to database...")
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO features_course (res_id, h_course_runs, h_course_wins, h_course_place_pc)
        VALUES (?, ?, ?, ?)
    """, feature_rows)
    
    conn.commit()
    conn.execute("CREATE INDEX idx_course_res_id ON features_course(res_id)")
    conn.close()
    print("Done! Course Specialist features created.")

if __name__ == "__main__":
    generate_course_features()