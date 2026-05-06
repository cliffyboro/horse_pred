import sqlite3
import pandas as pd
import os
import sys
from collections import deque

# Adjust for your 'database' folder structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def generate_class_features():
    conn = sqlite3.connect(DB_PATH)
    
    conn.execute("DROP TABLE IF EXISTS features_class")
    conn.execute('''
        CREATE TABLE features_class (
            res_id INTEGER PRIMARY KEY,
            class_diff INTEGER,
            avg_class_3 REAL,
            FOREIGN KEY(res_id) REFERENCES historic_results(id)
        )
    ''')

    print("Loading data for Class Move calculation...")
    # Wrap "class" in double quotes because it's a reserved keyword
    df = pd.read_sql("""
        SELECT id, date, horse, "class" as race_class 
        FROM historic_results 
        ORDER BY date ASC
    """, conn)
    
    # Extract numeric class (e.g., "Class 4" -> 4)
    # If your data is just the number, to_numeric handles it. 
    # If it's "Class 4", we extract the digit.
    df['race_class'] = df['race_class'].str.extract('(\d+)').astype(float).fillna(0)

    history = {}
    feature_rows = []

    print("Calculating Class Transitions...")
    
    for i, row in df.iterrows():
        h = row['horse']
        curr_class = row['race_class']
        
        if h not in history or curr_class == 0:
            feature_rows.append((row['id'], 0, 0.0))
            history[h] = deque([curr_class], maxlen=3)
        else:
            recent_classes = [x for x in history[h] if x > 0]
            
            if len(recent_classes) > 0:
                last_class = recent_classes[-1]
                # Negative = Drop in class (Good)
                diff = int(curr_class - last_class)
                avg_3 = round(sum(recent_classes) / len(recent_classes), 2)
                feature_rows.append((row['id'], diff, avg_3))
            else:
                feature_rows.append((row['id'], 0, 0.0))
            
            history[h].append(curr_class)

        if i % 200000 == 0 and i > 0:
            print(f"Processed {i:,} / {len(df):,} rows...")

    print("Saving features to database...")
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO features_class VALUES (?, ?, ?)", feature_rows)
    
    conn.commit()
    conn.execute("CREATE INDEX idx_class_res_id ON features_class(res_id)")
    conn.close()
    print("Done! Class features created.")

if __name__ == "__main__":
    generate_class_features()