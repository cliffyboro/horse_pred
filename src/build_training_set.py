import sqlite3
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def build_matrix():
    conn = sqlite3.connect(DB_PATH)
    print("Building matrix for diverse ensemble...")

    query = """
        SELECT 
            r.id, r.race_id, r.date, r.course, r.dist, r.going, r."class",
            r.wgt_lbs, r.draw, r.btn,
            s.combo_runs, s.combo_win_pc, s.combo_place_pc,
            c.h_course_runs, c.h_course_wins, c.h_course_place_pc,
            f.avg_btn_3, f.avg_btn_5, f.days_since_last, f.last_pos,
            cl.class_diff, cl.avg_class_3,
            m.gear_change, m.or_change, m.true_wgt, m.log_prize,
            CASE WHEN r.pos = 1 THEN 1 ELSE 0 END as target_win
        FROM historic_results r
        JOIN features_synergy s ON r.id = s.res_id
        JOIN features_course c ON r.id = c.res_id
        JOIN features_form f ON r.id = f.res_id
        JOIN features_class cl ON r.id = cl.res_id
        LEFT JOIN features_micro m ON r.id = m.id
        WHERE f.avg_btn_3 IS NOT NULL 
          AND (r.type LIKE '%Flat%' OR r.type LIKE '%AWT%')
    """
    
    df = pd.read_sql(query, conn)
    
    # Fill NaNs
    numeric_fix = ['gear_change', 'or_change', 'log_prize', 'true_wgt']
    df[numeric_fix] = df[numeric_fix].fillna(0)
    
    # Categoricals
    cat_cols = ['course', 'going', 'dist', 'class']
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    os.makedirs('data', exist_ok=True)
    df.to_csv("data/training_matrix.csv", index=False)
    
    print(f"✅ Matrix ready: {len(df):,} rows | {df['race_id'].nunique():,} races")
    print(f"Win rate: {df['target_win'].mean():.4f}")
    conn.close()

if __name__ == "__main__":
    build_matrix()