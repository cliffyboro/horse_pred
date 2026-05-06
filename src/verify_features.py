import sqlite3
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def verify():
    conn = sqlite3.connect(DB_PATH)
    
    # We'll pick a horse that appears frequently to see the 'evolution' of its features
    # First, find a horse with at least 10 runs
    target_horse = pd.read_sql("""
        SELECT horse, COUNT(*) as runs 
        FROM historic_results 
        GROUP BY horse 
        HAVING runs > 15 
        LIMIT 1
    """, conn).iloc[0,0]
    
    print(f"--- VERIFYING FEATURES FOR HORSE: {target_horse} ---")

    query = f"""
        SELECT 
            r.date, r.course, r.pos, r.btn,
            s.combo_win_pc as synergy_win_pc,
            c.h_course_runs,
            f.avg_btn_3,
            f.days_since_last
        FROM historic_results r
        LEFT JOIN features_synergy s ON r.id = s.res_id
        LEFT JOIN features_course  c ON r.id = c.res_id
        LEFT JOIN features_form    f ON r.id = f.res_id
        WHERE r.horse = ?
        ORDER BY r.date ASC
    """
    
    df = pd.read_sql(query, conn, params=(target_horse,))
    print(df.to_string(index=False))
    
    # Check for 'The Black Hole' (Are any features 100% Null?)
    print("\n--- GLOBAL NULL CHECK ---")
    null_check = pd.read_sql("""
        SELECT 
            (SELECT COUNT(*) FROM features_synergy WHERE combo_win_pc IS NULL) as null_synergy,
            (SELECT COUNT(*) FROM features_course WHERE h_course_runs IS NULL) as null_course,
            (SELECT COUNT(*) FROM features_form WHERE avg_btn_3 IS NULL) as null_form
    """, conn)
    print(null_check)
    
    conn.close()

if __name__ == "__main__":
    verify()