import sqlite3
import pandas as pd
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def inspect():
    conn = sqlite3.connect(DB_PATH)
    
    # Check a sample of these logical errors
    query = """
        SELECT date, course, race_name, pos, btn, horse, comment 
        FROM historic_results 
        WHERE pos > 1 AND btn = 0 
        LIMIT 20
    """
    df = pd.read_sql(query, conn)
    
    print("\n--- SAMPLE OF LOGICAL ERRORS (POS > 1, BTN = 0) ---")
    if df.empty:
        print("No errors found.")
    else:
        # Using to_string to ensure we see the full comment
        print(df.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    inspect()