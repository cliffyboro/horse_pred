import sqlite3
import pandas as pd
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def run_audit():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    print("--- HORSE RACING DATA AUDIT ---\n")

    # 1. Basic Counts
    total_rows = pd.read_sql("SELECT COUNT(*) FROM historic_results", conn).iloc[0,0]
    unique_races = pd.read_sql("SELECT COUNT(DISTINCT race_id) FROM historic_results", conn).iloc[0,0]
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Unique Races: {unique_races:,}")
    print("-" * 30)

    # 2. Winner Check (Critical for Target Variable)
    # Group by race_id and count how many 'pos == 1' exists per race
    winners_per_race = pd.read_sql("""
        SELECT race_id, COUNT(*) as win_count 
        FROM historic_results 
        WHERE pos = 1 
        GROUP BY race_id
    """, conn)

    no_winners = unique_races - len(winners_per_race)
    multi_winners = len(winners_per_race[winners_per_race['win_count'] > 1])
    
    print(f"Races with NO winner recorded: {no_winners:,}")
    print(f"Races with multiple winners (Dead Heats?): {multi_winners:,}")
    
    # 3. Target Data Integrity (pos and btn)
    null_pos = pd.read_sql("SELECT COUNT(*) FROM historic_results WHERE pos IS NULL", conn).iloc[0,0]
    null_btn = pd.read_sql("SELECT COUNT(*) FROM historic_results WHERE btn IS NULL AND pos > 1", conn).iloc[0,0]
    null_wgt = pd.read_sql("SELECT COUNT(*) FROM historic_results WHERE wgt_lbs IS NULL", conn).iloc[0,0]
    
    print(f"Rows missing Position: {null_pos:,}")
    print(f"Non-winners missing Beaten Distance (btn): {null_btn:,}")
    print(f"Rows missing Weight (lbs): {null_wgt:,}")
    print("-" * 30)

    # 4. Temporal Distribution
    print("Races per Year:")
    yearly_dist = pd.read_sql("""
        SELECT strftime('%Y', date) as year, COUNT(DISTINCT race_id) as race_count
        FROM historic_results 
        GROUP BY year 
        ORDER BY year
    """, conn)
    print(yearly_dist.to_string(index=False))
    print("-" * 30)

    # 5. Non-Winner btn=0 check (Logical error)
    logical_errors = pd.read_sql("""
        SELECT COUNT(*) FROM historic_results 
        WHERE pos > 1 AND btn = 0
    """, conn).iloc[0,0]
    print(f"Logical Errors (Finished >1st but btn is 0): {logical_errors:,}")

    conn.close()

if __name__ == "__main__":
    run_audit()