import sqlite3
import pandas as pd
import numpy as np
import os

def generate_micro_details():
    db_path = r"E:\horse_pred\database\horse_racing.db"
    conn = sqlite3.connect(db_path)
    
    print("Step 1: Extracting raw data (including race_id)...")
    # Added race_id here so we can group horses together
    query = """
    SELECT 
        id, race_id, horse, date, hg, official_rating, jockey_allowance, wgt_lbs, prize
    FROM historic_results
    ORDER BY horse, date
    """
    df = pd.read_sql(query, conn)
    print(f"Found {len(df):,} rows.")

    print("Step 2: Calculating micro-signals (fixing leakage)...")
    
    # Clean prize strings
    if df['prize'].dtype == 'object':
        df['prize_clean'] = df['prize'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['prize_clean'] = pd.to_numeric(df['prize_clean'], errors='coerce').fillna(0)
    else:
        df['prize_clean'] = df['prize'].fillna(0)

    # ANTI-LEAK: Calculate Total Race Value (Max prize seen in that race_id)
    # This ensures the winner and the 10th place horse have the SAME log_prize
    df['race_value'] = df.groupby('race_id')['prize_clean'].transform('max')
    df['log_prize'] = np.log1p(df['race_value'].astype(float))

    # 1. Gear Change Detail
    df['prev_hg'] = df.groupby('horse')['hg'].shift(1).fillna('')
    df['hg'] = df['hg'].fillna('')
    df['gear_change'] = ((df['hg'] != df['prev_hg']) & (df['hg'] != '')).astype(int)
    
    # 2. Handicap 'Snip' 
    df['prev_or'] = df.groupby('horse')['official_rating'].shift(1)
    df['or_change'] = (df['official_rating'] - df['prev_or']).fillna(0)
    
    # 3. True Weight
    df['true_wgt'] = df['wgt_lbs'] - df['jockey_allowance'].fillna(0)

    print("Step 3: Writing to 'features_micro' table...")
    output_df = df[['id', 'gear_change', 'or_change', 'true_wgt', 'log_prize']]
    output_df.to_sql('features_micro', conn, index=False, if_exists='replace')
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_micro_id ON features_micro(id)")
    conn.commit()
    
    # Quick Check: Ensure the first 5 rows of a single race have the SAME log_prize
    print("\n--- ANTI-LEAK CHECK ---")
    sample_race = df[df['race_id'] == df['race_id'].iloc[0]][['horse', 'log_prize']]
    print(sample_race.head())

    conn.close()
    print("\nDone! Now your model has to work for its wins.")

if __name__ == "__main__":
    generate_micro_details()