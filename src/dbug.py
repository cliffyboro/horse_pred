import sqlite3
import pandas as pd

conn = sqlite3.connect(r"E:\horse_pred\database\horse_racing.db")
# Check one random race
query = """
SELECT horse, pos, prize, official_rating 
FROM historic_results 
WHERE race_id = (SELECT race_id FROM historic_results WHERE prize > 0 LIMIT 1)
"""
print(pd.read_sql(query, conn))
conn.close()