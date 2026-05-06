import sqlite3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def cleanup():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Cleaning up logical errors...")

    # 1. Close-finish correction (Positions 2 and 3)
    # If they are 2nd or 3rd with 0 btn, give them 0.01 (Nose/Dead Heat)
    cursor.execute("""
        UPDATE historic_results 
        SET btn = 0.01 
        WHERE pos IN (2, 3) AND btn = 0
    """)
    print(f"Adjusted {cursor.rowcount} close-finishers (Pos 2-3) to 0.01 lengths.")

    # 2. Missing data correction (Positions 4+)
    # If they are 4th or worse with 0 btn, they likely trailed off. 
    # We set them to 20 lengths as a conservative 'beaten' estimate.
    cursor.execute("""
        UPDATE historic_results 
        SET btn = 20.0 
        WHERE pos > 3 AND btn = 0
    """)
    print(f"Adjusted {cursor.rowcount} trailing horses (Pos 4+) to 20.0 lengths.")

    conn.commit()
    conn.close()
    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup()