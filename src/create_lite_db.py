import sqlite3
import os
from pathlib import Path

def create_lite_db(months=6):
    # 1. SETUP PATHS
    base_dir = Path(r"e:\horse_pred")
    full_db_path = base_dir / "database" / "horse_racing.db"
    lite_db_path = base_dir / "database" / "horse_racing_lite.db"

    if not full_db_path.exists():
        print(f"❌ Error: Full database not found at {full_db_path}")
        return

    print(f"🚀 Starting Deep-Lite DB creation (Last {months} months)...")
    conn_full = sqlite3.connect(full_db_path)
    conn_lite = sqlite3.connect(lite_db_path)
    
    # 2. GET TABLES
    tables = conn_full.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        if table_name.startswith('sqlite_'): continue
            
        print(f"📦 Processing: {table_name}...")
        
        # Create schema in Lite DB
        schema = conn_full.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}';").fetchone()[0]
        conn_lite.execute(f"DROP TABLE IF EXISTS \"{table_name}\"")
        conn_lite.execute(schema)
        
        # 3. SELECT LOGIC BASED ON TABLE STRUCTURE
        try:
            if table_name == 'historic_results':
                # Direct date filter for the main results table
                query = f"SELECT * FROM \"{table_name}\" WHERE date >= date('now', '-{months} months')"
            
            elif table_name == 'features_micro':
                # Prune features_micro using 'id'
                query = f"""
                    SELECT t.* FROM \"{table_name}\" t
                    JOIN historic_results h ON t.id = h.id
                    WHERE h.date >= date('now', '-{months} months')
                """
            
            elif table_name.startswith('features_'):
                # Prune all other feature tables using 'res_id'
                query = f"""
                    SELECT t.* FROM \"{table_name}\" t
                    JOIN historic_results h ON t.res_id = h.id
                    WHERE h.date >= date('now', '-{months} months')
                """
            else:
                # Default for any other tables: Copy everything
                query = f"SELECT * FROM \"{table_name}\""

            cursor = conn_full.execute(query)
            rows = cursor.fetchall()
            
            if rows:
                placeholders = ",".join(["?"] * len(cursor.description))
                conn_lite.executemany(f"INSERT INTO \"{table_name}\" VALUES ({placeholders})", rows)
                print(f"   ✅ Copied {len(rows)} rows.")
            else:
                print(f"   ⚠️ No rows found for the selected period.")

        except sqlite3.OperationalError as e:
            print(f"   ❌ Error processing {table_name}: {e}")

    # 4. FINALIZATION
    print("\nFinalizing...")
    conn_lite.commit()
    conn_lite.execute("VACUUM") # Critical for shrinking the file
    conn_lite.close()
    conn_full.close()
    
    size_mb = os.path.getsize(lite_db_path) / (1024 * 1024)
    print(f"✅ Success! Lite DB created at: {lite_db_path}")
    print(f"📊 Final Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    create_lite_db(months=6)