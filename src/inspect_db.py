import sqlite3
import os

def inspect_horse_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print(f"\n{'='*60}")
    print(f"DATABASE: {db_path}")
    print(f"Total Tables: {len(tables)}")
    print(f"{'='*60}\n")

    for table_tuple in tables:
        table_name = table_tuple[0]
        
        # Get row count for context
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        print(f"TABLE: {table_name} ({row_count:,} rows)")
        print(f"{'-' * (len(table_name) + 7)}")
        
        # 2. Get column info
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()
        
        for col in columns:
            # col structure: (id, name, type, notnull, default, pk)
            col_id = col[0]
            col_name = col[1]
            col_type = col[2]
            pk_flag = " [PK]" if col[5] == 1 else ""
            
            print(f"  {col_id:>2} | {col_name:20} | {col_type}{pk_flag}")
        print("\n")

    conn.close()

if __name__ == "__main__":
    # Using your absolute path
    DB_PATH = r"E:\horse_pred\database\horse_racing.db"
    inspect_horse_db(DB_PATH)