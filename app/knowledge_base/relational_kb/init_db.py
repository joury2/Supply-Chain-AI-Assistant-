# app/knowledge_base/relational_kb/init_db.py
import sqlite3
import os
from pathlib import Path

def init_database(db_path: str = "supply_chain.db", sql_file: str = None):
    """
    Initialize the database by executing SQL commands from a file.
    """
    if sql_file is None:
        # Default to the SQL file in the same directory
        sql_file = Path(__file__).parent / "init_kb.sql"
    
    print(f"📂 Initializing database from: {sql_file}")
    print(f"📁 Database location: {db_path}")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    try:
        # Read and execute the SQL file
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        
        # Execute the entire SQL script
        conn.executescript(sql_script)
        conn.commit()
        
        print("✅ Database initialized successfully!")
        
        # Verify the tables were created
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        
        tables = cursor.fetchall()
        print(f"📋 Created {len(tables)} tables:")
        for table in tables:
            print(f"  • {table['name']}")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def check_database_status(db_path: str = "supply_chain.db"):
    """Check the current status of the database"""
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Get table counts
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        
        tables = cursor.fetchall()
        print(f"📊 Database: {db_path}")
        print(f"📋 Found {len(tables)} tables:")
        
        for table in tables:
            table_name = table['name']
            count_cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count = count_cursor.fetchone()['count']
            print(f"  • {table_name}: {count} records")
            
    finally:
        conn.close()

if __name__ == "__main__":
    # Initialize the database
    init_database("supply_chain.db")
    
    print("\n" + "="*50)
    
    # Check the status
    check_database_status("supply_chain.db")