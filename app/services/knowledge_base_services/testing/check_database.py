# app/services/knowledge_base_services/check_database.py
import os
import sys
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def check_database_contents():
    """Check what's actually in the database"""
    db_path = "supply_chain.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔍 DATABASE CONTENTS")
    print("=" * 50)
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in database: {[table[0] for table in tables]}")
    
    # Check ML_Models table
    print("\n📊 ML_MODELS TABLE:")
    cursor.execute("SELECT * FROM ML_Models")
    models = cursor.fetchall()
    print(f"Number of models: {len(models)}")
    
    if models:
        # Get column names
        cursor.execute("PRAGMA table_info(ML_Models)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Columns: {columns}")
        
        for model in models:
            print(f"\n  Model: {model[1]}")  # model_name
            print(f"    Type: {model[2]}")   # model_type
            print(f"    Target: {model[3]}") # target_variable
            print(f"    Required Features: {model[4]}") # required_features
            print(f"    Is Active: {model[7]}") # is_active
    
    # Check Dataset_Schemas table
    print("\n📋 DATASET_SCHEMAS TABLE:")
    cursor.execute("SELECT * FROM Dataset_Schemas")
    schemas = cursor.fetchall()
    print(f"Number of schemas: {len(schemas)}")
    
    for schema in schemas:
        print(f"  Schema: {schema[1]}")  # dataset_name
        print(f"    Description: {schema[2]}") # description
    
    # Check Column_Definitions table
    print("\n🗂️ COLUMN_DEFINITIONS TABLE:")
    cursor.execute("SELECT * FROM Column_Definitions")
    columns = cursor.fetchall()
    print(f"Number of column definitions: {len(columns)}")
    
    for col in columns[:10]:  # Show first 10
        print(f"  Column: {col[2]} (Dataset ID: {col[1]})")  # column_name, dataset_id
        print(f"    Data Type: {col[3]}, Required: {col[5]}") # data_type, requirement_level
    
    # Check Rules table
    print("\n📜 RULES TABLE:")
    cursor.execute("SELECT * FROM Rules")
    rules = cursor.fetchall()
    print(f"Number of rules: {len(rules)}")
    
    for rule in rules:
        print(f"  Rule: {rule[1]}")  # name
        print(f"    Priority: {rule[3]}, Active: {rule[6]}") # priority, is_active
    
    conn.close()
    print("\n🎉 DATABASE CHECK COMPLETED")

if __name__ == "__main__":
    check_database_contents()