# test_database.py
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from app.knowledge_base.relational_kb.sqlite_manger import SQLiteManager

def test_database():
    db = SQLiteManager(":memory:")
    
    try:
        # Get table counts
        counts = db.get_table_counts()
        print("📊 DATABASE CONTENTS:")
        for table, count in counts.items():
            print(f"  {table}: {count} records")
        
        # Show some sample data
        print("\n🤖 ML MODELS:")
        models = db.execute_query("SELECT model_name, model_type FROM ML_Models")
        for model in models:
            print(f"  • {model['model_name']} ({model['model_type']})")
        
        print("\n📜 RULES:")
        rules = db.execute_query("SELECT name, priority FROM Rules ORDER BY priority")
        for rule in rules:
            print(f"  • {rule['name']} (Priority: {rule['priority']})")
            
    finally:
        db.close()

if __name__ == "__main__":
    test_database()