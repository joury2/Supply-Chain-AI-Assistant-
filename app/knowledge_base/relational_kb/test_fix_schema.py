# app/knowledge_base/relational_kb/test_fix_schema.py
from app.knowledge_base.relational_kb.sqlite_manager import SQLiteManager
from app.knowledge_base.relational_kb.sqlite_schema import SQLiteRepository

def test_fixed_repository():
    print("🧪 Testing Fixed Repository...")
    
    # Initialize with existing database (keeps your data)
    db_manager = SQLiteManager("supply_chain.db")
    repository = SQLiteRepository(db_manager)
    
    try:
        # Test 1: Basic connection
        print("1. Testing connection...")
        if repository.test_connection():
            print("   ✅ Connection successful")
        else:
            print("   ❌ Connection failed")
            return
        
        # Test 2: Get active models (should work now)
        print("2. Testing model retrieval...")
        models = repository.get_active_models()
        print(f"   ✅ Found {len(models)} active models")
        
        # Test 3: Get dataset schemas (should work now)  
        print("3. Testing schema retrieval...")
        schemas = repository.get_all_schemas()
        print(f"   ✅ Found {len(schemas)} dataset schemas")
        
        # Test 4: Get rules (should work now)
        print("4. Testing rules retrieval...")
        rules = repository.get_active_rules()
        print(f"   ✅ Found {len(rules)} active rules")
        
        # Test 5: Test missing method (now fixed)
        print("5. Testing get_model_by_id...")
        if models:
            model = repository.get_model_by_id(models[0]['model_id'])
            print(f"   ✅ Model retrieval: {model['model_name']}")
        
        print("\n🎉 ALL TESTS PASSED! Repository is fixed.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    test_fixed_repository()