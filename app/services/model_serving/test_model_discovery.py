# test_model_discovery.py
import json
import os
import sys

def test_model_files():
    """Simple test to check if model files exist"""
    print("🧪 TESTING MODEL FILES")
    print("=" * 40)
    
    base_path = "models/forecasting/lightgbm"
    
    if not os.path.exists(base_path):
        print(f"❌ Base path not found: {base_path}")
        print("Current directory:", os.getcwd())
        return
    
    print(f"✅ Base path exists: {base_path}")
    
    # Check each model directory
    models = ["monthly_sales_forecaster", "daily_sales_forecaster"]
    
    for model in models:
        model_path = os.path.join(base_path, model)
        print(f"\n🔍 Checking: {model}")
        
        if not os.path.exists(model_path):
            print(f"   ❌ Directory not found: {model_path}")
            continue
            
        print(f"   ✅ Directory exists")
        
        # Check required files
        files_to_check = [
            "rag_metadata.json",
            "feature_schema.json", 
            "performance_metrics.json",
            "model.pkl"
        ]
        
        for file in files_to_check:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
                
                # Try to load JSON files
                if file.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        print(f"     📊 Loaded successfully")
                    except Exception as e:
                        print(f"     ❌ Error loading: {e}")
            else:
                print(f"   ❌ Missing: {file}")

if __name__ == "__main__":
    test_model_files()