# app/services/knowledge_base_services/debugging/model_registry_check.py
import os
import json
from pathlib import Path

def check_model_registry():
    """Check if models are properly registered"""
    print("🔍 Checking Model Registry...")
    
    # Check models directory
    models_dir = Path("models/forecasting")
    print(f"📁 Models directory: {models_dir.absolute()}")
    print(f"   Exists: {models_dir.exists()}")
    
    if models_dir.exists():
        # List all files in models directory
        print("   Files in models directory:")
        for file_path in models_dir.iterdir():
            print(f"   - {file_path.name} (Size: {file_path.stat().st_size} bytes)")
    
    # Check for available_models.json
    models_file = models_dir / "available_models.json"
    print(f"📄 Models config file: {models_file}")
    print(f"   Exists: {models_file.exists()}")
    
    if models_file.exists():
        try:
            with open(models_file, 'r') as f:
                models_config = json.load(f)
            print(f"   ✅ Loaded successfully")
            print(f"   Number of models: {len(models_config)}")
            
            for model_id, model_info in models_config.items():
                print(f"   🎯 Model: {model_id}")
                print(f"      Name: {model_info.get('model_name', 'N/A')}")
                print(f"      Type: {model_info.get('model_type', 'N/A')}")
                print(f"      Status: {model_info.get('status', 'N/A')}")
                print(f"      Min data: {model_info.get('min_data_points', 'N/A')}")
                
        except Exception as e:
            print(f"   ❌ Error loading models file: {e}")
    else:
        print("   ❌ Models config file not found!")
    
    # Check model registry service
    print("\n🔧 Checking Model Registry Service...")
    try:
        from app.services.model_serving.model_registry_service import ModelRegistryService
        registry = ModelRegistryService()
        print(f"   ✅ ModelRegistryService imported successfully")
        
        # Try to get active models
        active_models = registry.get_active_models()
        print(f"   Active models from service: {len(active_models)}")
        for model in active_models:
            print(f"   - {model.get('model_name', 'Unknown')}")
            
    except Exception as e:
        print(f"   ❌ ModelRegistryService error: {e}")
    
    # Check if models directory exists in the right location
    print(f"\n📂 Current working directory: {Path.cwd()}")
    print(f"📂 Project root models: {Path('models/forecasting').absolute()}")

if __name__ == "__main__":
    check_model_registry()