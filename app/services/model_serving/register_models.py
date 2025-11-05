# app/services/model_serving/register_models.py
"""
Simple script to discover and register all models to the database
Run this once to populate your database with model metadata
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.model_serving.smart_model_registry import discover_and_register_models

if __name__ == "__main__":
    print("🚀 Starting Model Registration...")
    print("="*70)
    
    # Configuration
    MODELS_DIR = "models/forecasting"
    DB_PATH = "supply_chain.db"
    
    print(f"📁 Models Directory: {MODELS_DIR}")
    print(f"💾 Database: {DB_PATH}")
    print()
    
    # Run discovery and registration
    try:
        discover_and_register_models(
            models_dir=MODELS_DIR,
            db_path=DB_PATH
        )
        
        print("\n✅ SUCCESS!")
        print("Your models are now registered and ready to use.")
        print()
        print("Next steps:")
        print("1. Start FastAPI: uvicorn app.api.main:app --reload")
        print("2. Start Streamlit: streamlit run streamlit_app/pages/ForecastWithAPI.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)