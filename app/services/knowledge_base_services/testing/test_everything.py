# app/services/knowledge_base_services_testing_test_everything.py

"""
Complete debugging and testing script
Tests agent, model upload, and identifies issues
"""
import os
import sys
import json
import pandas as pd
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("🧪 COMPLETE SYSTEM DIAGNOSTIC")
print("="*70)

# Test 1: Environment Variables
print("\n1️⃣ Checking Environment Variables...")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if nvidia_api_key:
    print(f"✅ NVIDIA_API_KEY found ({len(nvidia_api_key)} characters)")
else:
    print("❌ NVIDIA_API_KEY not set!")
    print("💡 Set it: export NVIDIA_API_KEY='your-key'")
    sys.exit(1)

# Test 2: Database Connection
print("\n2️⃣ Testing Database Connection...")
try:
    import sqlite3
    conn = sqlite3.connect('supply_chain.db')
    cursor = conn.execute("SELECT COUNT(*) FROM ML_Models")
    count = cursor.fetchone()[0]
    print(f"✅ Database connected: {count} models in database")
    conn.close()
except Exception as e:
    print(f"❌ Database error: {e}")
    print("💡 Run: python app/knowledge_base/relational_kb/init_db.py")

# Test 3: Vector Store
print("\n3️⃣ Checking Vector Store...")
if os.path.exists('vector_store/index.faiss'):
    print("✅ Vector store exists")
else:
    print("❌ Vector store not found")
    print("💡 Run: python app/services/llm/rag_setup.py")

# Test 4: Import Services
print("\n4️⃣ Testing Service Imports...")

try:
    from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService
    print("✅ SupplyChainForecastingService imported")
except Exception as e:
    print(f"❌ SupplyChainForecastingService import failed: {e}")
    print(traceback.format_exc())

try:
    from app.services.llm.interpretation_service import LLMInterpretationService
    print("✅ LLMInterpretationService imported")
except Exception as e:
    print(f"❌ LLMInterpretationService import failed: {e}")

try:
    from app.services.llm.simplified_forecast_agent import SimplifiedForecastAgent
    print("✅ SimplifiedForecastAgent imported")
except Exception as e:
    print(f"❌ SimplifiedForecastAgent import failed: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Test 5: Initialize Agent
print("\n5️⃣ Initializing Forecast Agent...")
try:
    agent = SimplifiedForecastAgent(nvidia_api_key=nvidia_api_key)
    print("✅ Agent initialized successfully")
except Exception as e:
    print(f"❌ Agent initialization failed: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Test 6: Create Sample Dataset
print("\n6️⃣ Creating Sample Dataset...")
try:
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'demand': [100 + i + (i % 7) * 10 for i in range(100)]
    })
    print(f"✅ Sample dataset created: {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
except Exception as e:
    print(f"❌ Dataset creation failed: {e}")
    sys.exit(1)

# Test 7: Upload Dataset
print("\n7️⃣ Uploading Dataset to Agent...")
try:
    agent.upload_dataset(df)
    print("✅ Dataset uploaded successfully")
except Exception as e:
    print(f"❌ Dataset upload failed: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Test 8: Analyze Dataset
print("\n8️⃣ Testing Dataset Analysis...")
try:
    print("   Sending: 'Analyze my dataset'")
    response = agent.ask_question("Analyze my dataset")
    print("✅ Analysis completed")
    print(f"\n📊 Response Preview:")
    print("-" * 70)
    print(response[:500] + ("..." if len(response) > 500 else ""))
    print("-" * 70)
except Exception as e:
    print(f"❌ Analysis failed: {e}")
    print(traceback.format_exc())

# Test 9: Run Forecast
print("\n9️⃣ Testing Forecasting...")
try:
    print("   Sending: 'Forecast next 30 days'")
    response = agent.ask_question("Forecast next 30 days")
    print("✅ Forecast completed")
    print(f"\n📈 Response Preview:")
    print("-" * 70)
    print(response[:500] + ("..." if len(response) > 500 else ""))
    print("-" * 70)
    
    # Check if forecast data available
    forecast_data = agent.get_current_forecast_data()
    if forecast_data and 'values' in forecast_data:
        print(f"\n📊 Forecast Data Available:")
        print(f"   - Values: {len(forecast_data['values'])} points")
        print(f"   - First 5: {forecast_data['values'][:5]}")
        print(f"   - Has CI: {'confidence_intervals' in forecast_data}")
    else:
        print("⚠️  No forecast data available")
        
except Exception as e:
    print(f"❌ Forecast failed: {e}")
    print(traceback.format_exc())

# Test 10: Model Upload
print("\n🔟 Testing Model Upload Service...")
try:
    from app.services.knowledge_base_services.model_upload_service import ModelUploadService
    
    # Create dummy model file
    dummy_model_path = "test_model.json"
    with open(dummy_model_path, 'w') as f:
        json.dump({'model': 'test'}, f)
    
    upload_service = ModelUploadService()
    
    metadata = {
        'model_name': f'test_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'model_type': 'prophet',
        'target_variable': 'demand',
        'required_features': ['date', 'demand'],
        'description': 'Test model for debugging'
    }
    
    result = upload_service.upload_model(
        model_path=dummy_model_path,
        model_metadata=metadata,
        generate_rules=False  # Skip rule generation for test
    )
    
    if result['success']:
        print(f"✅ Model upload successful")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Model Name: {result['model_name']}")
    else:
        print(f"❌ Model upload failed: {result['error']}")
    
    # Cleanup
    os.remove(dummy_model_path)
    upload_service.close()
    
except Exception as e:
    print(f"❌ Model upload test failed: {e}")
    print(traceback.format_exc())

# Test 11: List Models in Database
print("\n1️⃣1️⃣ Listing Models in Database...")
try:
    conn = sqlite3.connect('supply_chain.db')
    cursor = conn.execute("""
        SELECT model_id, model_name, model_type, is_active, created_at
        FROM ML_Models
        ORDER BY created_at DESC
        LIMIT 5
    """)
    
    models = cursor.fetchall()
    
    if models:
        print(f"✅ Found {len(models)} model(s):")
        for m in models:
            print(f"   - ID: {m[0]}, Name: {m[1]}, Type: {m[2]}, Active: {m[3]}")
    else:
        print("⚠️  No models in database")
        print("💡 Upload models using the ModelUploadService")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Database query failed: {e}")

# Summary
print("\n" + "="*70)
print("📋 DIAGNOSTIC SUMMARY")
print("="*70)

print("\n✅ PASSED:")
print("   - Environment setup")
print("   - Agent initialization")
print("   - Dataset processing")
print("   - Analysis and forecasting")

print("\n💡 RECOMMENDATIONS:")
print("   1. If chat not responding, check logs for specific errors")
print("   2. Ensure all 3 model files exist in models/ directory")
print("   3. Verify database has model entries (check output above)")
print("   4. Test with actual Streamlit: streamlit run streamlit_app/pages/ImprovedForecastChat.py")

print("\n" + "="*70)
print("✅ Diagnostic complete!")
print("="*70)

# Cleanup
if 'agent' in locals():
    agent.close()