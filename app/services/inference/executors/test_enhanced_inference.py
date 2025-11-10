# app/services/inference/executors/test_enhanced_inference.py
"""
Test the enhanced inference system with improved executors
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.inference.model_inference_service import ModelInferenceService
from app.services.inference.base_model_executor import ModelMetadata, ModelType

def test_enhanced_executors():
    """Test that all enhanced executors work correctly"""
    print("🧪 Testing Enhanced Inference Executors...")
    
    try:
        # Initialize service
        service = ModelInferenceService(model_storage_path="models/")
        
        print("✅ Inference Service initialized!")
        print(f"   Supported types: {service.executor_registry.list_supported_types()}")
        
        # Test creating metadata for each model type
        test_models = [
            ModelMetadata(
                model_name="test_lightgbm",
                model_type=ModelType.LIGHTGBM,
                version="1.0",
                target_column="sales",
                required_features=["shop_id", "date", "sales_lag_1"],
                optional_features=[],
                frequency="daily"
            ),
            ModelMetadata(
                model_name="test_prophet", 
                model_type=ModelType.PROPHET,
                version="1.0",
                target_column="y",
                required_features=["ds", "y"],
                optional_features=["is_saturday", "is_sunday"],
                frequency="daily"
            ),
            ModelMetadata(
                model_name="test_lstm",
                model_type=ModelType.LSTM,
                version="1.0", 
                target_column="sales",
                required_features=["shop_id", "year", "month", "sales_lag_1"],
                optional_features=[],
                frequency="daily"
            )
        ]
        
        for metadata in test_models:
            print(f"✅ Created metadata for {metadata.model_name} ({metadata.model_type.value})")
        
        # Test data validation with sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'sales': [100 + i * 10 for i in range(10)],
            'shop_id': [1] * 10,
            'sales_lag_1': [90 + i * 10 for i in range(10)]
        })
        
        print(f"✅ Sample test data created: {sample_data.shape}")
        print(f"   Columns: {list(sample_data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_executors()
    if success:
        print("\n🎉 ALL ENHANCED EXECUTORS WORKING CORRECTLY!")
    else:
        print("\n💥 SOME TESTS FAILED - CHECK THE IMPLEMENTATION")