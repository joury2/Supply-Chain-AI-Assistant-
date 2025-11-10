#!/usr/bin/env python3
"""
FINAL Diagnostic - After fixes
"""

import os
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinalDiagnostic")

def test_fixes():
    """Test that the fixes work"""
    logger.info("🧪 TESTING FIXES")
    
    # Test 1: Check if os import is fixed
    try:
        from app.services.inference.model_inference_service import ModelInferenceService
        inference = ModelInferenceService(model_storage_path="models/")
        logger.info("✅ ModelInferenceService imports work")
    except NameError as e:
        if "'os' is not defined" in str(e):
            logger.error("❌ 'os' import still missing in ModelInferenceService")
            return False
        else:
            raise
    
    # Test 2: Check if feature engineering knows the models
    try:
        from app.services.transformation.feature_engineering import ForecastDataPreprocessor
        preprocessor = ForecastDataPreprocessor()
        
        # Test with models that should now be recognized
        test_models = [
            "Supply_Chain_Prophet_Forecaster",
            "Daily_Shop_Sales_Forecaster", 
            "Monthly_Shop_Sales_Forecaster",
            "LSTM Daily TotalRevenue Forecaster"
        ]
        
        for model in test_models:
            if model in preprocessor.model_requirements:
                logger.info(f"✅ {model} is recognized")
            else:
                logger.error(f"❌ {model} still not recognized")
                return False
                
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        return False
    
    # Test 3: Try model loading
    try:
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'sales': np.random.normal(100, 10, 50).cumsum() + 1000
        })
        
        processed = preprocessor.prepare_data_for_model(test_data, "Supply_Chain_Prophet_Forecaster")
        if processed is not None:
            logger.info("✅ Data preprocessing works")
        else:
            logger.warning("⚠️ Data preprocessing returned None (might be expected for some models)")
            
    except Exception as e:
        logger.error(f"❌ Data preprocessing failed: {e}")
        return False
    
    logger.info("🎉 ALL FIXES VERIFIED!")
    return True

def quick_forecast_test():
    """Quick test of the complete forecasting pipeline"""
    logger.info("🚀 QUICK FORECAST TEST")
    
    try:
        from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService
        import pandas as pd
        import numpy as np
        
        service = SupplyChainForecastingService()
        
        # Create test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.random.normal(100, 15, 100).cumsum() + 1000
        })
        
        dataset_info = {
            'name': 'quick_test',
            'columns': list(test_data.columns),
            'row_count': len(test_data),
            'data': test_data,
            'frequency': 'daily'
        }
        
        logger.info("📈 Running forecast...")
        result = service.process_forecasting_request(
            dataset_info=dataset_info,
            forecast_horizon=30
        )
        
        logger.info(f"📊 Result: {result.get('status')}")
        
        if result['status'] == 'success':
            logger.info("🎉 FORECAST SUCCESS!")
            forecast_data = result.get('forecast', {})
            logger.info(f"   Model: {result.get('selected_model', {}).get('model_name')}")
            logger.info(f"   Values: {len(forecast_data.get('values', []))}")
            return True
        else:
            logger.error(f"❌ Forecast failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Quick forecast test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 FINAL DIAGNOSTIC AFTER FIXES")
    print("=" * 60)
    
    # Test the fixes
    fixes_ok = test_fixes()
    
    if fixes_ok:
        print("\n✅ Fixes verified! Testing complete pipeline...")
        quick_forecast_test()
    else:
        print("\n❌ Fixes failed - check the logs above")
    
    print("\n" + "=" * 60)