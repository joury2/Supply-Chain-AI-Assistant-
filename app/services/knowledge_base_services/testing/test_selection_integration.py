# app/services/knowledge_base_services/testing/test_selection_integration.py
"""
Complete integration test for rule selection with database compatibility
FIXED: Test data now matches rule expectations
"""

import pandas as pd
import logging
from app.repositories.model_repository import get_model_repository
from app.knowledge_base.rule_layer.rule_engine import RuleEngine
from app.services.transformation.feature_engineering import ForecastDataPreprocessor
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_selection_pipeline():
    """Test entire selection pipeline: data → metadata → rules → model"""
    
    print("="*70)
    print("🧪 COMPLETE SELECTION PIPELINE TEST WITH HORIZONS")
    print("="*70)
    
    repo = get_model_repository()
    engine = RuleEngine(model_repository=repo)
    preprocessor = ForecastDataPreprocessor()
    
    # ========================================================================
    # Test Case 1: Monthly Shop-Level Sales (FIXED)
    # ========================================================================
    print("\n📊 TEST 1: Monthly Shop-Level Sales Data")
    print("-"*70)
    
    monthly_shop_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=24, freq='MS'),  # Monthly Start
        'shop_id': ['shop_1'] * 24,  # This is the COLUMN name
        'sales': [1000 + i*50 for i in range(24)]
    })
    
    metadata1 = repo.extract_dataset_metadata(monthly_shop_data, 'monthly_sales.csv')
    print(f"Extracted Metadata: {metadata1}")
    
    validation1 = engine.validate_dataset(metadata1)
    print(f"Validation: {validation1['valid']} - {validation1.get('errors', [])}")
    
    if validation1['valid']:
        selection1 = engine.select_model(metadata1)
        print(f"✅ Selected Model: {selection1['selected_model']}")
        print(f"   Reason: {selection1['reason']}")
        print(f"   Confidence: {selection1['confidence']}")
        
        # Test horizon determination
        test_horizons = [None, 6, 18]  # Auto, 6 months, 18 months
        for horizon in test_horizons:
            prepared_data = preprocessor.prepare_data_for_model(
                monthly_shop_data, selection1['selected_model'], horizon
            )
            if prepared_data is not None:
                actual_horizon = getattr(prepared_data, 'attrs', {}).get('forecast_horizon', 'unknown')
                print(f"   Horizon {horizon} → Prepared with: {actual_horizon}")
        
        # Verify model exists in DB
        model_info = repo.get_model_by_name(selection1['selected_model'])
        if model_info:
            print(f"   ✅ Model found in DB")
            print(f"   Target: {model_info['target_variable']}")
            print(f"   Type: {model_info['model_type']}")
        else:
            print(f"   ❌ Model NOT found in DB!")
    
    # ========================================================================
    # Test Case 2: Daily Transaction Revenue (LSTM) - FIXED
    # ========================================================================
    print("\n📊 TEST 2: Daily Transaction Revenue Data (LSTM) - 45 day horizon")
    print("-"*70)
    
    lstm_data = pd.DataFrame({
        'WorkDate': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Customer': ['Walmart'] * 100,
        'Location': ['New York'] * 100,
        'BusinessType': ['Final Mile'] * 100,
        'OrderCount': [25 + i % 10 for i in range(100)],
        'TotalRevenue': [2500 + i*50 for i in range(100)]
    })
    
    metadata2 = repo.extract_dataset_metadata(lstm_data, 'daily_revenue.csv')
    print(f"Extracted Metadata: {metadata2}")
    
    validation2 = engine.validate_dataset(metadata2)
    print(f"Validation: {validation2['valid']}")
    
    if validation2['valid']:
        selection2 = engine.select_model(metadata2)
        print(f"✅ Selected Model: {selection2['selected_model']}")
        print(f"   Reason: {selection2['reason']}")
        
        # Test with custom 45-day horizon
        prepared_data = preprocessor.prepare_data_for_model(
            lstm_data, selection2['selected_model'], 45
        )
        if prepared_data is not None:
            actual_horizon = getattr(prepared_data, 'attrs', {}).get('forecast_horizon', 'unknown')
            print(f"   Custom horizon 45 → Prepared with: {actual_horizon}")
    
    # ========================================================================
    # Test Case 3: Monthly Location Pieces (XGBoost) - FIXED WITH MONTHLY DATA
    # ========================================================================
    print("\n📊 TEST 3: Monthly Location Pieces Data (XGBoost) - 3 month horizon")
    print("-"*70)
    
    # FIXED: Use MONTHLY data since XGBoost rule expects monthly frequency
    xgboost_data = pd.DataFrame({
        'WorkDate': pd.date_range('2023-01-01', periods=18, freq='MS'),  # MONTHLY!
        'Customer': ['Amazon'] * 18,
        'Location': ['Dubai'] * 18,
        'BusinessType': ['Middle Mile'] * 18,
        'NumberOfPieces': [500 + i*10 for i in range(18)],
        'TotalRevenue': [5000 + i*100 for i in range(18)]
    })
    
    metadata3 = repo.extract_dataset_metadata(xgboost_data, 'location_pieces.csv')
    print(f"Extracted Metadata: {metadata3}")
    
    validation3 = engine.validate_dataset(metadata3)
    print(f"Validation: {validation3['valid']}")
    
    if validation3['valid']:
        selection3 = engine.select_model(metadata3)
        print(f"✅ Selected Model: {selection3['selected_model']}")
        print(f"   Reason: {selection3['reason']}")
        
        # Test with short 3-month horizon
        prepared_data = preprocessor.prepare_data_for_model(
            xgboost_data, selection3['selected_model'], 3
        )
        if prepared_data is not None:
            actual_horizon = getattr(prepared_data, 'attrs', {}).get('forecast_horizon', 'unknown')
            frequency = getattr(prepared_data, 'attrs', {}).get('model_frequency', 'unknown')
            print(f"   Short horizon 3 → Prepared with: {actual_horizon} ({frequency})")
    
    # ========================================================================
    # Test Case 4: Prophet Aggregate Data (NO SHOP_ID)
    # ========================================================================
    print("\n📊 TEST 4: Prophet Aggregate Time Series - No Shop Level")
    print("-"*70)
    
    prophet_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=60, freq='D'),
        'sales': [1000 + i*10 for i in range(60)]
    })
    
    metadata4 = repo.extract_dataset_metadata(prophet_data, 'aggregate_sales.csv')
    print(f"Extracted Metadata: {metadata4}")
    
    validation4 = engine.validate_dataset(metadata4)
    print(f"Validation: {validation4['valid']}")
    
    if validation4['valid']:
        selection4 = engine.select_model(metadata4)
        print(f"✅ Selected Model: {selection4['selected_model']}")
        print(f"   Reason: {selection4['reason']}")
    
    # ========================================================================
    # Test Case 5: Horizon Boundary Testing
    # ========================================================================
    print("\n📊 TEST 5: Horizon Boundary Testing")
    print("-"*70)
    
    test_cases = [
        ("daily", 365, "1 year daily"),
        ("weekly", 52, "1 year weekly"), 
        ("monthly", 24, "2 years monthly"),
        ("quarterly", 12, "3 years quarterly")
    ]
    
    for freq, horizon, description in test_cases:
        print(f"   {description}: horizon {horizon} → would use {min(horizon, 90)} (capped)")
    
    print("\n" + "="*70)
    print("🎉 INTEGRATION TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_complete_selection_pipeline()