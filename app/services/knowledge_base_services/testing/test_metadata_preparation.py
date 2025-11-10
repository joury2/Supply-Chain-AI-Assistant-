# test_metadata_preparation.py
# app/services/knowledge_base_services/testing/test_metadata_preparation.py

from app.services.transformation.feature_engineering import ForecastDataPreprocessor
import pandas as pd
import json

def test_metadata_based_preparation():
    """Test that data preparation matches metadata specifications"""
    
    preprocessor = ForecastDataPreprocessor()
    
    print("🧪 METADATA-BASED PREPARATION TEST")
    print("=" * 60)
    
    # Test LSTM data preparation
    print("\n📊 TESTING LSTM DATA PREPARATION")
    lstm_data = pd.DataFrame({
        'WorkDate': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Customer': ['Walmart'] * 100,
        'Location': ['New York'] * 100,
        'BusinessType': ['Final Mile'] * 100, 
        'OrderCount': [25 + i % 10 for i in range(100)],
        'TotalRevenue': [2500 + i*50 for i in range(100)]
    })
    
    lstm_prepared = preprocessor.prepare_data_for_model(
        lstm_data, 'LSTM Daily TotalRevenue Forecaster', 45
    )
    
    if lstm_prepared is not None:
        print("✅ LSTM preparation SUCCESS")
        print(f"   Shape: {lstm_prepared.shape}")
        print(f"   Features: {lstm_prepared.columns.tolist()}")
        print(f"   Date features extracted: {[f for f in lstm_prepared.columns if f in ['year', 'month', 'day', 'dayofweek', 'quarter']]}")
        print(f"   WorkDate dropped: {'WorkDate' not in lstm_prepared.columns}")
    else:
        print("❌ LSTM preparation FAILED")
    
    # Test XGBoost data preparation  
    print("\n📊 TESTING XGBOOST DATA PREPARATION")
    xgboost_data = pd.DataFrame({
        'WorkDate': pd.date_range('2023-01-01', periods=18, freq='MS'),
        'Customer': ['Amazon'] * 18,
        'Location': ['Dubai'] * 18, 
        'BusinessType': ['Middle Mile'] * 18,
        'NumberOfPieces': [500 + i*10 for i in range(18)],
        'TotalRevenue': [5000 + i*100 for i in range(18)]
    })
    
    xgboost_prepared = preprocessor.prepare_data_for_model(
        xgboost_data, 'XGBoost Multi-Location NumberOfPieces Forecaster', 3
    )
    
    if xgboost_prepared is not None:
        print("✅ XGBoost preparation SUCCESS") 
        print(f"   Shape: {xgboost_prepared.shape}")
        print(f"   Features: {xgboost_prepared.columns.tolist()}")
        print(f"   Lag features: {[f for f in xgboost_prepared.columns if 'lag_' in f]}")
        print(f"   Rolling features: {[f for f in xgboost_prepared.columns if 'roll_' in f]}")
        print(f"   Temporal features: {[f for f in xgboost_prepared.columns if f in ['month', 'quarter', 'year']]}")
    else:
        print("❌ XGBoost preparation FAILED")

if __name__ == "__main__":
    test_metadata_based_preparation()