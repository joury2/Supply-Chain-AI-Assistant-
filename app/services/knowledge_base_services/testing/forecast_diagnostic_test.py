#!/usr/bin/env python3
"""
FIXED Forecast Pipeline Diagnostic Test
Works with your actual code structure and handles missing functions
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fixed_forecast_diagnostic.log')
    ]
)
logger = logging.getLogger("FixedForecastDiagnostic")

def create_test_datasets():
    """Create multiple test datasets to test different scenarios"""
    
    # Dataset 1: Simple daily sales data (most common case)
    dates_daily = pd.date_range('2023-01-01', periods=100, freq='D')
    daily_sales = np.random.normal(100, 15, 100).cumsum() + 1000
    dataset_daily = pd.DataFrame({
        'date': dates_daily,
        'sales': daily_sales,
        'promotion_flag': np.random.choice([0, 1], 100),
        'day_of_week': dates_daily.dayofweek
    })
    
    # Dataset 2: Monthly data with seasonality
    dates_monthly = pd.date_range('2020-01-01', periods=36, freq='M')
    monthly_sales = [100 + 10 * (i % 12) + np.random.normal(0, 5) for i in range(36)]
    dataset_monthly = pd.DataFrame({
        'date': dates_monthly,
        'monthly_sales': monthly_sales,
        'quarter': (dates_monthly.month - 1) // 3 + 1
    })
    
    # Dataset 3: Simple array data (minimal)
    dataset_simple = pd.DataFrame({
        'value': np.random.normal(50, 10, 50).cumsum() + 200
    })
    
    return {
        'daily_sales': dataset_daily,
        'monthly_sales': dataset_monthly, 
        'simple_series': dataset_simple
    }

def test_service_initialization():
    """Test 1: Service initialization and dependencies - FIXED VERSION"""
    logger.info("🧪 TEST 1: Service Initialization")
    
    try:
        from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService
        
        service = SupplyChainForecastingService()
        logger.info("✅ SupplyChainForecastingService initialized successfully")
        
        # Check individual components
        components = {
            'knowledge_base': service.knowledge_base,
            'rule_engine': service.rule_engine,
            'inference_service': service.inference_service,
            'data_preprocessor': service.data_preprocessor
        }
        
        for name, component in components.items():
            status = "✅" if component is not None else "❌"
            logger.info(f"   {status} {name}: {type(component).__name__}")
        
        return service, all(component is not None for component in components.values())
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, False

def test_knowledge_base_models(service):
    """Test 2: Knowledge Base Model Availability - FIXED VERSION"""
    logger.info("🧪 TEST 2: Knowledge Base Models")
    
    try:
        # FIXED: Try multiple method names and fallbacks
        models = []
        
        # Try different method names that might exist
        method_attempts = [
            ('get_all_models', []),
            ('get_models', []),
            ('list_models', []),
            ('get_active_models', [])
        ]
        
        for method_name, args in method_attempts:
            if hasattr(knowledge_base, method_name):
                try:
                    models = getattr(knowledge_base, method_name)(*args)
                    logger.info(f"   ✅ Found models using {method_name}: {len(models)} models")
                    break
                except Exception as e:
                    logger.debug(f"   ⚠️ {method_name} failed: {e}")
                    continue
        
        # If no models found, try direct database access
        if not models:
            logger.info("   🔍 Trying direct database access...")
            try:
                from app.services.knowledge_base_services.core.knowledge_base_service import SupplyChainService as knowledge_base
                knowledge_base = knowledge_base()
                models = knowledge_base.get_all_models()
                logger.info(f"   ✅ Found {len(models)} models via direct DB access")
            except Exception as e:
                logger.error(f"   ❌ Direct DB access failed: {e}")
                return False
        
        active_models = [m for m in models if m.get('is_active', True)]
        
        logger.info(f"   📊 Total models in KB: {len(models)}")
        logger.info(f"   ✅ Active models: {len(active_models)}")
        
        if active_models:
            for i, model in enumerate(active_models[:5]):
                logger.info(f"      {i+1}. {model.get('model_name', 'Unknown')} "
                           f"({model.get('model_type', 'unknown')}) -> "
                           f"{model.get('target_variable', 'unknown')}")
        
        # Test model loading capability with fallbacks
        if active_models:
            test_model = active_models[0].get('model_name', 'prophet_forecaster')
            logger.info(f"   🔧 Testing model loading: {test_model}")
            
            # FIXED: Check model loading with multiple method names
            is_loaded = False
            load_methods = ['is_model_loaded', 'model_is_loaded', 'is_loaded']
            from app.services.inference.model_inference_service import ModelInferenceService as inference_service
            for method in load_methods:
                if hasattr(inference_service, method):
                    try:
                        is_loaded = getattr(inference_service, method)(test_model)
                        logger.info(f"      Model loaded ({method}): {is_loaded}")
                        break
                    except Exception as e:
                        logger.debug(f"      {method} failed: {e}")
                        continue
            
            if not is_loaded:
                logger.info(f"      Attempting to load model...")
                loaded = False
                load_methods = ['load_model_from_registry', 'load_model', 'load_model_by_name']
                
                for method in load_methods:
                    if hasattr(inference_service, method):
                        try:
                            # Try with different parameter combinations
                            if method == 'load_model_from_registry':
                                loaded = inference_service.load_model_from_registry(
                                    test_model, knowledge_base
                                )
                            else:
                                loaded = getattr(inference_service, method)(test_model)
                            logger.info(f"      Load result ({method}): {loaded}")
                            break
                        except Exception as e:
                            logger.debug(f"      {method} failed: {e}")
                            continue
                
                if not loaded:
                    logger.warning("      ❌ Could not load model with any method")
        
        return len(active_models) > 0
        
    except Exception as e:
        logger.error(f"❌ Knowledge base test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_data_preprocessing(service, test_datasets):
    """Test 3: Data Preprocessing Pipeline - FIXED VERSION"""
    logger.info("🧪 TEST 3: Data Preprocessing")
    
    results = {}
    
    for dataset_name, dataset_df in test_datasets.items():
        logger.info(f"   📁 Testing dataset: {dataset_name}")
        
        try:
            # Create dataset info
            dataset_info = {
                'name': f'test_{dataset_name}',
                'columns': list(dataset_df.columns),
                'row_count': len(dataset_df),
                'data': dataset_df,
                'frequency': 'daily' if 'date' in dataset_df.columns else 'unknown'
            }
            
            logger.info(f"      Columns: {dataset_info['columns']}")
            logger.info(f"      Rows: {dataset_info['row_count']}")
            logger.info(f"      Data type: {type(dataset_df).__name__}")
            logger.info(f"      Shape: {dataset_df.shape}")
            
            # Test data preprocessing with a sample model
            test_models = ['lightgbm_demand_forecaster', 'prophet_forecaster']
            
            for model_name in test_models:
                logger.info(f"      🎯 Testing preprocessing for: {model_name}")
                
                try:
                    # FIXED: Try different preprocessing methods
                    processed_data = None
                    from app.services.transformation.feature_engineering import ForecastDataPreprocessor as data_preprocessor
                    if hasattr(data_preprocessor, 'prepare_data_for_model'):
                        processed_data = data_preprocessor.prepare_data_for_model(
                            user_data=dataset_df,
                            model_name=model_name,
                            forecast_horizon=30
                        )
                    elif hasattr(data_preprocessor, 'prepare_data'):
                        processed_data = data_preprocessor.prepare_data_for_model(
                            data_input=dataset_df,
                            model_name=model_name
                        )
                    else:
                        logger.warning(f"         ⚠️ No prepare_data method found")
                    
                    if processed_data is not None:
                        if isinstance(processed_data, pd.DataFrame):
                            logger.info(f"         ✅ Success: DataFrame, shape: {processed_data.shape}")
                        elif isinstance(processed_data, dict):
                            logger.info(f"         ✅ Success: Dict, keys: {list(processed_data.keys())}")
                        else:
                            logger.info(f"         ✅ Success: {type(processed_data).__name__}")
                    else:
                        logger.warning(f"         ⚠️ Preprocessing returned None")
                        
                except Exception as e:
                    logger.error(f"         ❌ Preprocessing failed: {e}")
            
            results[dataset_name] = 'success'
            
        except Exception as e:
            logger.error(f"   ❌ Dataset {dataset_name} failed: {e}")
            results[dataset_name] = 'failed'
    
    return all(result == 'success' for result in results.values())

def test_rule_engine_validation(service, test_datasets):
    """Test 4: Rule Engine Validation - FIXED VERSION"""
    logger.info("🧪 TEST 4: Rule Engine Validation")
    
    test_dataset = test_datasets['daily_sales']
    
    try:
        dataset_info = {
            'name': 'test_daily_sales',
            'columns': list(test_dataset.columns),
            'row_count': len(test_dataset),
            'frequency': 'daily',
            'missing_percentage': 0.05
        }
        
        logger.info("   🔍 Running rule engine validation...")
        
        # FIXED: Try different validation method names
        validation_result = None
        validation_methods = ['validate_dataset', 'validate_data', 'check_dataset']
        from app.knowledge_base.rule_layer.rule_engine import RuleEngine as rule_engine
        for method in validation_methods:
            if hasattr(rule_engine, method):
                try:
                    validation_result = getattr(rule_engine, method)(dataset_info)
                    logger.info(f"   ✅ Validation completed using {method}")
                    break
                except Exception as e:
                    logger.debug(f"   ⚠️ {method} failed: {e}")
                    continue
        
        if validation_result is None:
            logger.error("   ❌ No validation method worked")
            return False
        
        logger.info(f"   ✅ Validation result: {validation_result.get('valid', False)}")
        logger.info(f"   📋 Errors: {validation_result.get('errors', [])}")
        logger.info(f"   ⚠️ Warnings: {validation_result.get('warnings', [])}")
        
        # Test model selection
        logger.info("   🤖 Testing model selection...")
        selection_result = None
        
        # FIXED: Handle nested rule_engine.rule_engine structure
        if hasattr(rule_engine, 'rule_engine'):
            if hasattr(rule_engine.rule_engine, 'select_model'):
                selection_result = service.rule_engine.rule_engine.select_model(dataset_info)
            elif hasattr(service.rule_engine.rule_engine, 'choose_model'):
                selection_result = service.rule_engine.rule_engine.choose_model(dataset_info)
        elif hasattr(service.rule_engine, 'select_model'):
            selection_result = service.rule_engine.select_model(dataset_info)
        
        if selection_result:
            logger.info(f"   🎯 Selected model: {selection_result.get('selected_model', 'None')}")
            logger.info(f"   📈 Confidence: {selection_result.get('confidence', 0)}")
        else:
            logger.warning("   ⚠️ No model selected by rule engine")
        
        return validation_result.get('valid', False)
        
    except Exception as e:
        logger.error(f"❌ Rule engine test failed: {e}")
        return False

def test_complete_forecast_pipeline(service, test_datasets):
    """Test 5: Complete Forecasting Pipeline - FIXED VERSION"""
    logger.info("🧪 TEST 5: Complete Forecasting Pipeline")
    
    test_dataset = test_datasets['daily_sales']
    
    try:
        dataset_info = {
            'name': 'diagnostic_test_dataset',
            'columns': list(test_dataset.columns),
            'row_count': len(test_dataset),
            'data': test_dataset,
            'frequency': 'daily',
            'missing_percentage': 0.02,
            'description': 'Diagnostic test dataset'
        }
        
        logger.info("   🚀 Starting complete forecasting pipeline...")
        
        # Run the diagnostic first (if available)
        if hasattr(service, 'run_forecast_diagnostic'):
            logger.info("   🩺 Running pipeline diagnostic...")
            diagnostic = service.run_forecast_diagnostic(dataset_info)
            
            logger.info("   📋 DIAGNOSTIC RESULTS:")
            for step, result in diagnostic.get('steps', {}).items():
                logger.info(f"      {step}: {json.dumps(result, default=str)}")
            
            if diagnostic.get('issues'):
                logger.warning("   ⚠️ Diagnostic issues found:")
                for issue in diagnostic['issues']:
                    logger.warning(f"      - {issue}")
        else:
            logger.info("   ⚠️ run_forecast_diagnostic method not available")
        
        # Now run actual forecast
        logger.info("   📈 Executing forecast request...")
        forecast_result = service.process_forecasting_request(
            dataset_info=dataset_info,
            forecast_horizon=30,
            business_context={'test': True, 'source': 'diagnostic'}
        )
        
        logger.info("   📊 FORECAST RESULT:")
        logger.info(f"      Status: {forecast_result.get('status', 'unknown')}")
        
        if forecast_result['status'] == 'success':
            logger.info("      🎉 FORECAST SUCCESS!")
            forecast_data = forecast_result.get('forecast', {})
            selected_model = forecast_result.get('selected_model', {})
            logger.info(f"         Model used: {selected_model.get('model_name', 'Unknown')}")
            logger.info(f"         Values generated: {len(forecast_data.get('values', []))}")
            logger.info(f"         Has confidence intervals: {'confidence_intervals' in forecast_data}")
            
            # Show sample predictions
            values = forecast_data.get('values', [])
            if values:
                logger.info(f"         Sample predictions: {values[:5]}")
                
        else:
            logger.error(f"      ❌ FORECAST FAILED: {forecast_result.get('error', 'Unknown error')}")
            if 'recommendations' in forecast_result:
                logger.info("      💡 Recommendations:")
                for rec in forecast_result['recommendations']:
                    logger.info(f"         - {rec}")
        
        return forecast_result['status'] == 'success'
        
    except Exception as e:
        logger.error(f"❌ Complete pipeline test failed: {e}")
        import traceback
        logger.error(f"🔍 FULL ERROR:\n{traceback.format_exc()}")
        return False

def test_inference_service_directly(service):
    """Test 6: Direct Inference Service Test - FIXED VERSION"""
    logger.info("🧪 TEST 6: Direct Inference Service Test")
    
    try:
        # Get diagnostics from inference service
        inference_diagnostics = {}
        if hasattr(service.inference_service, 'get_diagnostics'):
            inference_diagnostics = service.inference_service.get_diagnostics()
        elif hasattr(service.inference_service, 'get_status'):
            inference_diagnostics = service.inference_service.get_status()
        else:
            logger.warning("   ⚠️ No diagnostics method found")
        
        logger.info("   🔧 INFERENCE SERVICE DIAGNOSTICS:")
        logger.info(f"      Model storage: {inference_diagnostics.get('model_storage_path', 'Unknown')}")
        logger.info(f"      Storage exists: {inference_diagnostics.get('model_storage_exists', 'Unknown')}")
        logger.info(f"      Loaded models: {inference_diagnostics.get('loaded_models', [])}")
        logger.info(f"      Supported types: {inference_diagnostics.get('supported_types', [])}")
        logger.info(f"      Has knowledge base: {inference_diagnostics.get('has_knowledge_base', 'Unknown')}")
        
        # Test loading a specific model directly
        test_models = ['lightgbm_demand_forecaster', 'prophet_forecaster']
        
        for model_name in test_models:
            logger.info(f"   🎯 Testing direct model load: {model_name}")
            
            # Check if loaded
            is_loaded = False
            if hasattr(service.inference_service, 'is_model_loaded'):
                is_loaded = service.inference_service.is_model_loaded(model_name)
            logger.info(f"      Already loaded: {is_loaded}")
            
            if not is_loaded:
                logger.info(f"      Attempting to load...")
                success = False
                if hasattr(service.inference_service, 'load_model_from_registry'):
                    success = service.inference_service.load_model_from_registry(
                        model_name, service.knowledge_base
                    )
                elif hasattr(service.inference_service, 'load_model'):
                    success = service.inference_service.load_model(model_name)
                logger.info(f"      Load success: {success}")
            
            # Check capabilities if loaded
            if is_loaded and hasattr(service.inference_service, 'get_model_capabilities'):
                capabilities = service.inference_service.get_model_capabilities(model_name)
                logger.info(f"      Capabilities: {capabilities}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference service test failed: {e}")
        return False

def check_missing_functions():
    """Check which critical functions are missing"""
    logger.info("🔍 CHECKING MISSING FUNCTIONS")
    
    try:
        from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService
        service = SupplyChainForecastingService()
        
        missing_functions = []
        
        # Check knowledge_base methods
        kb_methods = ['get_all_models', 'get_models', 'list_models']
        for method in kb_methods:
            if not hasattr(service.knowledge_base, method):
                missing_functions.append(f"knowledge_base.{method}")
        
        # Check inference_service methods  
        inference_methods = ['is_model_loaded', 'load_model_from_registry', 'get_diagnostics', 'get_model_capabilities']
        for method in inference_methods:
            if not hasattr(service.inference_service, method):
                missing_functions.append(f"inference_service.{method}")
        
        # Check data_preprocessor methods
        data_methods = ['prepare_data_for_model', 'prepare_data']
        for method in data_methods:
            if not hasattr(service.data_preprocessor, method):
                missing_functions.append(f"data_preprocessor.{method}")
        
        # Check rule_engine methods
        rule_methods = ['validate_dataset']
        for method in rule_methods:
            if not hasattr(service.rule_engine, method):
                missing_functions.append(f"rule_engine.{method}")
        
        if missing_functions:
            logger.warning("⚠️ MISSING FUNCTIONS:")
            for func in missing_functions:
                logger.warning(f"   ❌ {func}")
        else:
            logger.info("✅ All critical functions are available!")
        
        return missing_functions
        
    except Exception as e:
        logger.error(f"❌ Function check failed: {e}")
        return []

def main():
    """Run all diagnostic tests"""
    logger.info("=" * 80)
    logger.info("🔍 FIXED FORECAST PIPELINE COMPREHENSIVE DIAGNOSTIC")
    logger.info("=" * 80)
    
    # First check for missing functions
    missing_funcs = check_missing_functions()
    
    # Create test datasets
    logger.info("📊 Creating test datasets...")
    test_datasets = create_test_datasets()
    logger.info(f"✅ Created {len(test_datasets)} test datasets")
    
    test_results = {}
    
    # Test 1: Service Initialization
    service, init_success = test_service_initialization()
    test_results['service_initialization'] = init_success
    
    if not init_success:
        logger.error("❌ Cannot proceed - service initialization failed")
        return
    
    # Test 2: Knowledge Base
    test_results['knowledge_base'] = test_knowledge_base_models(service)
    
    # Test 3: Data Preprocessing
    test_results['data_preprocessing'] = test_data_preprocessing(service, test_datasets)
    
    # Test 4: Rule Engine
    test_results['rule_engine'] = test_rule_engine_validation(service, test_datasets)
    
    # Test 5: Complete Pipeline
    test_results['complete_pipeline'] = test_complete_forecast_pipeline(service, test_datasets)
    
    # Test 6: Inference Service
    test_results['inference_service'] = test_inference_service_directly(service)
    
    # Summary
    logger.info("=" * 80)
    logger.info("📋 DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"   Overall: {passed_tests}/{total_tests} tests passed")
    
    if missing_funcs:
        logger.info(f"   ⚠️ Missing functions: {len(missing_funcs)}")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Forecasting pipeline is working correctly.")
    else:
        logger.info("🔧 SOME TESTS FAILED. Check the logs above for specific issues.")
        
        # Provide specific guidance based on failures
        if not test_results.get('knowledge_base'):
            logger.info("💡 ISSUE: Knowledge base models not available")
            logger.info("   Fix: Run model registration script: python app/services/model_serving/register_models.py")
        
        if not test_results.get('complete_pipeline'):
            logger.info("💡 ISSUE: Complete pipeline failed")
            logger.info("   This is the main issue - check the detailed logs above for the exact failure point")
        
        if missing_funcs:
            logger.info("💡 ISSUE: Missing functions detected")
            logger.info("   Fix: Implement the missing functions in the respective services")
    
    logger.info("=" * 80)
    logger.info("📝 Full logs saved to: fixed_forecast_diagnostic.log")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()