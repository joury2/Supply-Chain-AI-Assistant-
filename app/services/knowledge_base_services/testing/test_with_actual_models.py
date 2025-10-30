# app/services/knowledge_base_services/test_with_actual_models.py
# Test datasets that align with actual models in the knowledge base
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def test_with_actual_models():
    """Test with datasets that match your actual available models"""
    print("🧪 TESTING WITH ACTUAL MODELS")
    print("=" * 50)
    
    from app.services.knowledge_base_services.core.rule_engine_service import RuleEngineService
    from app.services.knowledge_base_services.core.knowledge_base_service import SupplyChainService as KnowledgeBaseService
    
    # Initialize services
    rule_service = RuleEngineService()
    kb_service = KnowledgeBaseService("supply_chain.db")
    
    # Check what models are actually available
    print("📊 AVAILABLE MODELS IN DATABASE:")
    models = kb_service.get_all_models()
    for model in models:
        print(f"  • {model['model_name']} (Type: {model['model_type']}, Target: {model['target_variable']})")
        print(f"    Required: {model.get('required_features', 'N/A')}")
    
    # Test datasets tailored to your actual models
    test_datasets = [
        {
            'name': 'monthly_shop_lightgbm',
            'frequency': 'monthly',
            'granularity': 'shop_level',
            'row_count': 48,
            'columns': ['shop_id', 'month', 'year', 'sales_per_active_day', 'unique_items'],
            'missing_percentage': 0.02,
            'description': 'Matches Monthly_Shop_Sales_Forecaster'
        },
        {
            'name': 'daily_shop_lightgbm', 
            'frequency': 'daily',
            'granularity': 'shop_level',
            'row_count': 365,
            'columns': ['shop_id', 'date', 'daily_sales'],
            'missing_percentage': 0.05,
            'description': 'Matches Daily_Shop_Sales_Forecaster'
        },
        {
            'name': 'time_series_prophet',
            'frequency': 'monthly',
            'granularity': 'product_level', 
            'row_count': 120,
            'columns': ['date', 'demand', 'product_id'],
            'missing_percentage': 0.03,
            'description': 'Matches Prophet time series model'
        },
        {
            'name': 'simple_demand',
            'frequency': 'monthly',
            'granularity': 'regional_level',
            'row_count': 36,
            'columns': ['date', 'demand'],
            'missing_percentage': 0.08,
            'description': 'Simple time series for Prophet'
        }
    ]
    
    print(f"\n🔍 TESTING {len(test_datasets)} DATASETS:")
    
    for i, dataset in enumerate(test_datasets, 1):
        print(f"\n{'='*40}")
        print(f"TEST {i}: {dataset['description']}")
        print(f"{'='*40}")
        print(f"Dataset: {dataset['name']}")
        print(f"Columns: {dataset['columns']}")
        
        result = rule_service.analyze_dataset(dataset)
        
        print(f"✅ Validation: {'PASS' if result['validation']['valid'] else 'FAIL'}")
        print(f"🤖 Model Selected: {result['model_selection'].get('selected_model', 'None')}")
        print(f"🎯 Confidence: {result['model_selection'].get('confidence', 0):.1%}")
        print(f"📝 Reason: {result['model_selection'].get('reason', 'N/A')}")
        
        if result['validation']['errors']:
            print("❌ Errors:")
            for error in result['validation']['errors']:
                print(f"   - {error}")
        
        if result['validation']['warnings']:
            print("⚠️  Warnings:")
            for warning in result['validation']['warnings']:
                print(f"   - {warning}")
    
    kb_service.close()
    print(f"\n🎉 Testing completed!")

if __name__ == "__main__":
    test_with_actual_models()