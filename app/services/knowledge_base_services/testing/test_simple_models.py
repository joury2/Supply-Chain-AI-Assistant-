# app/services/knowledge_base_services/test_simple_models.py
# app/services/knowledge_base_services/testing/test_simple_models.py
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

def test_simple_models():
    """Test with simplified model requirements"""
    print("🧪 TESTING WITH SIMPLIFIED MODEL REQUIREMENTS")
    print("=" * 50)
    
    # Use relative import
    from ..core.rule_engine_service import RuleEngineService
    
    service = RuleEngineService()
    
    # Test datasets with basic requirements
    test_datasets = [
        {
            'name': 'basic_time_series',
            'frequency': 'monthly',
            'granularity': 'product_level',
            'row_count': 100,
            'columns': ['date', 'demand'],
            'missing_percentage': 0.02,
            'description': 'Basic time series for Prophet'
        }
    ]
    
    for dataset in test_datasets:
        print(f"\n{'='*40}")
        print(f"TEST: {dataset['description']}")
        print(f"{'='*40}")
        
        result = service.analyze_dataset(dataset)
        summary = result['summary']
        
        print(f"📋 STATUS: {summary['status_message']}")
        print(f"🎯 Confidence: {summary['confidence']:.1%}")
        
        if not summary['can_proceed']:
            print(f"🚫 Reason: {summary['rejection_reason']}")

if __name__ == "__main__":
    test_simple_models()