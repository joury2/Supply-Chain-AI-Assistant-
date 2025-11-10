# debug_selection.py (FINAL VERSION)
from app.knowledge_base.rule_layer.rule_engine import RuleEngine
from app.repositories.model_repository import get_model_repository
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def debug_selection_failure():
    """Test the fixed RuleEngine constructor"""
    
    repo = get_model_repository()
    
    # ✅ Test 1: RuleEngine with default rules file
    print("🧪 TEST 1: RuleEngine with default rules")
    engine1 = RuleEngine(model_repository=repo)  # Uses default path
    print(f"   Rules loaded: {engine1.rules_loaded}")
    print(f"   Validation rules: {len(engine1.validation_rules)}")
    print(f"   Selection rules: {len(engine1.selection_rules)}")
    
    # ✅ Test 2: RuleEngine with explicit rules file
    print("\n🧪 TEST 2: RuleEngine with explicit rules file")
    engine2 = RuleEngine(
        model_repository=repo,
        rules_file="app/knowledge_base/rule_layer/model_selection_rules.yaml"  # Explicit path
    )
    print(f"   Rules loaded: {engine2.rules_loaded}")
    print(f"   Validation rules: {len(engine2.validation_rules)}")
    print(f"   Selection rules: {len(engine2.selection_rules)}")
    
    # Test selection functionality
    print("\n🧪 TEST 3: Selection functionality")
    monthly_shop_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=24, freq='MS'),
        'shop_id': ['shop_1'] * 24,
        'sales': [1000 + i*50 for i in range(24)]
    })
    
    metadata = repo.extract_dataset_metadata(monthly_shop_data, 'monthly_sales.csv')
    
    selection = engine2.select_model(metadata)
    print(f"   Selection result: {selection}")
    
    if selection and selection.get('selected_model'):
        print("✅ SUCCESS: Selection returned valid result!")
    else:
        print("❌ FAILED: Selection returned None or invalid result")

if __name__ == "__main__":
    debug_selection_failure()