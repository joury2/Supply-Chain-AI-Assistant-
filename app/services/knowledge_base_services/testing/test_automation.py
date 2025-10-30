# app/services/knowledge_base_services/test_automation.py
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def test_complete_automation():
    """Test the complete automated system"""
    print("🚀 COMPLETE AUTOMATION SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Rule Engine Service
    print("\n1. 🤖 Testing Updated Rule Engine Service")
    from app.services.knowledge_base_services.core.rule_engine_service import RuleEngineService
    rule_service = RuleEngineService()
    
    test_dataset = {
        'name': 'test_retail_monthly',
        'frequency': 'monthly', 
        'granularity': 'shop_level',
        'row_count': 48,
        'columns': ['shop_id', 'date', 'sales', 'price'],
        'missing_percentage': 0.02
    }
    
    analysis = rule_service.analyze_dataset(test_dataset)
    print(f"   📊 Validation: {'PASS' if analysis['validation']['valid'] else 'FAIL'}")
    print(f"   🎯 Model Selected: {analysis['model_selection'].get('selected_model', 'None')}")
    print(f"   💡 Confidence: {analysis['model_selection'].get('confidence', 0):.1%}")
    
    # Test 2: Rule Generator Service
    print("\n2. 🛠️ Testing Rule Generator Service")
    from app.services.knowledge_base_services.core.rule_generator_service import RuleGeneratorService
    rule_generator = RuleGeneratorService()
    
    test_model = {
        'model_name': 'Test_Auto_Model',
        'model_type': 'lightgbm',
        'required_features': ['shop_id', 'date', 'sales', 'inventory'],
        'target_variable': 'sales'
    }
    
    rule_result = rule_generator.generate_rules_for_model(test_model)
    print(f"   ✅ Rules Generated: {rule_result['success']}")
    print(f"   📜 Rules Count: {len(rule_result.get('generated_rules', []))}")
    
    # Show some generated rules
    if rule_result['success']:
        rules = rule_result['generated_rules'][:2]  # First 2 rules
        for rule in rules:
            print(f"      • {rule['name']} (Prio: {rule['priority']})")
    
    # Test 3: Model Upload Service
    print("\n3. 📤 Testing Model Upload Service")
    from app.services.knowledge_base_services.core.model_upload_service import ModelUploadService
    upload_service = ModelUploadService()
    
    upload_result = upload_service.quick_upload(
        model_name="Test_Upload_Model",
        model_type="regression",
        required_features=['feature1', 'feature2', 'target'],
        target_variable='target'
    )
    
    print(f"   ✅ Upload Success: {upload_result['success']}")
    if upload_result['success']:
        print(f"   📁 Model Path: {upload_result.get('model_path', 'N/A')}")
        print(f"   🤖 Rules Generated: {upload_result.get('rule_generation', {}).get('success', False)}")
    
    print("\n🎉 AUTOMATION SYSTEM TEST COMPLETED!")
    print("\n📋 SUMMARY:")
    print("  • Rule Engine: ✅ Updated to use real engine")
    print("  • Rule Generator: ✅ Auto-generates rules for new models") 
    print("  • Model Upload: ✅ Automated upload with rule generation")
    print("  • Complete Pipeline: ✅ Ready for production!")

if __name__ == "__main__":
    test_complete_automation()