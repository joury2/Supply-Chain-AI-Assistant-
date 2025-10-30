# app/services/knowledge_base_services/run_tests.py
"""
Main test runner for knowledge base services
Run specific test suites or all tests
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

def run_all_tests():
    """Run all test suites"""
    print("🚀 RUNNING ALL KNOWLEDGE BASE SERVICE TESTS")
    print("=" * 50)
    
    tests = [
        ("Core Services Test", "testing.test_simple_models"),
        ("Integration Test", "testing.integration_test"), 
        ("Automation Test", "testing.test_automation"),
    ]
    
    for test_name, test_module in tests:
        print(f"\n📋 RUNNING: {test_name}")
        print("-" * 30)
        try:
            module = __import__(f"app.services.knowledge_base_services.{test_module}", fromlist=[''])
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"⚠️  No main() function in {test_module}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n🎉 ALL TESTS COMPLETED!")

def run_specific_test(test_name):
    """Run a specific test"""
    test_map = {
        'core': 'testing.test_simple_models',
        'integration': 'testing.integration_test',
        'automation': 'testing.test_automation',
        'debug': 'debugging.quick_test'
    }
    
    if test_name in test_map:
        module_path = test_map[test_name]
        print(f"🚀 RUNNING: {test_name.upper()} TEST")
        module = __import__(f"app.services.knowledge_base_services.{module_path}", fromlist=[''])
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"❌ No main() function in {module_path}")
    else:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_map.keys())}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_specific_test(sys.argv[1])
    else:
        run_all_tests()