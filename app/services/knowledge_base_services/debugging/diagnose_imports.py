# app/services/knowledge_base_services/diagnose_imports.py
# to check why imports are failing for RuleEngine
import os
import sys

def diagnose_imports():
    """Diagnose why imports are failing"""
    print("🔍 DIAGNOSING IMPORT ISSUES")
    print("=" * 50)
    
    # Get project root
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    print(f"📁 Project root: {project_root}")
    
    # Check if RuleEngine file exists
    rule_engine_path = os.path.join(project_root, 'app', 'knowledge_base', 'rule_layer', 'rule_engine.py')
    print(f"📄 RuleEngine path: {rule_engine_path}")
    print(f"   Exists: {os.path.exists(rule_engine_path)}")
    
    if os.path.exists(rule_engine_path):
        print("✅ RuleEngine file exists!")
        # Check file content
        with open(rule_engine_path, 'r') as f:
            first_lines = [f.readline() for _ in range(10)]
        print("📝 First 10 lines:")
        for i, line in enumerate(first_lines, 1):
            print(f"   {i}: {line.strip()}")
    else:
        print("❌ RuleEngine file not found!")
        # Check directory structure
        rule_layer_dir = os.path.dirname(rule_engine_path)
        print(f"📁 Rule layer directory: {rule_layer_dir}")
        print(f"   Exists: {os.path.exists(rule_layer_dir)}")
        if os.path.exists(rule_layer_dir):
            print("📋 Files in rule_layer directory:")
            for file in os.listdir(rule_layer_dir):
                print(f"   - {file}")
    
    # Check Python path
    print(f"\n🐍 Python path:")
    for path in sys.path:
        print(f"   - {path}")
    
    # Try to import
    print(f"\n🔄 Trying to import RuleEngine...")
    try:
        sys.path.insert(0, project_root)
        from app.knowledge_base.rule_layer.rule_engine import RuleEngine
        print("✅ SUCCESS: RuleEngine imported successfully!")
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        # Try direct import
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("rule_engine", rule_engine_path)
            rule_engine_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rule_engine_module)
            RuleEngine = rule_engine_module.RuleEngine
            print("✅ SUCCESS: RuleEngine loaded via direct import!")
        except Exception as e2:
            print(f"❌ Direct import also failed: {e2}")

if __name__ == "__main__":
    diagnose_imports()