# app/services/knowledge_base_services/debugging/fix_agent.py
import re

def fix_supply_chain_service():
    """Fix DataFrame caching issue"""
    file_path = "app/services/knowledge_base_services/core/supply_chain_service.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find analyze_dataset_with_knowledge_base method
    old_pattern = r"def analyze_dataset_with_knowledge_base\(self, dataset_info: Dict\[str, Any\]\) -> Dict\[str, Any\]:\s+\"\"\""
    
    replacement = """def analyze_dataset_with_knowledge_base(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Enhanced dataset analysis using both rule engine and knowledge base with caching
        \"\"\"
        # ✅ FIX: Extract DataFrame BEFORE caching
        dataset_df = dataset_info.pop('data', None)
        """
    
    content = re.sub(old_pattern, replacement, content)
    
    # Fix the caching line
    old_cache = "self._analysis_cache[cache_key] = {\n                'result': combined_analysis,"
    new_cache = """# ✅ Remove DataFrame before caching
        cache_data = {k: v for k, v in combined_analysis.items() if k != 'data'}
        self._analysis_cache[cache_key] = {
                'result': cache_data,"""
    
    content = content.replace(old_cache, new_cache)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed supply_chain_service.py")

def fix_forecast_agent():
    """Fix agent early_stopping_method"""
    file_path = "app/services/llm/forecast_agent.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove early_stopping_method line
    content = content.replace(
        'early_stopping_method="generate"  # Better error recovery',
        'return_intermediate_steps=False,\n            max_execution_time=60  # 60 second timeout'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed forecast_agent.py")

if __name__ == "__main__":
    print("🔧 Applying fixes...")
    fix_supply_chain_service()
    fix_forecast_agent()
    print("🎉 All fixes applied! Run your agent again.")