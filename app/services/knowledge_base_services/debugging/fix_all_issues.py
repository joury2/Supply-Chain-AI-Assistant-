#!/usr/bin/env python3
"""Fix all identified issues automatically"""

import os
import re
from pathlib import Path

def fix_model_inference_service():
    """Add missing 'os' import to model_inference_service.py"""
    file_path = Path("app/services/inference/model_inference_service.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if os is already imported
    if "import os" in content:
        print("✅ 'os' import already exists")
        return True
    
    # Add import after the first line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('"""') or line.startswith("'''"):
            # Found the docstring, add import after it
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith('"""') or lines[j].startswith("'''")):
                j += 1
            lines.insert(j, "import os  # Added by fix script")
            break
    else:
        # If no docstring found, add at the top
        lines.insert(0, "import os  # Added by fix script")
    
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Added 'os' import to model_inference_service.py")
    return True

def fix_feature_engineering():
    """Add missing model requirements to feature_engineering.py"""
    file_path = Path("app/services/transformation/feature_engineering.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the _load_model_requirements method
    pattern = r'def _load_model_requirements\(self\) -> Dict\[str, Any\]:'
    match = re.search(pattern, content)
    
    if not match:
        print("❌ Could not find _load_model_requirements method")
        return False
    
    # Replace the entire method with the fixed version
    new_method = '''    def _load_model_requirements(self) -> Dict[str, Any]:
        """Load model-specific input requirements with EXPLICIT mappings"""
        return {
            "Supply_Chain_Prophet_Forecaster": {
                "type": "prophet", 
                "required_columns": ["ds", "y"],
                "required_regressors": ["is_saturday", "is_sunday"],
                "optional_regressors": ["unique_customers", "avg_revenue"],
                "feature_engineering": "light",
                "preprocessing": "none",
                "column_mapping": {
                    "date_columns": "ds",
                    "target_columns": "y"
                }
            },
            "Daily_Shop_Sales_Forecaster": {
                "type": "lightgbm",
                "required_columns": ["shop_id", "date", "sales"],
                "target_column": "sales", 
                "feature_engineering": "extensive_lags_rollings",
                "preprocessing": "categorical_encoding",
                "frequency": "daily",
                "column_mapping": {
                    "date_columns": "date",
                    "target_columns": "sales", 
                    "entity_columns": "shop_id"
                }
            },
            "Monthly_Shop_Sales_Forecaster": {
                "type": "lightgbm", 
                "required_columns": ["shop_id", "date", "sales"],
                "target_column": "sales",
                "feature_engineering": "monthly_lags", 
                "preprocessing": "categorical_encoding",
                "frequency": "monthly",
                "column_mapping": {
                    "date_columns": "date",
                    "target_columns": "sales",
                    "entity_columns": "shop_id"
                }
            },
            "LSTM Daily TotalRevenue Forecaster": {
                "type": "lstm",
                "required_columns": ["date", "daily_demand"],
                "target_column": "daily_demand",
                "feature_engineering": "time_series_lags",
                "preprocessing": "normalization",
                "frequency": "daily",
                "column_mapping": {
                    "date_columns": "date",
                    "target_columns": "daily_demand"
                }
            },
            "lightgbm_demand_forecaster": {
                "type": "lightgbm",
                "required_columns": ["date", "sales"],
                "target_column": "sales",
                "feature_engineering": "basic",
                "preprocessing": "none",
                "column_mapping": {
                    "date_columns": "date",
                    "target_columns": "sales"
                }
            },
            "prophet_forecaster": {
                "type": "prophet",
                "required_columns": ["ds", "y"],
                "target_column": "y",
                "feature_engineering": "light", 
                "preprocessing": "none",
                "column_mapping": {
                    "date_columns": "ds",
                    "target_columns": "y"
                }
            }
        }'''
    
    # Find the method and replace it
    start = match.start()
    # Find the end of the method (next method or class)
    lines = content.split('\n')
    method_start_line = content[:start].count('\n')
    
    # Find where this method ends
    indent_level = 0
    method_end_line = method_start_line
    for i in range(method_start_line, len(lines)):
        line = lines[i]
        if line.strip().startswith('def ') and i > method_start_line:
            break
        if line.strip().startswith('class ') and i > method_start_line:
            break
        method_end_line = i
    
    # Replace the method
    new_content = '\n'.join(lines[:method_start_line]) + '\n' + new_method + '\n' + '\n'.join(lines[method_end_line+1:])
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Updated model requirements in feature_engineering.py")
    return True

def main():
    print("🔧 FIXING ALL IDENTIFIED ISSUES")
    print("=" * 50)
    
    fix_model_inference_service()
    fix_feature_engineering()
    
    print("\n✅ All fixes applied!")
    print("💡 Now run the diagnostic again to test the fixes")

if __name__ == "__main__":
    main()