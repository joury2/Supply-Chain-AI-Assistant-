#!/usr/bin/env python3
"""
Quick Import Checker
Validates that all imports in a file are available
"""

import importlib
import ast
import sys
from pathlib import Path

def check_imports_in_file(file_path: str):
    """Check if all imports in a file are available"""
    print(f"🔍 Checking imports in: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        missing_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        importlib.import_module(alias.name)
                    except ImportError:
                        missing_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                    try:
                        if module:
                            full_name = f"{module}.{alias.name}"
                            # Try to import the module first
                            importlib.import_module(module)
                    except ImportError:
                        missing_imports.append(f"{module}.{alias.name}")
        
        if missing_imports:
            print(f"❌ Missing imports: {', '.join(missing_imports)}")
        else:
            print("✅ All imports are available")
            
    except Exception as e:
        print(f"⚠️  Could not check {file_path}: {e}")

# Check critical files
critical_files = [
    "app/services/knowledge_base_services/core/supply_chain_service.py",
    "app/api/main.py"
]

for file_path in critical_files:
    if Path(file_path).exists():
        check_imports_in_file(file_path)
    else:
        print(f"📁 {file_path} - File not found")