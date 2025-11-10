#!/usr/bin/env python3
"""
Function Dependency Analyzer
Finds functions that are called but not defined in the current file
Helps identify missing imports and undefined functions
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FunctionDependencyAnalyzer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.all_functions = {}  # file_path -> set of defined functions
        self.function_calls = {}  # file_path -> set of called functions
        self.imports = {}  # file_path -> import statements
        
    def analyze_project(self) -> Dict[str, List[str]]:
        """Analyze entire project for function dependencies"""
        logger.info(f"🔍 Analyzing project: {self.project_root}")
        
        # First pass: collect all defined functions
        self._collect_all_functions()
        
        # Second pass: analyze function calls
        self._analyze_all_function_calls()
        
        # Third pass: find undefined functions
        return self._find_undefined_functions()
    
    def _collect_all_functions(self):
        """Collect all function definitions in the project"""
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if any(part.startswith('.') or part == '__pycache__' for part in file_path.parts):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to get function definitions
                tree = ast.parse(content)
                defined_functions = set()
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        defined_functions.add(node.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(f"import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            imports.append(f"from {module} import {alias.name}")
                
                self.all_functions[str(file_path)] = defined_functions
                self.imports[str(file_path)] = imports
                
            except Exception as e:
                logger.warning(f"⚠️ Could not parse {file_path}: {e}")
    
    def _analyze_all_function_calls(self):
        """Analyze function calls in all files"""
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if any(part.startswith('.') or part == '__pycache__' for part in file_path.parts):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                called_functions = self._extract_function_calls(content, str(file_path))
                self.function_calls[str(file_path)] = called_functions
                
            except Exception as e:
                logger.warning(f"⚠️ Could not analyze calls in {file_path}: {e}")
    
    def _extract_function_calls(self, content: str, file_path: str) -> Set[str]:
        """Extract function calls from Python code"""
        called_functions = set()
        
        try:
            # Method 1: AST parsing for precise function calls
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_functions.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        # Handle method calls like obj.method()
                        called_functions.add(node.func.attr)
        except:
            # Method 2: Regex fallback if AST fails
            function_call_pattern = r'(\w+)\s*\('
            matches = re.findall(function_call_pattern, content)
            called_functions.update(matches)
            
            # Remove Python built-ins and common methods
            built_ins = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set'}
            called_functions = called_functions - built_ins
        
        return called_functions
    
    def _find_undefined_functions(self) -> Dict[str, List[str]]:
        """Find functions that are called but not defined in the same file"""
        undefined_functions = {}
        
        for file_path, called_funcs in self.function_calls.items():
            defined_funcs = self.all_functions.get(file_path, set())
            undefined_in_file = called_funcs - defined_funcs
            
            if undefined_in_file:
                # Filter out likely imported functions
                likely_undefined = self._filter_likely_imported(file_path, undefined_in_file)
                if likely_undefined:
                    undefined_functions[file_path] = sorted(likely_undefined)
        
        return undefined_functions
    
    def _filter_likely_imported(self, file_path: str, undefined_funcs: Set[str]) -> List[str]:
        """Filter out functions that are likely imported"""
        file_imports = self.imports.get(file_path, [])
        import_text = ' '.join(file_imports)
        
        likely_undefined = []
        for func in undefined_funcs:
            # Check if function might be imported
            if func in import_text:
                continue
            # Check if it's a common Python built-in (missed by first filter)
            if func in dir(__builtins__):
                continue
            likely_undefined.append(func)
        
        return likely_undefined
    
    def generate_report(self, undefined_functions: Dict[str, List[str]]):
        """Generate a comprehensive report"""
        print("\n" + "="*80)
        print("🚀 FUNCTION DEPENDENCY ANALYSIS REPORT")
        print("="*80)
        
        if not undefined_functions:
            print("✅ No undefined function calls found!")
            return
        
        print(f"❌ Found {len(undefined_functions)} files with potentially undefined functions:\n")
        
        for file_path, functions in undefined_functions.items():
            relative_path = Path(file_path).relative_to(self.project_root)
            print(f"📁 {relative_path}")
            print(f"   Undefined functions: {', '.join(functions)}")
            
            # Show context for each undefined function
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                
                for i, line in enumerate(content, 1):
                    for func in functions:
                        if func in line and '(' in line:
                            print(f"      Line {i}: {line.strip()}")
                            break
            except:
                pass
            print()


# Quick analyzer for single file
def analyze_single_file(file_path: str) -> List[str]:
    """Quick analysis for a single file"""
    analyzer = FunctionDependencyAnalyzer()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get defined functions
        tree = ast.parse(content)
        defined_functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)
        
        # Get called functions
        called_functions = analyzer._extract_function_calls(content, file_path)
        
        # Find undefined
        undefined = called_functions - defined_functions
        
        # Filter out built-ins
        built_ins = set(dir(__builtins__))
        undefined = undefined - built_ins
        
        return sorted(undefined)
        
    except Exception as e:
        logger.error(f"❌ Could not analyze {file_path}: {e}")
        return []


# Specific analyzer for your supply chain project
def analyze_critical_files():
    """Analyze critical files in your supply chain project"""
    critical_files = [
        "app/services/knowledge_base_services/core/supply_chain_service.py",
        "app/services/inference/model_inference_service.py", 
        "app/api/main.py",
        "app/services/llm/simplified_forecast_agent.py"
    ]
    
    print("🔍 ANALYZING CRITICAL FILES FOR UNDEFINED FUNCTIONS")
    print("="*60)
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            undefined = analyze_single_file(file_path)
            if undefined:
                print(f"❌ {file_path}")
                print(f"   Undefined: {', '.join(undefined)}")
            else:
                print(f"✅ {file_path} - No undefined functions")
        else:
            print(f"⚠️  {file_path} - File not found")
        print()


if __name__ == "__main__":
    # Analyze entire project
    analyzer = FunctionDependencyAnalyzer()
    undefined_functions = analyzer.analyze_project()
    analyzer.generate_report(undefined_functions)
    
    # Also analyze critical files specifically
    analyze_critical_files()