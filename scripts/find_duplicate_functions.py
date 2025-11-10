# scripts/find_duplicate_functions.py
"""
Duplicate Function Finder
Identifies duplicate or similar function implementations across codebase
"""

import os
import ast
import hashlib
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import difflib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FunctionInfo:
    """Store information about a function"""
    def __init__(self, name: str, file_path: str, line_num: int, 
                 source: str, signature: str):
        self.name = name
        self.file_path = file_path
        self.line_num = line_num
        self.source = source
        self.signature = signature
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash of function implementation (ignore whitespace/comments)"""
        # Normalize the source code
        normalized = self.source.strip()
        # Remove comments
        lines = [line for line in normalized.split('\n') 
                if not line.strip().startswith('#')]
        normalized = '\n'.join(lines)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def __repr__(self):
        return f"Function({self.name} at {self.file_path}:{self.line_num})"


class DuplicateFunctionFinder:
    """Find duplicate function definitions across files"""
    
    def __init__(self, root_dir: str = "app"):
        self.root_dir = root_dir
        self.functions: Dict[str, List[FunctionInfo]] = defaultdict(list)
        self.exact_duplicates: Dict[str, List[FunctionInfo]] = defaultdict(list)
        self.similar_functions: List[Tuple[FunctionInfo, FunctionInfo, float]] = []
    
    def scan_directory(self):
        """Scan all Python files for function definitions"""
        logger.info(f"🔍 Scanning {self.root_dir} for function definitions...")
        
        py_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('__')]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    py_files.append(file_path)
        
        logger.info(f"   Found {len(py_files)} Python files")
        
        # Parse functions from each file
        total_functions = 0
        for file_path in py_files:
            functions = self._parse_functions(file_path)
            total_functions += len(functions)
        
        logger.info(f"   Parsed {total_functions} functions")
    
    def _parse_functions(self, file_path: str) -> List[FunctionInfo]:
        """Extract all function definitions from a file"""
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            tree = ast.parse(content, filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]
                    signature = f"{node.name}({', '.join(args)})"
                    
                    # Get source code
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    source = '\n'.join(lines[start_line:end_line])
                    
                    func_info = FunctionInfo(
                        name=node.name,
                        file_path=file_path,
                        line_num=node.lineno,
                        source=source,
                        signature=signature
                    )
                    
                    functions.append(func_info)
                    self.functions[node.name].append(func_info)
        
        except Exception as e:
            logger.debug(f"Could not parse {file_path}: {e}")
        
        return functions
    
    def find_exact_duplicates(self):
        """Find functions with identical implementations"""
        logger.info("🔍 Finding exact duplicates...")
        
        # Group by hash
        hash_to_functions = defaultdict(list)
        for name, func_list in self.functions.items():
            for func in func_list:
                hash_to_functions[func.hash].append(func)
        
        # Find duplicates
        for func_hash, func_list in hash_to_functions.items():
            if len(func_list) > 1:
                # Group by name
                name_groups = defaultdict(list)
                for func in func_list:
                    name_groups[func.name].append(func)
                
                for name, duplicates in name_groups.items():
                    if len(duplicates) > 1:
                        self.exact_duplicates[name].extend(duplicates)
        
        logger.info(f"   Found {len(self.exact_duplicates)} duplicate function names")
    
    def find_similar_functions(self, similarity_threshold: float = 0.8):
        """Find functions with similar implementations"""
        logger.info(f"🔍 Finding similar functions (threshold: {similarity_threshold})...")
        
        # Compare functions with the same name
        for name, func_list in self.functions.items():
            if len(func_list) > 1:
                # Compare each pair
                for i in range(len(func_list)):
                    for j in range(i + 1, len(func_list)):
                        func1 = func_list[i]
                        func2 = func_list[j]
                        
                        similarity = self._compute_similarity(func1.source, func2.source)
                        
                        if similarity >= similarity_threshold and func1.hash != func2.hash:
                            self.similar_functions.append((func1, func2, similarity))
        
        logger.info(f"   Found {len(self.similar_functions)} similar function pairs")
    
    def _compute_similarity(self, source1: str, source2: str) -> float:
        """Compute similarity ratio between two code snippets"""
        return difflib.SequenceMatcher(None, source1, source2).ratio()
    
    def generate_report(self) -> str:
        """Generate comprehensive duplicate analysis report"""
        report = []
        report.append("=" * 80)
        report.append("🔍 DUPLICATE FUNCTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Section 1: Exact Duplicates
        report.append("🔴 EXACT DUPLICATES (100% identical)")
        report.append("-" * 80)
        
        if not self.exact_duplicates:
            report.append("✅ No exact duplicates found")
        else:
            for name, duplicates in sorted(self.exact_duplicates.items()):
                report.append(f"\nFunction: {name}()")
                report.append(f"   Defined in {len(duplicates)} locations:")
                
                for func in duplicates:
                    short_path = func.file_path.replace(self.root_dir + '/', '')
                    report.append(f"      • {short_path}:{func.line_num}")
                
                report.append(f"\n   💡 Solution: Keep ONE implementation, remove others")
                report.append(f"      Recommended location: {duplicates[0].file_path}")
        
        report.append("")
        report.append("")
        
        # Section 2: Similar Functions
        report.append("🟡 SIMILAR FUNCTIONS (80%+ identical)")
        report.append("-" * 80)
        
        if not self.similar_functions:
            report.append("✅ No highly similar functions found")
        else:
            for func1, func2, similarity in sorted(self.similar_functions, 
                                                   key=lambda x: x[2], reverse=True):
                report.append(f"\nFunction: {func1.name}()")
                report.append(f"   Similarity: {similarity*100:.1f}%")
                
                short_path1 = func1.file_path.replace(self.root_dir + '/', '')
                short_path2 = func2.file_path.replace(self.root_dir + '/', '')
                
                report.append(f"   Location 1: {short_path1}:{func1.line_num}")
                report.append(f"   Location 2: {short_path2}:{func2.line_num}")
                
                # Show diff
                diff = self._generate_diff(func1.source, func2.source)
                if diff:
                    report.append(f"\n   Differences:")
                    for line in diff[:5]:  # Show first 5 differences
                        report.append(f"      {line}")
                
                report.append(f"\n   💡 Solution: Consolidate into single implementation")
        
        report.append("")
        report.append("")
        
        # Section 3: Priority Actions
        report.append("=" * 80)
        report.append("🎯 PRIORITY ACTIONS")
        report.append("=" * 80)
        
        priority_funcs = self._identify_priority_duplicates()
        for idx, (name, locations, reason) in enumerate(priority_funcs, 1):
            report.append(f"\n{idx}. {name}()")
            report.append(f"   Locations: {len(locations)}")
            report.append(f"   Priority: {reason}")
            report.append(f"   Action: Consolidate to one location")
        
        report.append("")
        report.append("")
        
        # Summary Statistics
        report.append("=" * 80)
        report.append("📊 SUMMARY STATISTICS")
        report.append("=" * 80)
        report.append(f"Total Functions Analyzed: {sum(len(v) for v in self.functions.values())}")
        report.append(f"Functions with Exact Duplicates: {len(self.exact_duplicates)}")
        report.append(f"Total Exact Duplicate Instances: {sum(len(v) for v in self.exact_duplicates.values())}")
        report.append(f"Similar Function Pairs: {len(self.similar_functions)}")
        report.append("")
        
        return "\n".join(report)
    
    def _generate_diff(self, source1: str, source2: str) -> List[str]:
        """Generate diff between two code snippets"""
        diff = difflib.unified_diff(
            source1.split('\n'),
            source2.split('\n'),
            lineterm=''
        )
        return [line for line in diff if line.startswith(('+', '-')) 
                and not line.startswith(('+++', '---'))]
    
    def _identify_priority_duplicates(self) -> List[Tuple[str, List[str], str]]:
        """Identify which duplicates should be fixed first"""
        priority = []
        
        # High priority: duplicates in core services
        core_keywords = ['service', 'engine', 'repository', 'knowledge_base']
        
        for name, duplicates in self.exact_duplicates.items():
            locations = [func.file_path for func in duplicates]
            
            # Check if in core modules
            is_core = any(keyword in ' '.join(locations).lower() 
                         for keyword in core_keywords)
            
            if is_core:
                priority.append((name, locations, "🔴 HIGH - Core service duplication"))
            elif len(duplicates) > 2:
                priority.append((name, locations, "🟡 MEDIUM - Multiple copies"))
            else:
                priority.append((name, locations, "🟢 LOW - Two copies"))
        
        return sorted(priority, key=lambda x: (x[2], len(x[1])), reverse=True)[:10]


def main():
    """Run duplicate function detection"""
    print("\n" + "🔍" * 40)
    print("DUPLICATE FUNCTION FINDER")
    print("🔍" * 40 + "\n")
    
    finder = DuplicateFunctionFinder(root_dir="app")
    
    # Scan files
    finder.scan_directory()
    
    # Find duplicates
    finder.find_exact_duplicates()
    finder.find_similar_functions(similarity_threshold=0.8)
    
    # Generate report
    report = finder.generate_report()
    print(report)
    
    # Save to file
    with open("duplicate_functions_report.txt", 'w') as f:
        f.write(report)
    
    print("📄 Full report saved to: duplicate_functions_report.txt")
    print("\n" + "✅" * 40)
    print(f"ANALYSIS COMPLETE")
    print(f"Found {len(finder.exact_duplicates)} exact duplicates")
    print(f"Found {len(finder.similar_functions)} similar function pairs")
    print("✅" * 40 + "\n")


if __name__ == "__main__":
    main()