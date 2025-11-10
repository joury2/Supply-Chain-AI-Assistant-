# scripts/detect_circular_imports.py
"""
Circular Import Detection Tool
Automatically finds and reports circular dependencies in your codebase
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircularImportDetector:
    """Detect circular imports in Python codebase"""
    
    def __init__(self, root_dir: str = "app"):
        self.root_dir = root_dir
        self.import_graph = defaultdict(set)  # file -> set of imported files
        self.file_map = {}  # module_name -> file_path
        
    def scan_directory(self):
        """Scan all Python files and build import graph"""
        logger.info(f"🔍 Scanning {self.root_dir} for Python files...")
        
        py_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip __pycache__ and similar
            dirs[:] = [d for d in dirs if not d.startswith('__')]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    py_files.append(file_path)
                    
                    # Create module name from path
                    module_name = self._path_to_module(file_path)
                    self.file_map[module_name] = file_path
        
        logger.info(f"   Found {len(py_files)} Python files")
        
        # Parse imports from each file
        for file_path in py_files:
            self._parse_imports(file_path)
        
        logger.info(f"   Built import graph with {len(self.import_graph)} nodes")
    
    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name"""
        # Remove root_dir and .py extension
        rel_path = os.path.relpath(file_path, self.root_dir)
        module_path = rel_path.replace(os.sep, '.').replace('.py', '')
        return f"{self.root_dir}.{module_path}"
    
    def _parse_imports(self, file_path: str):
        """Parse imports from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            current_module = self._path_to_module(file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module = alias.name
                        self.import_graph[current_module].add(imported_module)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_module = node.module
                        self.import_graph[current_module].add(imported_module)
        
        except Exception as e:
            logger.debug(f"Could not parse {file_path}: {e}")
    
    def find_circular_imports(self) -> List[List[str]]:
        """Find all circular import chains using DFS"""
        logger.info("🔄 Detecting circular imports...")
        
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.import_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    
                    # Normalize cycle (start with smallest element)
                    min_idx = cycle.index(min(cycle[:-1]))
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    
                    if normalized not in cycles:
                        cycles.append(normalized)
            
            rec_stack.remove(node)
        
        # Run DFS from each node
        for node in self.import_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def analyze_specific_cycle(self, cycle: List[str]) -> Dict:
        """Analyze a specific circular import cycle"""
        analysis = {
            'cycle': cycle,
            'length': len(cycle) - 1,  # Exclude duplicate start/end
            'files': [],
            'import_details': []
        }
        
        for i in range(len(cycle) - 1):
            from_module = cycle[i]
            to_module = cycle[i + 1]
            
            from_file = self.file_map.get(from_module, 'Unknown')
            to_file = self.file_map.get(to_module, 'Unknown')
            
            analysis['files'].append(from_file)
            
            # Find the specific import statement
            import_detail = self._find_import_statement(from_file, to_module)
            analysis['import_details'].append({
                'from': from_module,
                'to': to_module,
                'line': import_detail['line'],
                'statement': import_detail['statement']
            })
        
        return analysis
    
    def _find_import_statement(self, file_path: str, target_module: str) -> Dict:
        """Find the specific line where an import occurs"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Look for import patterns
            patterns = [
                rf'from\s+{re.escape(target_module)}\s+import',
                rf'import\s+{re.escape(target_module)}\b'
            ]
            
            for line_num, line in enumerate(lines, 1):
                for pattern in patterns:
                    if re.search(pattern, line):
                        return {
                            'line': line_num,
                            'statement': line.strip()
                        }
            
            return {'line': 0, 'statement': 'Import not found'}
        
        except Exception as e:
            return {'line': 0, 'statement': f'Error: {e}'}
    
    def generate_report(self, cycles: List[List[str]]) -> str:
        """Generate a comprehensive report of circular imports"""
        report = []
        report.append("=" * 80)
        report.append("🔄 CIRCULAR IMPORT DETECTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        if not cycles:
            report.append("✅ No circular imports detected!")
            report.append("")
            return "\n".join(report)
        
        report.append(f"❌ Found {len(cycles)} circular import chain(s)")
        report.append("")
        
        for idx, cycle in enumerate(cycles, 1):
            report.append("-" * 80)
            report.append(f"🔴 Circular Import #{idx}")
            report.append("-" * 80)
            
            analysis = self.analyze_specific_cycle(cycle)
            
            report.append(f"Chain Length: {analysis['length']} modules")
            report.append("")
            report.append("Import Chain:")
            
            for detail in analysis['import_details']:
                from_module = detail['from'].split('.')[-1]
                to_module = detail['to'].split('.')[-1]
                report.append(f"   {from_module}")
                report.append(f"      ↓ imports")
                report.append(f"   {to_module}")
                report.append(f"      Line {detail['line']}: {detail['statement']}")
                report.append("")
            
            # Add the cycle closure
            report.append("   ⚠️ CREATES CYCLE - Returns to first module")
            report.append("")
            
            # Suggest fixes
            report.append("💡 Suggested Fixes:")
            report.append(self._suggest_fix(cycle))
            report.append("")
        
        report.append("=" * 80)
        report.append("SUMMARY")
        report.append("=" * 80)
        report.append(f"Total Circular Chains: {len(cycles)}")
        report.append(f"Average Chain Length: {sum(len(c)-1 for c in cycles) / len(cycles):.1f}")
        report.append("")
        report.append("🔧 Recommended Actions:")
        report.append("1. Implement Repository Pattern for data access")
        report.append("2. Use Dependency Injection instead of direct imports")
        report.append("3. Move shared code to a separate module")
        report.append("4. Consider using Protocol/ABC for interfaces")
        report.append("")
        
        return "\n".join(report)
    
    def _suggest_fix(self, cycle: List[str]) -> str:
        """Suggest how to fix a circular import"""
        fixes = []
        
        # Identify the type of cycle
        if 'knowledge_base_service' in ' '.join(cycle) and 'rule_engine' in ' '.join(cycle):
            fixes.append("   • Use ModelRepository pattern (already created)")
            fixes.append("   • Both services should import from repository, not each other")
            fixes.append("   • Example: from app.repositories.model_repository import get_model_repository")
        
        elif 'rule_engine_service' in ' '.join(cycle):
            fixes.append("   • RuleEngineService should delegate to RuleEngine, not import from it")
            fixes.append("   • Use composition: self.rule_engine = RuleEngine()")
        
        else:
            fixes.append("   • Extract shared functionality to a separate module")
            fixes.append("   • Use lazy imports (import inside functions)")
            fixes.append("   • Implement dependency injection")
        
        return "\n".join(fixes)
    
    def visualize_dependencies(self, output_file: str = "dependency_graph.dot"):
        """Generate a DOT file for visualization with Graphviz"""
        logger.info(f"📊 Generating dependency graph: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("digraph Dependencies {\n")
            f.write("    rankdir=LR;\n")
            f.write("    node [shape=box];\n")
            f.write("\n")
            
            # Add nodes
            for module in self.import_graph.keys():
                short_name = module.split('.')[-1]
                f.write(f'    "{short_name}";\n')
            
            f.write("\n")
            
            # Add edges
            for from_module, to_modules in self.import_graph.items():
                from_short = from_module.split('.')[-1]
                for to_module in to_modules:
                    to_short = to_module.split('.')[-1]
                    
                    # Highlight circular edges in red
                    if to_module in self.import_graph and from_module in self.import_graph[to_module]:
                        f.write(f'    "{from_short}" -> "{to_short}" [color=red, penwidth=2.0];\n')
                    else:
                        f.write(f'    "{from_short}" -> "{to_short}";\n')
            
            f.write("}\n")
        
        logger.info(f"   ✅ Graph saved to {output_file}")
        logger.info("   💡 Visualize with: dot -Tpng dependency_graph.dot -o deps.png")


def main():
    """Run circular import detection"""
    print("\n" + "🔍" * 40)
    print("CIRCULAR IMPORT DETECTOR")
    print("🔍" * 40 + "\n")
    
    # Detect in app directory
    detector = CircularImportDetector(root_dir="app")
    
    # Scan all files
    detector.scan_directory()
    
    # Find circular imports
    cycles = detector.find_circular_imports()
    
    # Generate report
    report = detector.generate_report(cycles)
    print(report)
    
    # Save report to file
    with open("circular_imports_report.txt", 'w') as f:
        f.write(report)
    
    print("📄 Full report saved to: circular_imports_report.txt")
    
    # Generate visualization
    try:
        detector.visualize_dependencies()
    except Exception as e:
        print(f"⚠️ Could not generate graph: {e}")
    
    print("\n" + "✅" * 40)
    print(f"DETECTION COMPLETE - Found {len(cycles)} circular import(s)")
    print("✅" * 40 + "\n")
    
    return len(cycles)


if __name__ == "__main__":
    import sys
    num_issues = main()
    sys.exit(num_issues)  # Exit with error code if issues found