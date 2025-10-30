# app/services/knowledge_base_services/utils/update_imports.py
import os
import re

def update_imports_in_file(file_path, old_import, new_import):
    """Update import statements in a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace import statements
    updated_content = content.replace(old_import, new_import)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Updated imports in {os.path.basename(file_path)}")

def main():
    """Update imports for all reorganized files"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Files to update and their new import paths
    updates = [
        # Core services
        ('core/rule_engine_service.py', 
         'from app.services.knowledge_base_services.knowledge_base_service import',
         'from ..core.knowledge_base_service import'),
        
        ('core/model_upload_service.py',
         'from app.services.knowledge_base_services.rule_generator_service import', 
         'from ..core.rule_generator_service import'),
        
        ('core/forecasting_report_service.py',
         'from app.services.knowledge_base_services.rule_engine_service import',
         'from ..core.rule_engine_service import'),
        
        # Test files
        ('testing/test_simple_models.py',
         'from app.services.knowledge_base_services.rule_engine_service import',
         'from ..core.rule_engine_service import'),
        
        ('testing/test_with_actual_models.py', 
         'from app.services.knowledge_base_services.rule_engine_service import',
         'from ..core.rule_engine_service import'),
        
        ('testing/test_with_actual_models.py',
         'from app.services.knowledge_base_services.knowledge_base_service import',
         'from ..core.knowledge_base_service import'),
        
        ('testing/test_automation.py',
         'from app.services.knowledge_base_services.knowledge_base_service import',
         'from ..core.knowledge_base_service import'),
        
        ('testing/test_automation.py',
         'from app.services.knowledge_base_services.rule_engine_service import', 
         'from ..core.rule_engine_service import'),
        
        ('testing/test_automation.py',
         'from app.services.knowledge_base_services.model_upload_service import',
         'from ..core.model_upload_service import'),
        
        ('testing/integration_test.py',
         'from app.services.knowledge_base_services.rule_engine_service import',
         'from ..core.rule_engine_service import'),
        
        ('testing/integration_test.py',
         'from app.services.knowledge_base_services.forecasting_report_service import',
         'from ..core.forecasting_report_service import'),
        
        # Debug files
        ('debugging/debug_supply_chain.py',
         'from app.services.knowledge_base_services.knowledge_base_service import',
         'from ..core.knowledge_base_service import'),
        
        ('debugging/debug_supply_chain.py',
         'from app.services.knowledge_base_services.rule_engine_service import',
         'from ..core.rule_engine_service import'),
        
        ('debugging/quick_test.py',
         'from app.services.knowledge_base_services.rule_engine_service import',
         'from ..core.rule_engine_service import'),
    ]
    
    for file_rel_path, old_import, new_import in updates:
        file_path = os.path.join(base_dir, file_rel_path)
        if os.path.exists(file_path):
            update_imports_in_file(file_path, old_import, new_import)
        else:
            print(f"⚠️  File not found: {file_rel_path}")

if __name__ == "__main__":
    main()