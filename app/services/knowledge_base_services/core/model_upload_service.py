# app/services/knowledge_base_services/model_upload_service.py
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ModelUploadService:
    """
    Handles automated model upload with rule generation
    """
    
    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        from app.services.knowledge_base_services.core.rule_generator_service import RuleGeneratorService
        self.rule_generator = RuleGeneratorService(db_connection)
    
    def upload_model(self, model_path: str, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload a model and automatically generate rules for it
        """
        logger.info(f"📤 Uploading model: {model_metadata.get('model_name', 'Unknown')}")
        
        # Validate model file exists
        if not os.path.exists(model_path):
            return {'success': False, 'error': f'Model file not found: {model_path}'}
        
        try:
            # Step 1: Add model to database
            model_id = self._add_model_to_database(model_path, model_metadata)
            
            # Step 2: Generate automatic rules
            rule_result = self.rule_generator.generate_rules_for_model(model_metadata)
            
            # Step 3: Update model with generated rule info
            self._update_model_with_rules(model_id, rule_result)
            
            return {
                'success': True,
                'model_id': model_id,
                'model_name': model_metadata['model_name'],
                'generated_rules': len(rule_result['generated_rules']),
                'message': f"Model uploaded successfully with {len(rule_result['generated_rules'])} auto-generated rules"
            }
            
        except Exception as e:
            logger.error(f"❌ Model upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _add_model_to_database(self, model_path: str, metadata: Dict[str, Any]) -> int:
        """Add model to ML_Models table"""
        cursor = self.db_connection.execute("""
            INSERT INTO ML_Models 
            (model_name, model_type, model_path, required_features, target_variable, 
             performance_metrics, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata['model_name'],
            metadata.get('model_type', 'unknown'),
            model_path,
            json.dumps(metadata.get('required_features', [])),
            metadata.get('target_variable', 'unknown'),
            json.dumps(metadata.get('performance_metrics', {})),
            True,
            datetime.now().isoformat()
        ))
        
        self.db_connection.commit()
        return cursor.lastrowid
    
    def _update_model_with_rules(self, model_id: int, rule_result: Dict[str, Any]):
        """Update model with information about generated rules"""
        # Could store rule references in model metadata if needed
        logger.info(f"📝 Model {model_id} now has {rule_result['generated_rules']} auto-generated rules")