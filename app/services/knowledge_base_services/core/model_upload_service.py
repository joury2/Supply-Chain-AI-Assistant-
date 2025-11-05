# app/services/knowledge_base_services/core/model_upload_service.py
"""
Fixed Model Upload Service
Properly uploads models and metadata to database
"""
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelUploadService:
    """
    Handles automated model upload with rule generation
    """
    
    def __init__(self, db_connection=None):
        """
        Initialize service
        
        Args:
            db_connection: SQLite connection (if None, will create one)
        """
        if db_connection is None:
            import sqlite3
            self.db_connection = sqlite3.connect('supply_chain.db')
            self.owns_connection = True
        else:
            self.db_connection = db_connection
            self.owns_connection = False
        
        logger.info("📦 Model Upload Service initialized")
    
    def upload_model(
        self, 
        model_path: str, 
        model_metadata: Dict[str, Any],
        generate_rules: bool = True
    ) -> Dict[str, Any]:
        """
        Upload a model and automatically generate rules for it
        
        Args:
            model_path: Path to model file
            model_metadata: Model metadata dict
            generate_rules: Whether to auto-generate rules
        
        Returns:
            Upload result with success status
        """
        logger.info(f"📤 Uploading model: {model_metadata.get('model_name', 'Unknown')}")
        
        # Validate model file exists
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return {
                'success': False, 
                'error': f'Model file not found: {model_path}'
            }
        
        try:
            # Step 1: Validate metadata
            validation = self._validate_metadata(model_metadata)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Invalid metadata: {', '.join(validation['errors'])}"
                }
            
            # Step 2: Check if model already exists
            existing = self._check_existing_model(model_metadata['model_name'])
            if existing:
                logger.warning(f"⚠️ Model {model_metadata['model_name']} already exists")
                return {
                    'success': False,
                    'error': f"Model {model_metadata['model_name']} already exists. Use update instead.",
                    'existing_model_id': existing
                }
            
            # Step 3: Add model to database
            model_id = self._add_model_to_database(model_path, model_metadata)
            logger.info(f"✅ Model added to database with ID: {model_id}")
            
            # Step 4: Generate automatic rules (if requested)
            generated_rules = []
            if generate_rules:
                try:
                    from app.services.knowledge_base_services.core.rule_generator_service import RuleGeneratorService
                    rule_generator = RuleGeneratorService(self.db_connection)
                    
                    rule_result = rule_generator.generate_rules_for_model(model_metadata)
                    generated_rules = rule_result.get('generated_rules', [])
                    
                    # Update model with rule info
                    self._update_model_with_rules(model_id, rule_result)
                    logger.info(f"✅ Generated {len(generated_rules)} rules")
                    
                except ImportError:
                    logger.warning("⚠️ Rule generator not available - skipping rule generation")
                except Exception as e:
                    logger.warning(f"⚠️ Rule generation failed: {e}")
            
            # Step 5: Verify upload
            uploaded_model = self._get_model_by_id(model_id)
            
            return {
                'success': True,
                'model_id': model_id,
                'model_name': model_metadata['model_name'],
                'model_type': model_metadata.get('model_type', 'unknown'),
                'generated_rules': len(generated_rules),
                'model_details': uploaded_model,
                'message': f"Model uploaded successfully with {len(generated_rules)} auto-generated rules"
            }
            
        except Exception as e:
            logger.error(f"❌ Model upload failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False, 
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model metadata"""
        errors = []
        
        # Required fields
        required = ['model_name', 'model_type']
        for field in required:
            if field not in metadata or not metadata[field]:
                errors.append(f"Missing required field: {field}")
        
        # Model type validation
        valid_types = ['prophet', 'arima', 'sarima', 'xgboost', 'lightgbm', 'lstm', 'other']
        if metadata.get('model_type') and metadata['model_type'].lower() not in valid_types:
            logger.warning(f"⚠️ Unknown model type: {metadata['model_type']}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _check_existing_model(self, model_name: str) -> Optional[int]:
        """Check if model already exists"""
        try:
            cursor = self.db_connection.execute(
                "SELECT model_id FROM ML_Models WHERE model_name = ?",
                (model_name,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error checking existing model: {e}")
            return None
    
    def _add_model_to_database(self, model_path: str, metadata: Dict[str, Any]) -> int:
        """Add model to ML_Models table"""
        try:
            # Prepare data
            model_name = metadata['model_name']
            model_type = metadata.get('model_type', 'unknown')
            required_features = json.dumps(metadata.get('required_features', []))
            target_variable = metadata.get('target_variable', 'unknown')
            performance_metrics = json.dumps(metadata.get('performance_metrics', {}))
            
            # Additional metadata
            metadata_json = json.dumps({
                'description': metadata.get('description', ''),
                'training_date': metadata.get('training_date', datetime.now().isoformat()),
                'framework': metadata.get('framework', 'unknown'),
                'version': metadata.get('version', '1.0'),
                'tags': metadata.get('tags', [])
            })
            
            # Insert into database
            cursor = self.db_connection.execute("""
                INSERT INTO ML_Models 
                (model_name, model_type, model_path, required_features, target_variable, 
                 performance_metrics, metadata, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                model_type,
                model_path,
                required_features,
                target_variable,
                performance_metrics,
                metadata_json,
                True,  # is_active
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            self.db_connection.commit()
            model_id = cursor.lastrowid
            
            logger.info(f"✅ Model inserted with ID: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Database insert failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _update_model_with_rules(self, model_id: int, rule_result: Dict[str, Any]):
        """Update model with information about generated rules"""
        try:
            # Get current metadata
            cursor = self.db_connection.execute(
                "SELECT metadata FROM ML_Models WHERE model_id = ?",
                (model_id,)
            )
            result = cursor.fetchone()
            
            if result and result[0]:
                metadata = json.loads(result[0])
            else:
                metadata = {}
            
            # Add rule information
            metadata['auto_generated_rules'] = {
                'count': len(rule_result.get('generated_rules', [])),
                'generated_at': datetime.now().isoformat(),
                'rules': rule_result.get('generated_rules', [])
            }
            
            # Update database
            self.db_connection.execute(
                "UPDATE ML_Models SET metadata = ?, updated_at = ? WHERE model_id = ?",
                (json.dumps(metadata), datetime.now().isoformat(), model_id)
            )
            self.db_connection.commit()
            
            logger.info(f"✅ Model {model_id} updated with rule information")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to update model with rules: {e}")
    
    def _get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve model details by ID"""
        try:
            cursor = self.db_connection.execute("""
                SELECT model_id, model_name, model_type, model_path, 
                       required_features, target_variable, performance_metrics,
                       metadata, is_active, created_at, updated_at
                FROM ML_Models 
                WHERE model_id = ?
            """, (model_id,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'model_id': result[0],
                    'model_name': result[1],
                    'model_type': result[2],
                    'model_path': result[3],
                    'required_features': json.loads(result[4]) if result[4] else [],
                    'target_variable': result[5],
                    'performance_metrics': json.loads(result[6]) if result[6] else {},
                    'metadata': json.loads(result[7]) if result[7] else {},
                    'is_active': result[8],
                    'created_at': result[9],
                    'updated_at': result[10]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving model: {e}")
            return None
    
    def list_models(self) -> list:
        """List all models in database"""
        try:
            cursor = self.db_connection.execute("""
                SELECT model_id, model_name, model_type, is_active, created_at
                FROM ML_Models
                ORDER BY created_at DESC
            """)
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    'model_id': row[0],
                    'model_name': row[1],
                    'model_type': row[2],
                    'is_active': row[3],
                    'created_at': row[4]
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def delete_model(self, model_id: int) -> Dict[str, Any]:
        """Delete a model from database"""
        try:
            # Get model details first
            model = self._get_model_by_id(model_id)
            if not model:
                return {'success': False, 'error': f'Model {model_id} not found'}
            
            # Delete from database
            self.db_connection.execute(
                "DELETE FROM ML_Models WHERE model_id = ?",
                (model_id,)
            )
            self.db_connection.commit()
            
            logger.info(f"✅ Model {model_id} deleted")
            
            return {
                'success': True,
                'message': f"Model {model['model_name']} deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close database connection if owned"""
        if self.owns_connection and self.db_connection:
            self.db_connection.close()


# CLI tool for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload ML model to knowledge base')
    parser.add_argument('--model-path', required=True, help='Path to model file')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--model-type', required=True, help='Model type (prophet, arima, etc.)')
    parser.add_argument('--target', default='demand', help='Target variable')
    parser.add_argument('--features', nargs='+', help='Required features')
    parser.add_argument('--no-rules', action='store_true', help='Skip rule generation')
    
    args = parser.parse_args()
    
    # Create metadata
    metadata = {
        'model_name': args.model_name,
        'model_type': args.model_type,
        'target_variable': args.target,
        'required_features': args.features or [],
        'description': f'{args.model_type} model for {args.target}',
        'framework': args.model_type,
        'training_date': datetime.now().isoformat()
    }
    
    # Upload model
    service = ModelUploadService()
    
    print(f"\n📤 Uploading model: {args.model_name}")
    print(f"📍 Path: {args.model_path}")
    print(f"🏷️  Type: {args.model_type}\n")
    
    result = service.upload_model(
        model_path=args.model_path,
        model_metadata=metadata,
        generate_rules=not args.no_rules
    )
    
    if result['success']:
        print(f"✅ {result['message']}")
        print(f"\n📊 Model Details:")
        print(json.dumps(result['model_details'], indent=2))
    else:
        print(f"❌ Upload failed: {result['error']}")
    
    service.close()