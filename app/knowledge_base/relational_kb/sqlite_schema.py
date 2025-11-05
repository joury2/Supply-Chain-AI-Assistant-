# app/knowledge_base/relational_kb/sqlite_schema.py
"""
FIXED VERSION - SQLiteRepository that matches your actual database schema
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class SQLiteRepository:
    def __init__(self, db_manager):
        self.db = db_manager
    
    # ============================================================================
    # DATASET SCHEMAS - FIXED to match actual table structure
    # ============================================================================
    
    def create_dataset_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new dataset schema - FIXED to use actual column names"""
        try:
            cursor = self.db.conn.execute("""
                INSERT INTO Dataset_Schemas (dataset_name, description, min_rows, source_path)
                VALUES (?, ?, ?, ?)
            """, (
                schema_data['dataset_name'],
                schema_data.get('description', ''),
                schema_data.get('min_rows', 30),
                schema_data.get('source_path', '')
            ))
            self.db.conn.commit()
            
            dataset_id = cursor.lastrowid
            logger.info(f"✅ Created dataset schema: {schema_data['dataset_name']} (ID: {dataset_id})")
            return self.get_dataset_schema(dataset_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to create dataset schema: {e}")
            self.db.conn.rollback()
            raise
    
    def get_dataset_schema(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Get dataset schema by ID - FIXED to use dataset_id"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM Dataset_Schemas WHERE dataset_id = ?", (dataset_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Failed to get dataset schema {dataset_id}: {e}")
            return None
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all dataset schemas - FIXED table name"""
        try:
            cursor = self.db.conn.execute("SELECT * FROM Dataset_Schemas")
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get all schemas: {e}")
            return []
    
    # ============================================================================
    # ML MODELS - FIXED to match actual table structure  
    # ============================================================================
    
    def get_active_models(self) -> List[Dict[str, Any]]:
        """Get all active ML models"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM ML_Models WHERE is_active = TRUE"
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get active models: {e}")
            return []
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Get models by type"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM ML_Models WHERE model_type = ? AND is_active = TRUE", 
                (model_type,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get models by type {model_type}: {e}")
            return []
    
    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """CRITICAL MISSING METHOD - Get model by ID"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM ML_Models WHERE model_id = ?", (model_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Failed to get model {model_id}: {e}")
            return None
    
    def get_suitable_models(self, available_features: List[str], target_variable: str) -> List[Dict[str, Any]]:
        """Find models where required features are subset of available features"""
        try:
            all_models = self.get_active_models()
            suitable_models = []
            
            for model in all_models:
                # Parse JSON arrays with error handling
                try:
                    required_features = json.loads(model['required_features'])
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"⚠️ Invalid required_features for model {model['model_name']}")
                    continue
                
                # Check if all required features are available and target matches
                if (all(feature in available_features for feature in required_features) and
                    model['target_variable'] == target_variable):
                    suitable_models.append(model)
            
            # Sort by performance (MAPE lower is better)
            return sorted(
                suitable_models,
                key=lambda x: json.loads(x.get('performance_metrics', '{}')).get('MAPE', float('inf'))
            )
        except Exception as e:
            logger.error(f"❌ Failed to find suitable models: {e}")
            return []
    
    # ============================================================================
    # RULES - FIXED to match actual table structure
    # ============================================================================
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """CRITICAL MISSING METHOD - Get all active business rules"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM Rules WHERE is_active = TRUE ORDER BY priority"
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get active rules: {e}")
            return []
    
    def get_rules_by_type(self, rule_type: str) -> List[Dict[str, Any]]:
        """Get rules by type"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM Rules WHERE rule_type = ? AND is_active = TRUE ORDER BY priority",
                (rule_type,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get rules by type {rule_type}: {e}")
            return []
    
    # ============================================================================
    # FORECAST RUNS - FIXED to match actual table structure
    # ============================================================================
    
    def create_forecast_run(self, model_id: int, input_schema: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new forecast run - FIXED to use actual column names"""
        try:
            cursor = self.db.conn.execute("""
                INSERT INTO Forecast_Runs (model_id, input_dataset_schema, forecast_config, status)
                VALUES (?, ?, ?, ?)
            """, (
                model_id,
                input_schema,
                json.dumps(config),
                'pending'
            ))
            self.db.conn.commit()
            
            run_id = cursor.lastrowid
            logger.info(f"✅ Created forecast run: {run_id} for model {model_id}")
            return self.get_forecast_run_by_id(run_id)
        except Exception as e:
            logger.error(f"❌ Failed to create forecast run: {e}")
            self.db.conn.rollback()
            raise
    
    def get_forecast_run_by_id(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get forecast run by ID - FIXED parameter name"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM Forecast_Runs WHERE run_id = ?", (run_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Failed to get forecast run {run_id}: {e}")
            return None
    
    def update_forecast_results(self, run_id: int, results: Dict[str, Any], 
                              validation_issues: List[str] = None) -> Optional[Dict[str, Any]]:
        """Update forecast results - FIXED parameter name"""
        try:
            self.db.conn.execute("""
                UPDATE Forecast_Runs 
                SET results = ?, validation_issues = ?, status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """, (
                json.dumps(results),
                json.dumps(validation_issues or []),
                run_id
            ))
            self.db.conn.commit()
            
            logger.info(f"✅ Updated forecast results for run: {run_id}")
            return self.get_forecast_run_by_id(run_id)
        except Exception as e:
            logger.error(f"❌ Failed to update forecast results for run {run_id}: {e}")
            self.db.conn.rollback()
            return None
    
    def get_recent_forecast_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent forecast runs"""
        try:
            cursor = self.db.conn.execute("""
                SELECT * FROM Forecast_Runs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get recent forecast runs: {e}")
            return []
    
    # ============================================================================
    # PERFORMANCE HISTORY - FIXED to match actual table structure
    # ============================================================================
    
    def add_performance_record(self, model_id: int, metrics: Dict[str, float], 
                             dataset_schema: str, sample_size: int) -> Dict[str, Any]:
        """Add performance record - FIXED to use actual column names"""
        try:
            cursor = self.db.conn.execute("""
                INSERT INTO Model_Performance_History (model_id, dataset_schema, metrics, sample_size, training_date)
                VALUES (?, ?, ?, ?, ?)
            """, (
                model_id,
                dataset_schema,
                json.dumps(metrics),
                sample_size,
                datetime.now().date().isoformat()
            ))
            self.db.conn.commit()
            
            record_id = cursor.lastrowid
            logger.info(f"✅ Added performance record: {record_id} for model {model_id}")
            return self.get_performance_record(record_id)
        except Exception as e:
            logger.error(f"❌ Failed to add performance record: {e}")
            self.db.conn.rollback()
            raise
    
    def get_performance_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Get performance record by ID - FIXED parameter name"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM Model_Performance_History WHERE record_id = ?", (record_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Failed to get performance record {record_id}: {e}")
            return None
    
    def get_model_performance_history(self, model_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get performance history for a model"""
        try:
            cursor = self.db.conn.execute("""
                SELECT * FROM Model_Performance_History 
                WHERE model_id = ? 
                ORDER BY training_date DESC 
                LIMIT ?
            """, (model_id, limit))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get performance history for model {model_id}: {e}")
            return []
    
    # ============================================================================
    # COLUMN DEFINITIONS - NEW methods for column management
    # ============================================================================
    
    def get_column_definitions(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Get column definitions for a dataset"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM Column_Definitions WHERE dataset_id = ? ORDER BY column_id",
                (dataset_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get column definitions for dataset {dataset_id}: {e}")
            return []
    
    def get_required_columns(self, dataset_id: int) -> List[str]:
        """Get required column names for a dataset"""
        try:
            cursor = self.db.conn.execute("""
                SELECT column_name FROM Column_Definitions 
                WHERE dataset_id = ? AND requirement_level = 'required'
            """, (dataset_id,))
            return [row['column_name'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get required columns for dataset {dataset_id}: {e}")
            return []
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            tables = ['Dataset_Schemas', 'Column_Definitions', 'Rules', 
                     'ML_Models', 'Model_Performance_History', 'Forecast_Runs']
            
            stats = {}
            for table in tables:
                cursor = self.db.conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[table] = cursor.fetchone()['count']
            
            return {
                'status': 'healthy',
                'table_counts': stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_connection(self) -> bool:
        """Test database connection and basic operations"""
        try:
            # Test basic query
            cursor = self.db.conn.execute("SELECT 1 as test")
            result = cursor.fetchone()
            return result['test'] == 1
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False