# app/services/knowledge_base_services/core/knowledge_base_service.py
"""
THREAD-SAFE Knowledge Base Service for Supply Chain Forecasting
FIXED VERSION: Uses thread-safe database connections for Streamlit compatibility

Key Changes:
1. Uses ThreadSafeDB instead of direct SQLite connections
2. Implements proper connection pooling per thread
3. Maintains same interface as original service
4. Adds comprehensive error handling
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class SupplyChainService:
    """
    THREAD-SAFE knowledge base service for supply chain forecasting operations.
    
    Why this fix is needed:
    - Streamlit runs each session in separate threads
    - SQLite connections cannot be shared between threads
    - Original service caused: "SQLite objects created in a thread can only be used in that same thread"
    
    Solution: Use ThreadSafeDB which creates separate connections per thread
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        """Initialize with thread-safe database connection"""
        try:
            # Import thread-safe database manager
            from app.services.knowledge_base_services.core.thread_safe_db import get_thread_safe_db
            
            # Get thread-safe database instance
            self.db = get_thread_safe_db(db_path)
            
            # Initialize caches
            self._model_cache = None
            self._schema_cache = {}
            self._rules_cache = None
            
            logger.info(f"✅ Thread-safe Knowledge Base Service initialized: {db_path}")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import thread-safe DB: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise
    
    def get_all_models(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get all active models - THREAD-SAFE VERSION
        
        Args:
            use_cache: Whether to use cached results (faster)
            
        Returns:
            List of active model dictionaries
        """
        # Return cached results if available
        if use_cache and self._model_cache is not None:
            logger.debug("📦 Returning models from cache")
            return self._model_cache
        
        try:
            # Use thread-safe database execution
            rows = self.db.execute("""
                SELECT model_id, model_name, model_type, model_path,
                       required_features, optional_features, target_variable,
                       performance_metrics, training_config, hyperparameters,
                       is_active, created_at
                FROM ML_Models 
                WHERE is_active = TRUE
                ORDER BY model_name
            """)
            
            # Convert to dictionaries
            models = []
            for row in rows:
                model_dict = {
                    'model_id': row[0],
                    'model_name': row[1],
                    'model_type': row[2],
                    'model_path': row[3],
                    'required_features': row[4],
                    'optional_features': row[5],
                    'target_variable': row[6],
                    'performance_metrics': row[7],
                    'training_config': row[8],
                    'hyperparameters': row[9],
                    'is_active': bool(row[10]),
                    'created_at': row[11]
                }
                models.append(model_dict)
            
            # Cache results if requested
            if use_cache:
                self._model_cache = models
                logger.info(f"📦 Cached {len(models)} active models")
            
            logger.info(f"✅ Retrieved {len(models)} active models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Error getting all models: {e}")
            return []
    
    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """
        Get specific model by ID - THREAD-SAFE VERSION
        """
        try:
            rows = self.db.execute(
                "SELECT * FROM ML_Models WHERE model_id = ? AND is_active = TRUE",
                (model_id,)
            )
            
            if rows:
                row = rows[0]
                return {
                    'model_id': row[0],
                    'model_name': row[1],
                    'model_type': row[2],
                    'model_path': row[3],
                    'required_features': row[4],
                    'optional_features': row[5],
                    'target_variable': row[6],
                    'performance_metrics': row[7],
                    'training_config': row[8],
                    'hyperparameters': row[9],
                    'is_active': bool(row[10]),
                    'created_at': row[11]
                }
            
            logger.warning(f"⚠️ Model not found: {model_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting model {model_id}: {e}")
            return None
    
    def get_dataset_schema(self, dataset_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get dataset schema with columns - THREAD-SAFE VERSION
        """
        cache_key = f"schema_{dataset_name}"
        
        # Return cached schema if available
        if use_cache and cache_key in self._schema_cache:
            logger.debug(f"📦 Returning cached schema: {dataset_name}")
            return self._schema_cache[cache_key]
        
        try:
            # Get dataset schema
            rows = self.db.execute(
                "SELECT * FROM Dataset_Schemas WHERE dataset_name = ?",
                (dataset_name,)
            )
            
            if not rows:
                logger.warning(f"⚠️ Dataset schema not found: {dataset_name}")
                return None
            
            row = rows[0]
            schema_data = {
                'dataset_id': row[0],
                'dataset_name': row[1],
                'source_path': row[2],
                'description': row[3],
                'min_rows': row[4],
                'created_at': row[5],
                'updated_at': row[6],
                'columns': []
            }
            
            # Get column definitions
            column_rows = self.db.execute(
                "SELECT * FROM Column_Definitions WHERE dataset_id = ?",
                (schema_data['dataset_id'],)
            )
            
            for col_row in column_rows:
                schema_data['columns'].append({
                    'column_id': col_row[0],
                    'dataset_id': col_row[1],
                    'column_name': col_row[2],
                    'requirement_level': col_row[3],
                    'data_type': col_row[4],
                    'role': col_row[5],
                    'description': col_row[6],
                    'validation_rules': col_row[7],
                    'created_at': col_row[8]
                })
            
            # Cache if requested
            if use_cache:
                self._schema_cache[cache_key] = schema_data
                logger.debug(f"📦 Cached schema: {dataset_name}")
            
            logger.info(f"✅ Retrieved schema: {dataset_name} with {len(schema_data['columns'])} columns")
            return schema_data
            
        except Exception as e:
            logger.error(f"❌ Error getting schema for '{dataset_name}': {e}")
            return None
    
    def validate_dataset(self, dataset_name: str, provided_columns: List[str]) -> Dict[str, Any]:
        """
        Validate dataset against schema - THREAD-SAFE VERSION
        """
        try:
            schema = self.get_dataset_schema(dataset_name, use_cache=True)
            
            if not schema:
                return {
                    "valid": False,
                    "errors": [f"Dataset schema '{dataset_name}' not found"],
                    "schema_name": dataset_name,
                    "required_columns": [],
                    "provided_columns": provided_columns
                }
            
            # Find required columns
            required_columns = [
                col['column_name'] for col in schema['columns']
                if col['requirement_level'] == 'required'
            ]
            
            # Check for missing required columns
            errors = []
            for req_col in required_columns:
                if req_col not in provided_columns:
                    errors.append(f"Missing required column: {req_col}")
            
            result = {
                "valid": len(errors) == 0,
                "errors": errors,
                "schema_name": schema['dataset_name'],
                "required_columns": required_columns,
                "provided_columns": provided_columns,
                "recommended_columns": [
                    col['column_name'] for col in schema['columns']
                    if col['requirement_level'] in ['recommended', 'optional']
                ]
            }
            
            if result["valid"]:
                logger.info(f"✅ Dataset validation passed: {dataset_name}")
            else:
                logger.warning(f"❌ Dataset validation failed: {dataset_name} - {errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Validation error for '{dataset_name}': {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "schema_name": dataset_name,
                "required_columns": [],
                "provided_columns": provided_columns
            }
    
    def get_active_rules(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get all active rules - THREAD-SAFE VERSION
        """
        if use_cache and self._rules_cache is not None:
            return self._rules_cache
        
        try:
            rows = self.db.execute("""
                SELECT rule_id, name, description, condition, action,
                       rule_type, priority, is_active, version, created_at
                FROM Rules 
                WHERE is_active = TRUE 
                ORDER BY priority DESC, rule_id
            """)
            
            rules = []
            for row in rows:
                rules.append({
                    'rule_id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'condition': row[3],
                    'action': row[4],
                    'rule_type': row[5],
                    'priority': row[6],
                    'is_active': bool(row[7]),
                    'version': row[8],
                    'created_at': row[9]
                })
            
            if use_cache:
                self._rules_cache = rules
            
            logger.info(f"✅ Retrieved {len(rules)} active rules")
            return rules
            
        except Exception as e:
            logger.error(f"❌ Error getting active rules: {e}")
            return []
    
    def get_suitable_models(self, available_features: List[str], target_variable: str, 
                           sort_by_performance: bool = True) -> List[Dict[str, Any]]:
        """
        Find models compatible with available features - THREAD-SAFE VERSION
        """
        try:
            suitable_models = []
            all_models = self.get_all_models(use_cache=True)
            
            for model in all_models:
                # Parse required features (handle JSON string)
                required_features = []
                try:
                    if isinstance(model['required_features'], str):
                        required_features = json.loads(model['required_features'])
                    else:
                        required_features = model['required_features']
                except:
                    logger.warning(f"⚠️ Could not parse features for {model['model_name']}")
                    continue
                
                # Check compatibility
                if (all(feature in available_features for feature in required_features) and
                    model['target_variable'] == target_variable):
                    suitable_models.append(model)
            
            # Sort by performance if requested
            if sort_by_performance and suitable_models:
                suitable_models.sort(
                    key=lambda x: self._get_model_performance_score(x)
                )
                logger.info(f"✅ Found {len(suitable_models)} suitable models (sorted by performance)")
            else:
                logger.info(f"✅ Found {len(suitable_models)} suitable models")
            
            return suitable_models
            
        except Exception as e:
            logger.error(f"❌ Error finding suitable models: {e}")
            return []
    
    def _get_model_performance_score(self, model: Dict[str, Any]) -> float:
        """Extract performance score from model metrics (lower is better)"""
        try:
            metrics = json.loads(model['performance_metrics'])
            # Prefer MAPE (Mean Absolute Percentage Error) - lower is better
            return metrics.get('MAPE', float('inf'))
        except:
            return float('inf')
    
    def create_forecast_run(self, model_id: int, input_schema: str, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new forecast run - THREAD-SAFE VERSION
        """
        try:
            # Use context manager for transaction safety
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO Forecast_Runs 
                    (model_id, input_dataset_schema, forecast_config, status, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    model_id, 
                    input_schema, 
                    json.dumps(config), 
                    'pending', 
                    datetime.now().isoformat()
                ))
                
                # Get the inserted row
                cursor.execute("SELECT * FROM Forecast_Runs WHERE run_id = last_insert_rowid()")
                row = cursor.fetchone()
                
                result = {
                    'run_id': row[0],
                    'model_id': row[1],
                    'input_dataset_schema': row[2],
                    'forecast_config': row[3],
                    'results': row[4],
                    'validation_issues': row[5],
                    'llm_interpretation': row[6],
                    'status': row[7],
                    'executed_at': row[8],
                    'created_at': row[9],
                    'completed_at': row[10]
                }
            
            logger.info(f"✅ Created forecast run {result['run_id']} for model {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to create forecast run: {e}")
            raise
    
    def clear_cache(self):
        """Clear all cached data"""
        self._model_cache = None
        self._schema_cache = {}
        self._rules_cache = None
        logger.info("✅ Cleared all service caches")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics - THREAD-SAFE VERSION"""
        try:
            models = self.get_all_models(use_cache=False)
            stats = {
                "total_models": len(models),
                "model_types": {},
                "average_performance": {},
                "best_performing": None
            }
            
            mape_values = []
            for model in models:
                model_type = model['model_type']
                stats['model_types'][model_type] = stats['model_types'].get(model_type, 0) + 1
                
                # Extract performance metrics
                try:
                    metrics = json.loads(model['performance_metrics'])
                    mape = metrics.get('MAPE')
                    if mape is not None:
                        mape_values.append(mape)
                except:
                    continue
            
            if mape_values:
                stats['average_performance'] = {
                    'MAPE': sum(mape_values) / len(mape_values),
                    'min_MAPE': min(mape_values),
                    'max_MAPE': max(mape_values),
                    'model_count': len(mape_values)
                }
            
            logger.info(f"📊 Generated performance stats for {len(models)} models")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error generating performance stats: {e}")
            return {}
    
    def close(self):
        """Close service and cleanup - THREAD-SAFE VERSION"""
        try:
            self.clear_cache()
            # ThreadSafeDB handles its own connection cleanup
            logger.info("✅ Knowledge Base Service closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing service: {e}")


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_thread_safe_service():
    """Demonstrate the thread-safe service"""
    print("🚀 THREAD-SAFE KNOWLEDGE BASE SERVICE DEMO")
    print("=" * 60)
    
    try:
        service = SupplyChainService("supply_chain.db")
        
        print("✅ Service initialized successfully!")
        
        # Test basic operations
        print("\n📊 Testing model retrieval...")
        models = service.get_all_models()
        print(f"Found {len(models)} active models")
        
        print("\n📋 Testing schema retrieval...")
        schema = service.get_dataset_schema("demand_forecasting")
        if schema:
            print(f"Schema '{schema['dataset_name']}' has {len(schema['columns'])} columns")
        
        print("\n✅ Testing dataset validation...")
        validation = service.validate_dataset(
            "demand_forecasting", 
            ["date", "demand", "price"]
        )
        print(f"Validation: {'PASS' if validation['valid'] else 'FAIL'}")
        
        print("\n📈 Testing performance stats...")
        stats = service.get_performance_stats()
        print(f"Total models: {stats['total_models']}")
        print(f"Model types: {stats['model_types']}")
        
        service.close()
        print("\n🎉 Thread-safe service demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_thread_safe_service()