"""
ignore this file because it give us thread issue.
use fixed_kb_service.py instead
"""



# app/services/knowledge_base_services/core/fixed_kb_service.py
# Enhanced Supply Chain Knowledge Base Service with Caching, Error Handling, and Performance Optimizations
import json
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory (3 levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    # Try importing from the correct path
    from app.knowledge_base.relational_kb.sqlite_manager import SQLiteManager
    logger.info("✅ Successfully imported SQLiteManager from app.knowledge_base.relational_kb")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.info("Trying direct import...")
    # Fallback: try direct import
    from app.knowledge_base.relational_kb.sqlite_manager import SQLiteManager


class SupplyChainService:
    """
    Enhanced service class for supply chain forecasting operations.
    Includes error handling, caching, and performance optimizations.
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        """Initialize service with error handling and caching"""
        self.logger = logging.getLogger(f"{__name__}.SupplyChainService")
        try:
            self.db = SQLiteManager(db_path)
            self._model_cache = None
            self._schema_cache = {}
            self._rules_cache = None
            self.logger.info(f"✅ Service initialized successfully with database: {db_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize service: {e}")
            raise
    
    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Get model by ID with error handling"""
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM ML_Models WHERE model_id = ?", (model_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    def get_all_models(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get all active models with optional caching"""
        if use_cache and self._model_cache is not None:
            self.logger.debug("Returning models from cache")
            return self._model_cache
        
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM ML_Models WHERE is_active = TRUE"
            )
            models = [dict(row) for row in cursor.fetchall()]
            
            if use_cache:
                self._model_cache = models
                self.logger.info(f"Cached {len(models)} active models")
            
            return models
        except Exception as e:
            self.logger.error(f"Error getting all models: {e}")
            return []
    

    
    def get_dataset_schema(self, dataset_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get complete dataset schema with columns and caching"""
        cache_key = f"schema_{dataset_name}"
        
        if use_cache and cache_key in self._schema_cache:
            self.logger.debug(f"Returning schema for '{dataset_name}' from cache")
            return self._schema_cache[cache_key]
        
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM Dataset_Schemas WHERE dataset_name = ?", (dataset_name,)
            )
            row = cursor.fetchone()
            
            if not row:
                self.logger.warning(f"Dataset schema not found: {dataset_name}")
                return None
            
            schema_data = dict(row)
            
            # Get related columns
            cursor = self.db.conn.execute(
                "SELECT * FROM Column_Definitions WHERE dataset_id = ?", 
                (schema_data['dataset_id'],)
            )
            
            schema_data['columns'] = [dict(col_row) for col_row in cursor.fetchall()]
            
            if use_cache:
                self._schema_cache[cache_key] = schema_data
                self.logger.debug(f"Cached schema for '{dataset_name}'")
            
            return schema_data
        except Exception as e:
            self.logger.error(f"Error getting schema for '{dataset_name}': {e}")
            return None
    
    def validate_dataset(self, dataset_name: str, provided_columns: List[str]) -> Dict[str, Any]:
        """Validate dataset against schema requirements with enhanced error handling"""
        try:
            schema = self.get_dataset_schema(dataset_name)
            if not schema:
                return {
                    "valid": False, 
                    "errors": [f"Dataset schema '{dataset_name}' not found"],
                    "schema_name": dataset_name,
                    "required_columns": [],
                    "provided_columns": provided_columns
                }
            
            required_columns = [
                col['column_name'] for col in schema['columns'] 
                if col['requirement_level'] == 'required'
            ]
            
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
                    if col['requirement_level'] == 'recommended'
                ]
            }
            
            if result["valid"]:
                self.logger.info(f"✅ Dataset validation passed for '{dataset_name}'")
            else:
                self.logger.warning(f"❌ Dataset validation failed for '{dataset_name}': {errors}")
            
            return result
        except Exception as e:
            self.logger.error(f"Error during dataset validation for '{dataset_name}': {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "schema_name": dataset_name,
                "required_columns": [],
                "provided_columns": provided_columns
            }
    
    def get_active_rules(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get all active rules with caching"""
        if use_cache and self._rules_cache is not None:
            return self._rules_cache
        
        try:
            cursor = self.db.conn.execute("""
                SELECT * FROM Rules 
                WHERE is_active = TRUE 
                ORDER BY priority DESC, rule_id
            """)
            rules = [dict(rule) for rule in cursor.fetchall()]
            
            if use_cache:
                self._rules_cache = rules
            
            return rules
        except Exception as e:
            self.logger.error(f"Error getting active rules: {e}")
            return []
    
    def get_suitable_models(self, available_features: List[str], target_variable: str, 
                           sort_by_performance: bool = True) -> List[Dict[str, Any]]:
        """Find models that match available features and target, with performance ranking"""
        try:
            suitable_models = []
            
            for model in self.get_all_models():
                required_features = json.loads(model['required_features'])
                
                if (all(feature in available_features for feature in required_features) and
                    model['target_variable'] == target_variable):
                    suitable_models.append(model)
            
            if sort_by_performance:
                # Sort by MAPE (lower is better), with fallback for missing metrics
                suitable_models.sort(
                    key=lambda x: json.loads(x['performance_metrics']).get('MAPE', float('inf'))
                )
                self.logger.info(f"Found {len(suitable_models)} suitable models, sorted by performance")
            else:
                self.logger.info(f"Found {len(suitable_models)} suitable models")
            
            return suitable_models
        except Exception as e:
            self.logger.error(f"Error finding suitable models: {e}")
            return []
    
    def create_forecast_run(self, model_id: int, input_schema: str, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new forecast run with transaction safety"""
        try:
            self.db.conn.execute("""
                INSERT INTO Forecast_Runs 
                (model_id, input_dataset_schema, forecast_config, status, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (model_id, input_schema, json.dumps(config), 'pending', datetime.now()))
            
            self.db.conn.commit()
            
            cursor = self.db.conn.execute("SELECT * FROM Forecast_Runs WHERE rowid = last_insert_rowid()")
            result = dict(cursor.fetchone())
            
            self.logger.info(f"✅ Created forecast run {result.get('run_id')} for model {model_id}")
            return result
            
        except Exception as e:
            self.db.conn.rollback()
            self.logger.error(f"❌ Failed to create forecast run: {e}")
            raise
    
    def clear_cache(self):
        """Clear all cached data"""
        self._model_cache = None
        self._schema_cache = {}
        self._rules_cache = None
        self.logger.info("✅ Cleared all service caches")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
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
                metrics = json.loads(model['performance_metrics'])
                mape = metrics.get('MAPE')
                
                if model_type not in stats['model_types']:
                    stats['model_types'][model_type] = 0
                stats['model_types'][model_type] += 1
                
                if mape is not None:
                    mape_values.append(mape)
            
            if mape_values:
                stats['average_performance'] = {
                    'MAPE': sum(mape_values) / len(mape_values),
                    'min_MAPE': min(mape_values),
                    'max_MAPE': max(mape_values)
                }
            
            self.logger.info(f"Generated performance stats for {len(models)} models")
            return stats
        except Exception as e:
            self.logger.error(f"Error generating performance stats: {e}")
            return {}
    
    def close(self):
        """Close database connection and cleanup"""
        try:
            self.clear_cache()
            self.db.close()
            self.logger.info("✅ Service closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing service: {e}")


def run_enhanced_demo():
    """Enhanced demo with better error handling and logging"""
    print("🚀 ENHANCED SUPPLY CHAIN FORECASTING SYSTEM")
    print("=" * 50)
    
    try:
        # Use in-memory database for testing
        service = SupplyChainService(":memory:")
        
        print("✅ Service initialized successfully!")
        
        # 1. Show all models with caching
        print("\n📊 ACTIVE MODELS (WITH CACHING):")
        print("-" * 30)
        models = service.get_all_models()
        print(f"Found {len(models)} models:")
        for model in models:
            metrics = json.loads(model['performance_metrics'])
            print(f"  • {model['model_name']} ({model['model_type']})")
            print(f"    Target: {model['target_variable']}")
            print(f"    MAPE: {metrics.get('MAPE', 'N/A')}")
        
        # 2. Test performance-based model selection
        print("\n🎯 PERFORMANCE-BASED MODEL SELECTION:")
        print("-" * 30)
        suitable_models = service.get_suitable_models(
            ["date", "demand", "price"], "demand", sort_by_performance=True
        )
        print(f"Found {len(suitable_models)} suitable models (sorted by performance):")
        for i, model in enumerate(suitable_models, 1):
            metrics = json.loads(model['performance_metrics'])
            print(f"  {i}. {model['model_name']} - MAPE: {metrics.get('MAPE', 'N/A')}")
        
        # 3. Test dataset validation with error cases
        print("\n✅ ENHANCED DATASET VALIDATION:")
        print("-" * 30)
        
        # Test with missing required columns
        validation = service.validate_dataset(
            "demand_forecasting", 
            ["date", "price"]  # Missing 'demand'
        )
        print(f"Validation: {'✅ PASS' if validation['valid'] else '❌ FAIL'}")
        if validation['errors']:
            for error in validation['errors']:
                print(f"  • {error}")
        
        # 4. Show performance statistics
        print("\n📈 PERFORMANCE STATISTICS:")
        print("-" * 30)
        stats = service.get_performance_stats()
        print(f"Total models: {stats['total_models']}")
        print(f"Model types: {stats['model_types']}")
        if stats['average_performance']:
            print(f"Average MAPE: {stats['average_performance']['MAPE']:.2f}%")
        
        # 5. Test cache functionality
        print("\n💾 CACHE FUNCTIONALITY:")
        print("-" * 30)
        print("First call (populates cache):")
        models_first = service.get_all_models(use_cache=True)
        print(f"Retrieved {len(models_first)} models")
        
        print("Second call (uses cache):")
        models_second = service.get_all_models(use_cache=True)
        print(f"Retrieved {len(models_second)} models (cached)")
        
        service.clear_cache()
        print("After cache clear:")
        models_cleared = service.get_all_models(use_cache=True)
        print(f"Retrieved {len(models_cleared)} models (cache repopulated)")
        
        service.close()
        print("\n🎉 ENHANCED DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error during enhanced demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_enhanced_demo()