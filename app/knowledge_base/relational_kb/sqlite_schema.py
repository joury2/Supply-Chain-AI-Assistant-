# sqlite_schema.py
# This module provides a repository class for interacting with the SQLite database
# defined in sqlite_schema.py. It includes methods for CRUD operations on dataset schemas,
# to manage the entire Relational Knowledge Base (KB). 
# Its primary purpose is to provide clean, high-level methods for Create, Read, Update, and Delete (CRUD) operations on all the database tables (dataset_schemas, ml_models, forecast_runs, etc.) without exposing the raw SQL queries to the rest of your application.

from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import json
class SQLiteRepository:
    def __init__(self, db_manager):
        self.db = db_manager
    
    # DATASET SCHEMAS
    def create_dataset_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        schema_id = f"schema_{uuid.uuid4().hex[:8]}"
        
        self.db.conn.execute("""
            INSERT INTO dataset_schemas (id, schema_name, description, min_rows, max_rows, required_columns)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            schema_id,
            schema_data['schema_name'],
            schema_data['description'],
            schema_data['min_rows'],
            schema_data.get('max_rows'),
            json.dumps(schema_data['required_columns'])
        ))
        
        self.db.conn.commit()
        return self.get_dataset_schema(schema_id)
    
    def get_dataset_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.db.conn.execute(
            "SELECT * FROM dataset_schemas WHERE id = ?", (schema_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        cursor = self.db.conn.execute("SELECT * FROM dataset_schemas")
        return [dict(row) for row in cursor.fetchall()]
    
    # ML MODELS
    def get_active_models(self) -> List[Dict[str, Any]]:
        cursor = self.db.conn.execute(
            "SELECT * FROM ml_models WHERE is_active = TRUE"
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        cursor = self.db.conn.execute(
            "SELECT * FROM ml_models WHERE model_type = ? AND is_active = TRUE", 
            (model_type,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_suitable_models(self, available_features: List[str], target_variable: str) -> List[Dict[str, Any]]:
        """Find models where required features are subset of available features"""
        all_models = self.get_active_models()
        suitable_models = []
        
        for model in all_models:
            # Parse JSON arrays
            required_features = json.loads(model['required_features'])
            
            # Check if all required features are available and target matches
            if (all(feature in available_features for feature in required_features) and
                model['target_variable'] == target_variable):
                suitable_models.append(model)
        
        # Sort by performance (MAPE lower is better)
        return sorted(
            suitable_models,
            key=lambda x: json.loads(x['performance_metrics']).get('MAPE', float('inf'))
        )
    
    # FORECAST RUNS
    def create_forecast_run(self, model_id: str, input_schema: str, config: Dict[str, Any]) -> Dict[str, Any]:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        forecast_id = f"forecast_{uuid.uuid4().hex[:8]}"
        
        self.db.conn.execute("""
            INSERT INTO forecast_runs (id, run_id, model_id, input_dataset_schema, forecast_config, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            forecast_id,
            run_id,
            model_id,
            input_schema,
            json.dumps(config),
            'pending'
        ))
        
        self.db.conn.commit()
        return self.get_forecast_run_by_id(forecast_id)
    
    def get_forecast_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.db.conn.execute(
            "SELECT * FROM forecast_runs WHERE id = ?", (run_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def update_forecast_results(self, run_id: str, results: Dict[str, Any], 
                              validation_issues: List[str] = None) -> Optional[Dict[str, Any]]:
        self.db.conn.execute("""
            UPDATE forecast_runs 
            SET results = ?, validation_issues = ?, status = 'completed', completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            json.dumps(results),
            json.dumps(validation_issues or []),
            run_id
        ))
        
        self.db.conn.commit()
        return self.get_forecast_run_by_id(run_id)
    
    # PERFORMANCE HISTORY
    def add_performance_record(self, model_id: str, metrics: Dict[str, float], 
                             dataset_schema: str, sample_size: int) -> Dict[str, Any]:
        perf_id = f"perf_{uuid.uuid4().hex[:8]}"
        
        self.db.conn.execute("""
            INSERT INTO model_performance_history (id, model_id, dataset_schema, metrics, training_date, sample_size)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            perf_id,
            model_id,
            dataset_schema,
            json.dumps(metrics),
            datetime.now().date().isoformat(),
            sample_size
        ))
        
        self.db.conn.commit()
        return self.get_performance_record(perf_id)
    
    def get_performance_record(self, perf_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.db.conn.execute(
            "SELECT * FROM model_performance_history WHERE id = ?", (perf_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # FEATURE IMPORTANCE
    def update_feature_importance(self, model_id: str, feature_scores: Dict[str, float],
                                calculation_method: str = "permutation") -> List[Dict[str, Any]]:
        # Delete existing features
        self.db.conn.execute(
            "DELETE FROM feature_importance WHERE model_id = ?", (model_id,)
        )
        
        # Insert new features
        for feature_name, score in feature_scores.items():
            feature_id = f"fi_{uuid.uuid4().hex[:8]}"
            self.db.conn.execute("""
                INSERT INTO feature_importance (id, model_id, feature_name, importance_score, calculation_method)
                VALUES (?, ?, ?, ?, ?)
            """, (feature_id, model_id, feature_name, score, calculation_method))
        
        self.db.conn.commit()
        return self.get_feature_importance(model_id)
    
    def get_feature_importance(self, model_id: str) -> List[Dict[str, Any]]:
        cursor = self.db.conn.execute(
            "SELECT * FROM feature_importance WHERE model_id = ? ORDER BY importance_score DESC",
            (model_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    

    