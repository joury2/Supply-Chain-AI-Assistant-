# app/services/model_serving/model_registry_service.py
import json
import os
import glob
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Get the project root directory
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"🔍 Project root: {project_root}")

# Fallback SQLiteManager
class SQLiteManager:
    def __init__(self, db_path: str = "supply_chain.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        print(f"✅ Connected to database: {self.db_path}")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        if self.conn:
            self.conn.close()

class ModelRegistryService:
    """
    Registry service that works with your actual file naming pattern
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        self.db = SQLiteManager(db_path)
        self.models_base_path = "models/forecasting/lightgbm"
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models using your actual file names"""
        models = []
        
        model_dirs = [
            "monthly_sales_forecaster",
            "daily_sales_forecaster"
        ]
        
        for model_dir in model_dirs:
            model_path = os.path.join(self.models_base_path, model_dir)
            if os.path.exists(model_path):
                print(f"🔍 Checking: {model_path}")
                model_data = self._load_model_metadata(model_path, model_dir)
                if model_data:
                    models.append(model_data)
        
        return models
    
    def _load_model_metadata(self, model_path: str, model_dir: str) -> Optional[Dict[str, Any]]:
        """Load metadata using your actual file names"""
        try:
            print(f"📁 Loading metadata from: {model_path}")
            
            files = os.listdir(model_path)
            print(f"   Files found: {files}")
            
            # Find files with your naming pattern
            model_files = {
                'rag_metadata': None,
                'feature_schema': None, 
                'performance': None,
                'model': None
            }
            
            for file in files:
                file_lower = file.lower()
                if 'rag_metadata' in file_lower:
                    model_files['rag_metadata'] = file
                elif 'feature_schema' in file_lower:
                    model_files['feature_schema'] = file
                elif 'performance' in file_lower:
                    model_files['performance'] = file
                elif file.endswith('.pkl'):
                    model_files['model'] = file
            
            print(f"   Identified files: {model_files}")
            
            # Load RAG metadata
            if not model_files['rag_metadata']:
                print(f"❌ No RAG metadata file found in {model_path}")
                return None
                
            metadata_path = os.path.join(model_path, model_files['rag_metadata'])
            with open(metadata_path, 'r') as f:
                rag_metadata = json.load(f)
            print("   ✅ Loaded RAG metadata")
            
            # Load feature schema
            feature_schema = {}
            if model_files['feature_schema']:
                schema_path = os.path.join(model_path, model_files['feature_schema'])
                with open(schema_path, 'r') as f:
                    feature_schema = json.load(f)
                print("   ✅ Loaded feature schema")
            else:
                print(f"⚠️  No feature schema file found")
            
            # Load performance metrics
            performance_metrics = {}
            if model_files['performance']:
                metrics_path = os.path.join(model_path, model_files['performance'])
                with open(metrics_path, 'r') as f:
                    performance_metrics = json.load(f)
                print("   ✅ Loaded performance metrics")
            else:
                print(f"⚠️  No performance metrics file found")
            
            # Create model registry data
            model_identity = rag_metadata.get("model_identity", {})
            
            model_data = {
                "model_registry": {
                    "model_id": f"{model_dir}_v1",
                    "version": "1.0.0",
                    "status": "production",
                    "registered_at": "2024-01-15T10:18:46Z",
                    "registered_by": "data_science_team"
                },
                "model_identity": model_identity,
                "file_references": model_files,
                "performance": {
                    "r2_score": rag_metadata.get("performance_profile", {}).get("accuracy_metrics", {}).get("r2_score", 0),
                    "rmse": rag_metadata.get("performance_profile", {}).get("accuracy_metrics", {}).get("rmse", 0),
                    "mape": rag_metadata.get("performance_profile", {}).get("accuracy_metrics", {}).get("mape", 0),
                    "training_samples": performance_metrics.get("training_info", {}).get("training_samples", 0),
                    "validation_samples": performance_metrics.get("training_info", {}).get("validation_samples", 0)
                },
                "model_path": model_path,
                "feature_schema": feature_schema,
                "performance_metrics": performance_metrics,
                "rag_metadata": rag_metadata
            }
            
            model_name = model_identity.get('name', model_dir)
            print(f"✅ Discovered: {model_name}")
            return model_data
            
        except Exception as e:
            print(f"❌ Error loading {model_dir}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def register_all_models(self):
        """Register all discovered models"""
        models = self.discover_models()
        
        if not models:
            print("❌ No models found to register!")
            return
        
        print(f"\n📦 Registering {len(models)} models to knowledge base...")
        
        for model_data in models:
            try:
                self._register_single_model(model_data)
            except Exception as e:
                model_name = model_data['model_identity'].get('name', 'Unknown')
                print(f"❌ Failed to register {model_name}: {e}")
        
        print("🎉 Model registration completed!")
    
    def _register_single_model(self, model_data: Dict[str, Any]):
        """Register a single model"""
        model_id = model_data['model_registry']['model_id']
        model_name = model_data['model_identity']['name']
        model_type = "regression"
        
        print(f"\n🔧 Registering: {model_name} ({model_id})")
        
        # Create dataset schema
        dataset_name = f"{model_id}_dataset"
        forecasting_frequency = model_data['model_identity'].get('forecasting_frequency', 'unknown')
        
        self.db.conn.execute("""
            INSERT OR IGNORE INTO Dataset_Schemas 
            (dataset_name, description, min_rows) 
            VALUES (?, ?, ?)
        """, (
            dataset_name,
            f"Dataset for {model_name} - {forecasting_frequency} forecasting",
            self._get_min_rows(forecasting_frequency)
        ))
        
        cursor = self.db.conn.execute(
            "SELECT dataset_id FROM Dataset_Schemas WHERE dataset_name = ?", 
            (dataset_name,)
        )
        dataset_id = cursor.fetchone()['dataset_id']
        
        # Get model file path
        model_file = model_data['file_references']['model']
        model_path = os.path.join(model_data['model_path'], model_file) if model_file else "unknown.pkl"
        
        # Register the ML model
        model_record = {
            "model_name": model_name,
            "model_type": model_type,
            "model_path": model_path,
            "required_features": json.dumps(self._get_required_features(model_data)),
            "optional_features": json.dumps(self._get_optional_features(model_data)),
            "target_variable": self._get_target_variable(model_data),
            "performance_metrics": json.dumps(model_data['performance_metrics']),
            "training_config": json.dumps({
                "algorithm": "LightGBM",
                "forecasting_frequency": forecasting_frequency,
                "horizon": model_data['model_identity'].get('horizon', 'unknown'),
                "training_samples": model_data['performance_metrics'].get('training_info', {}).get('training_samples', 0),
                "best_iteration": model_data['performance_metrics'].get('training_info', {}).get('best_iteration', 0)
            }),
            "dataset_id": dataset_id,
            "hyperparameters": json.dumps({
                "model_family": "LightGBM",
                "version": model_data['model_registry']['version'],
                "n_features": model_data['performance_metrics'].get('training_info', {}).get('n_features', 0),
                "data_leak_prevention": model_data['performance_metrics'].get('data_leak_prevention', {})
            })
        }
        
        self.db.conn.execute("""
            INSERT OR REPLACE INTO ML_Models 
            (model_name, model_type, model_path, required_features, optional_features, 
             target_variable, performance_metrics, training_config, dataset_id, hyperparameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(model_record.values()))
        
        # Register business rules
        self._register_business_rules(model_data, dataset_id)
        
        self.db.conn.commit()
        print(f"✅ Registered: {model_name}")
    
    def _get_required_features(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract required features from model data"""
        feature_schema = model_data.get('feature_schema', {})
        rag_metadata = model_data.get('rag_metadata', {})
        
        # Get from feature schema
        features = feature_schema.get('feature_names', [])
        
        # Get from RAG metadata if available
        tech_specs = rag_metadata.get('technical_specifications', {})
        input_req = tech_specs.get('input_requirements', {})
        required_entities = input_req.get('required_entities', [])
        
        # Combine and return top features
        all_features = list(set(features + required_entities))
        return all_features[:10]  # Return top 10 as required
    
    def _get_optional_features(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract optional features from model data"""
        feature_schema = model_data.get('feature_schema', {})
        features = feature_schema.get('feature_names', [])
        required = self._get_required_features(model_data)
        
        # Return features not in required list
        return [f for f in features if f not in required]
    
    def _get_min_rows(self, frequency: str) -> int:
        return {
            'daily': 30,
            'monthly': 12,
            'weekly': 13
        }.get(frequency, 12)
    
    def _get_target_variable(self, model_data: Dict[str, Any]) -> str:
        forecasting_frequency = model_data['model_identity'].get('forecasting_frequency', '')
        if 'monthly' in forecasting_frequency:
            return 'monthly_sales'
        elif 'daily' in forecasting_frequency:
            return 'daily_sales'
        else:
            return 'sales'
    
    def _register_business_rules(self, model_data: Dict[str, Any], dataset_id: int):
        """Register business rules"""
        model_name = model_data['model_identity']['name']
        frequency = model_data['model_identity'].get('forecasting_frequency', 'unknown')
        
        rules = [
            (
                f"select_{model_name.lower().replace(' ', '_')}",
                f"Use {model_name} for {frequency} forecasting",
                f"forecast_frequency == '{frequency}'",
                f"SELECT model_id FROM ML_Models WHERE model_name = '{model_name}'",
                "model_selection",
                1,
                "1.0"
            )
        ]
        
        for rule in rules:
            self.db.conn.execute("""
                INSERT OR IGNORE INTO Rules 
                (name, description, condition, action, rule_type, priority, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rule)
    
    def show_database_status(self):
        """Show current database status"""
        try:
            tables = ['Dataset_Schemas', 'ML_Models', 'Rules', 'Column_Definitions']
            print(f"\n📊 DATABASE STATUS:")
            print("=" * 40)
            
            for table in tables:
                cursor = self.db.conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                count = cursor.fetchone()['count']
                print(f"  {table}: {count} records")
            
            # Show registered models with details
            cursor = self.db.conn.execute("""
                SELECT model_name, model_type, target_variable, 
                       json_extract(performance_metrics, '$.validation_metrics.r2') as r2_score,
                       json_extract(performance_metrics, '$.validation_metrics.mape') as mape
                FROM ML_Models 
                WHERE is_active = TRUE
            """)
            models = cursor.fetchall()
            
            print(f"\n🤖 REGISTERED MODELS:")
            for model in models:
                r2 = model['r2_score'] if model['r2_score'] else 'N/A'
                mape = model['mape'] if model['mape'] else 'N/A'
                print(f"  • {model['model_name']} ({model['model_type']})")
                print(f"    Target: {model['target_variable']}, R²: {r2}, MAPE: {mape}%")
                
        except Exception as e:
            print(f"❌ Error checking database: {e}")
    
    def close(self):
        self.db.close()

if __name__ == "__main__":
    print("🚀 MODEL REGISTRY SERVICE (FIXED FILE NAMES)")
    print("=" * 50)
    
    registry = ModelRegistryService("supply_chain.db")
    
    try:
        registry.register_all_models()
        registry.show_database_status()
        print("\n🎉 Registry service completed!")
    except Exception as e:
        print(f"❌ Registry service failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        registry.close()