# app/services/model_serving/model_registry_service.py
import json
import os
import glob
import sqlite3
from pathlib import Path #python
from typing import Dict, List, Any, Optional
import sys
import logging
import joblib

logger = logging.getLogger(__name__)

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
    Service for managing and loading trained forecasting models
    Add these methods to your existing class
    """
    

    
    def __init__(self, models_dir: str = "models/forecasting"):
        # ✅ This should point to where your models actually are
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        # ✅ Initialize database connection properly
        try:
            from app.knowledge_base.relational_kb.sqlite_manager import SQLiteManager
            self.db = SQLiteManager("supply_chain.db")
        except ImportError:
            logger.warning("SQLiteManager not available, using fallback")
            self.db = SQLiteManager("supply_chain.db")  # Your fallback class
        
        logger.info(f"📦 Model Registry initialized: {self.models_dir}")


    # In ModelRegistryService - ADD THIS METHOD:
    def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discover models in the models directory structure
        This method is called by register_all_models() but not implemented
        """
        discovered_models = []
        models_base = Path(self.models_dir) / "models"
        
        if not models_base.exists():
            logger.warning(f"Models directory not found: {models_base}")
            return discovered_models
        
        # Look for model directories
        for model_dir in models_base.iterdir():
            if model_dir.is_dir():
                # Check if this looks like a model directory
                model_data = self._load_model_metadata(str(model_dir), model_dir.name)
                if model_data:
                    discovered_models.append(model_data)
        
        logger.info(f"Discovered {len(discovered_models)} models")
        return discovered_models

    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a trained model from the registry
        Supports nested folder structure: models/algorithm/purpose/model_file
        
        Args:
            model_name: Name of the model to load (e.g., 'lightgbm_daily_forecasting')
            
        Returns:
            Loaded model object or None if loading fails
        """
        logger.info(f"📥 Loading model: {model_name}")
        
         # ✅ IMPROVE: Check database first for model metadata
        try:
            model_info = self._get_model_from_database(model_name)
            if model_info:
                logger.info(f"📋 Found model in database: {model_info['model_path']}")
                # Use the path from database instead of searching
                model_path = Path(model_info['model_path'])
                if model_path.exists():
                    model_paths = [model_path]
        except Exception as e:
            logger.debug(f"Database lookup failed: {e}")

        # Check if already loaded (cache)
        if model_name in self.loaded_models:
            logger.info(f"✅ Returning cached model: {model_name}")
            return self.loaded_models[model_name]
        
        try:
            # Find model file in nested structure
            model_paths = self._find_model_path(model_name)
            
            if not model_paths:
                logger.error(f"❌ No model file found for: {model_name}")
                return self._create_mock_model(model_name)
            
            # Try to load from first found path
            model_path = model_paths[0]
            logger.info(f"📂 Loading from: {model_path}")
            
            # Load based on file extension
            if model_path.suffix in ['.pkl', '.joblib']:
                model = joblib.load(model_path)
                
            elif model_path.suffix == '.h5':
                # For neural network models (TensorFlow/Keras)
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(str(model_path))
                except ImportError:
                    logger.warning("TensorFlow not available, using mock model")
                    return self._create_mock_model(model_name)
                    
            elif model_path.suffix in ['.pt', '.pth']:
                # For PyTorch models
                try:
                    import torch
                    model = torch.load(model_path)
                except ImportError:
                    logger.warning("PyTorch not available, using mock model")
                    return self._create_mock_model(model_name)
                    
            elif model_path.suffix == '.json':
                # For Prophet or other JSON-based models
                import json
                with open(model_path, 'r') as f:
                    model_config = json.load(f)
                # You might need custom logic here to reconstruct the model
                logger.info(f"Loaded JSON configuration for {model_name}")
                model = model_config
                
            else:
                logger.warning(f"Unknown model format: {model_path.suffix}")
                return self._create_mock_model(model_name)
            
            # Wrap model with metadata
            model_wrapper = ModelWrapper(model, model_name, str(model_path))
            
            # Cache the loaded model
            self.loaded_models[model_name] = model_wrapper
            
            logger.info(f"✅ Successfully loaded model: {model_name}")
            return model_wrapper
            
        except Exception as e:
            logger.error(f"❌ Error loading model {model_name}: {str(e)}")
            logger.exception("Full traceback:")
            # Return mock model as fallback
            return self._create_mock_model(model_name)


    def get_active_models_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all active models"""
        try:
            # Get active models from database
            models = self.db.execute_query(
                "SELECT * FROM ML_Models WHERE is_active = TRUE"
            )
            
            # Enhance with additional metadata if available
            enhanced_models = []
            for model in models:
                enhanced_model = dict(model)
                
                # Add any additional metadata processing here
                enhanced_model['metadata_loaded'] = True
                enhanced_model['registry_timestamp'] = '2024-01-01T00:00:00Z'  # You can make this dynamic
                
                enhanced_models.append(enhanced_model)
            
            logger.info(f"📋 Retrieved metadata for {len(enhanced_models)} active models")
            return enhanced_models
            
        except Exception as e:
            logger.error(f"❌ Error getting active models metadata: {e}")
            return []

    def _find_model_path(self, model_name: str) -> List[Path]:
        """
        Find model file(s) in nested folder structure:
        models/
        ├── lightgbm/
        │   ├── daily_forecasting/
        │   │   └── model.pkl
        │   └── monthly_forecasting/
        │       └── model.pkl
        
        Args:
            model_name: Model identifier (e.g., 'lightgbm_daily_forecasting')
            
        Returns:
            List of Path objects pointing to potential model files
        """
        model_paths = []
        models_base = Path(self.models_dir) 


        if not models_base.exists():
            logger.warning(f"Models base directory not found: {models_base}")
            return model_paths
        
        
        # Common model file extensions
        extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth', '.json']
        
        # Parse model_name (e.g., "lightgbm_daily_forecasting")
        parts = model_name.split('_')
        
        if len(parts) >= 2:
            # Try to extract algorithm and purpose
            algorithm = parts[0]  # e.g., 'lightgbm'
            purpose = '_'.join(parts[1:])  # e.g., 'daily_forecasting'
            
            # Check specific path: models/algorithm/purpose/
            specific_path = models_base / algorithm / purpose
            if specific_path.exists():
                logger.info(f"🔍 Searching in specific path: {specific_path}")
                for ext in extensions:
                    # Look for common naming patterns
                    patterns = [
                        specific_path / f"model{ext}",
                        specific_path / f"{algorithm}{ext}",
                        specific_path / f"{model_name}{ext}",
                        specific_path / f"trained_model{ext}",
                    ]
                    
                    for pattern_path in patterns:
                        if pattern_path.exists():
                            logger.info(f"✅ Found model file: {pattern_path}")
                            model_paths.append(pattern_path)
        
        # Fallback: Recursive search through all subdirectories
        if not model_paths:
            logger.info(f"🔍 Performing recursive search for: {model_name}")
            for ext in extensions:
                # Search for files matching the model_name
                found_files = list(models_base.rglob(f"*{model_name}*{ext}"))
                model_paths.extend(found_files)
                
                # Also search for generic model files
                if not found_files:
                    found_files = list(models_base.rglob(f"model{ext}"))
                    found_files.extend(list(models_base.rglob(f"trained_model{ext}")))
                    
                    # Filter by checking if path contains parts of model_name
                    for file_path in found_files:
                        if any(part in str(file_path) for part in parts):
                            model_paths.append(file_path)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in model_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        
        if unique_paths:
            logger.info(f"📦 Found {len(unique_paths)} potential model file(s)")
        else:
            logger.warning(f"⚠️ No model files found for: {model_name}")
        
        return unique_paths



    def _create_mock_model(self, model_name: str):
        """
        Create a mock model when actual model can't be loaded
        This allows the system to continue functioning for testing
        """
        logger.warning(f"⚠️ Creating mock model for: {model_name}")
        
        class MockModel:
            def __init__(self, name):
                self.name = name
                self.model_type = 'mock'
            
            def predict(self, X, horizon=30):
                """Generate mock predictions"""
                import numpy as np
                # Generate reasonable looking forecast data
                base = 100
                trend = np.linspace(0, 10, horizon)
                seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, horizon))
                noise = np.random.normal(0, 2, horizon)
                
                predictions = base + trend + seasonality + noise
                return predictions
            
            def __repr__(self):
                return f"MockModel({self.name})"
        
        mock = MockModel(model_name)
        model_wrapper = ModelWrapper(mock, model_name, "mock")
        
        # Cache it
        self.loaded_models[model_name] = model_wrapper
        
        return model_wrapper
    
    def list_available_models(self) -> list:
        """List all available models in the registry"""
        available_models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return available_models
        
        # Search for model files
        extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth']
        
        for ext in extensions:
            for model_file in self.models_dir.rglob(f"*{ext}"):
                model_name = model_file.stem
                available_models.append({
                    'name': model_name,
                    'path': str(model_file),
                    'type': ext[1:]  # Remove the dot
                })
        
        return available_models
    
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


    def _get_model_from_database(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information from knowledge base database"""
        try:
            result = self.db.execute_query(
                "SELECT * FROM ML_Models WHERE model_name = ? AND is_active = TRUE",
                (model_name,)
            )
            return result[0] if result else None
        except Exception as e:
            logger.debug(f"Database query failed: {e}")
            return None

    def unload_model(self, model_name: str):
        """Unload a model from cache"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"🗑️ Unloaded model: {model_name}")
    
    def clear_cache(self):
        """Clear all loaded models from cache"""
        self.loaded_models.clear()
        logger.info("🧹 Cleared model cache")

    #------------------------------------------------ 
    
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
    
class ModelWrapper:
    """
    Wrapper class for loaded models to provide consistent interface
    """
    
    def __init__(self, model, name: str, path: str):
        self.model = model
        self.name = name
        self.path = path
        self.model_type = self._detect_model_type(model)
        self.expected_features = self._extract_expected_features(model)
    
    def _detect_model_type(self, model) -> str:
        """Detect the type of model"""
        model_class = model.__class__.__name__
        
        if 'ARIMA' in model_class or 'SARIMAX' in model_class:
            return 'arima'
        elif 'Prophet' in model_class:
            return 'prophet'
        elif 'LightGBM' in model_class or 'LGBM' in model_class:
            return 'lightgbm'
        elif 'TFT' in model_class or 'TemporalFusion' in model_class:
            return 'tft'
        elif 'LSTM' in model_class:
            return 'lstm'
        elif 'Mock' in model_class:
            return 'mock'
        else:
            return 'unknown'
    
    def _extract_expected_features(self, model) -> List[str]:
        """Extract expected feature names from the model"""
        try:
            if hasattr(model, 'feature_name_'):
                # LightGBM models
                return model.feature_name_
            elif hasattr(model, 'feature_names_in_'):
                # Scikit-learn models
                return list(model.feature_names_in_)
            elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
                # LightGBM via scikit-learn wrapper
                return model.booster_.feature_name
            else:
                logger.warning(f"Could not extract feature names for {self.name}")
                return []
        except Exception as e:
            logger.warning(f"Error extracting feature names: {e}")
            return []
    
    def predict(self, data=None, horizon: int = 30):
        """
        Make predictions using the wrapped model with proper data formatting
        
        Args:
            data: Prepared data for prediction (can be None for time series)
            horizon: Number of periods to forecast
            
        Returns:
            Predictions array or dict
        """
        try:
            # Handle different model types
            if self.model_type == 'lightgbm':
                return self._predict_lightgbm(data, horizon)
            elif self.model_type == 'prophet':
                return self._predict_prophet(horizon)
            elif self.model_type == 'arima':
                return self._predict_arima(horizon)
            else:
                # Generic fallback
                return self._predict_generic(data, horizon)
                
        except Exception as e:
            logger.error(f"Error during prediction with {self.name}: {str(e)}")
            # Return mock predictions as fallback
            import numpy as np
            return np.random.normal(100, 10, horizon)
    
    def _predict_lightgbm(self, data, horizon: int):
        """Make predictions with LightGBM model"""
        import numpy as np
        import pandas as pd
        
        # If no data provided, create appropriate test data
        if data is None:
            logger.info("No data provided, creating test data for LightGBM")
            
            # Create proper 2D test data with expected features
            if self.expected_features:
                # Create DataFrame with correct feature names
                n_features = len(self.expected_features)
                test_data = np.random.randn(horizon, n_features)
                data = pd.DataFrame(test_data, columns=self.expected_features)
            else:
                # Fallback: create generic 2D array
                data = np.random.randn(horizon, 5).reshape(horizon, -1)
        
        # Ensure data is 2D
        if hasattr(data, 'shape') and len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Convert to DataFrame if we have feature names
        if self.expected_features and not isinstance(data, pd.DataFrame):
            if data.shape[1] == len(self.expected_features):
                data = pd.DataFrame(data, columns=self.expected_features)
            else:
                logger.warning(f"Feature count mismatch. Expected {len(self.expected_features)}, got {data.shape[1]}")
        
        # Make prediction
        predictions = self.model.predict(data)
        
        # Ensure we return a list
        if hasattr(predictions, 'tolist'):
            return predictions.tolist()
        elif isinstance(predictions, (pd.DataFrame, pd.Series)):
            return predictions.values.tolist()
        else:
            return list(predictions)
    
    def _predict_prophet(self, horizon: int):
        """Make predictions with Prophet model"""
        try:
            # Prophet needs future dataframe
            future = self.model.make_future_dataframe(periods=horizon)
            forecast = self.model.predict(future)
            return forecast['yhat'].tail(horizon).tolist()
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            # Fallback
            import numpy as np
            return np.random.normal(100, 10, horizon).tolist()
    
    def _predict_arima(self, horizon: int):
        """Make predictions with ARIMA model"""
        try:
            # ARIMA/SARIMAX forecast
            forecast = self.model.forecast(steps=horizon)
            if hasattr(forecast, 'tolist'):
                return forecast.tolist()
            else:
                return list(forecast)
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            # Fallback
            import numpy as np
            return np.random.normal(100, 10, horizon).tolist()
    
    def _predict_generic(self, data, horizon: int):
        """Generic prediction fallback"""
        import numpy as np
        
        if data is not None and hasattr(self.model, 'predict'):
            # Try generic predict
            try:
                predictions = self.model.predict(data)
                if hasattr(predictions, 'tolist'):
                    return predictions.tolist()
                return list(predictions)
            except Exception as e:
                logger.warning(f"Generic predict failed: {e}")
        
        # Final fallback
        return np.random.normal(100, 10, horizon).tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.name,
            'type': self.model_type,
            'path': self.path,
            'expected_features': self.expected_features,
            'feature_count': len(self.expected_features),
            'model_class': self.model.__class__.__name__
        }
    
    def __repr__(self):
        return f"ModelWrapper(name='{self.name}', type='{self.model_type}', path='{self.path}')"



# Example usage for testing
# In the test section at the bottom of the file
if __name__ == "__main__":
    # Test the model registry
    registry = ModelRegistryService()
    
    print("\n📦 Available Models:")
    models = registry.list_available_models()
    for model in models:
        print(f"  • {model['name']} ({model['type']})")
    
    # Try loading and testing each model
    print("\n🔍 Testing model loading and prediction...")
    test_models = ["monthly_shop_sales_predictor", "daily_shop_sales_predictor", "prophet_forecaster"]
    
    for model_name in test_models:
        print(f"\n🧪 Testing: {model_name}")
        model = registry.load_model(model_name)
        
        if model:
            print(f"   ✅ Loaded: {model}")
            
            # Get model info
            info = model.get_model_info()
            print(f"   📊 Model Info: {info['model_class']}")
            print(f"   🔧 Expected Features: {info['expected_features']}")
            
            # Test prediction
            print("   📈 Testing prediction...")
            try:
                predictions = model.predict(horizon=10)
                print(f"   ✅ Generated {len(predictions)} predictions")
                print(f"   📋 Sample predictions: {predictions[:5]}")
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        else:
            print(f"   ❌ Failed to load: {model_name}")


            