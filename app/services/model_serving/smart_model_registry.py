# app/services/model_serving/smart_model_registry.py
"""
Smart Model Discovery & Registration System
Automatically discovers models and their metadata files regardless of naming conventions
"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)


class SmartModelDiscovery:
    """
    Intelligently discovers models and their metadata files
    Works with ANY naming convention
    """
    
    def __init__(self, models_base_dir: str = "models/forecasting"):
        self.models_base_dir = Path(models_base_dir)
        
    def discover_all_models(self) -> List[Dict[str, Any]]:
        """
        Discover all models in the directory structure
        Returns list of model data dictionaries
        """
        discovered_models = []
        
        if not self.models_base_dir.exists():
            logger.error(f"❌ Models directory not found: {self.models_base_dir}")
            return discovered_models
        
        logger.info(f"🔍 Scanning: {self.models_base_dir}")
        
        # Find all directories that contain model files
        for root, dirs, files in os.walk(self.models_base_dir):
            root_path = Path(root)
            
            # Check if this directory contains model files
            model_files = [f for f in files if self._is_model_file(f)]
            json_files = [f for f in files if f.endswith('.json')]
            
            if model_files and json_files:
                logger.info(f"📁 Found model directory: {root_path.name}")
                
                model_data = self._extract_model_data(root_path, model_files, json_files)
                if model_data:
                    discovered_models.append(model_data)
                    logger.info(f"✅ Extracted: {model_data['model_name']}")
        
        logger.info(f"🎉 Discovered {len(discovered_models)} models")
        return discovered_models
    
    def _is_model_file(self, filename: str) -> bool:
        """Check if file is a trained model"""
        model_extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth', '.json']
        # Exclude metadata JSON files
        if filename.endswith('.json') and any(keyword in filename.lower() 
                                              for keyword in ['metadata', 'schema', 'performance', 'config']):
            return False
        return any(filename.endswith(ext) for ext in model_extensions)
    
    def _extract_model_data(
        self, 
        model_dir: Path, 
        model_files: List[str], 
        json_files: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract all model data from directory
        Intelligently identifies metadata, schema, and performance files
        """
        try:
            # Identify JSON file types by content and naming
            metadata_file = self._find_file_by_type(json_files, ['metadata', 'rag'])
            schema_file = self._find_file_by_type(json_files, ['schema', 'feature'])
            performance_file = self._find_file_by_type(json_files, ['performance', 'metrics', 'eval'])
            
            logger.info(f"   📄 Files identified:")
            logger.info(f"      - Metadata: {metadata_file}")
            logger.info(f"      - Schema: {schema_file}")
            logger.info(f"      - Performance: {performance_file}")
            logger.info(f"      - Model: {model_files[0]}")
            
            # Load metadata (required)
            if not metadata_file:
                logger.warning(f"⚠️ No metadata file found in {model_dir.name}")
                return None
            
            with open(model_dir / metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load schema (optional)
            schema = {}
            if schema_file:
                try:
                    with open(model_dir / schema_file, 'r') as f:
                        schema = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Could not load schema: {e}")
            
            # Load performance (optional)
            performance = {}
            if performance_file:
                try:
                    with open(model_dir / performance_file, 'r') as f:
                        performance = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Could not load performance: {e}")
            
            # Extract model information
            model_data = self._structure_model_data(
                metadata=metadata,
                schema=schema,
                performance=performance,
                model_file=model_files[0],
                model_path=str(model_dir / model_files[0]),
                model_dir_name=model_dir.name
            )
            
            return model_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting model data from {model_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_file_by_type(self, files: List[str], keywords: List[str]) -> Optional[str]:
        """
        Find file by checking keywords in filename
        Returns first match
        """
        for file in files:
            file_lower = file.lower()
            if any(keyword in file_lower for keyword in keywords):
                return file
        return None
    
    def _structure_model_data(
        self,
        metadata: Dict,
        schema: Dict,
        performance: Dict,
        model_file: str,
        model_path: str,
        model_dir_name: str
    ) -> Dict[str, Any]:
        """
        Structure model data in standardized format for database registration
        Handles different metadata structures intelligently
        """
        # Extract model identity (handles different structures)
        model_identity = metadata.get('model_identity', {})
        if not model_identity:
            # Fallback: create from directory name
            model_identity = {
                'name': model_dir_name.replace('_', ' ').title(),
                'id': model_dir_name,
                'type': self._detect_model_type(model_file),
                'framework': 'unknown'
            }
        
        model_name = model_identity.get('name', model_dir_name.replace('_', ' ').title())
        model_id = model_identity.get('id', model_dir_name)
        model_type = model_identity.get('type', self._detect_model_type(model_file))
        framework = model_identity.get('framework', 'unknown')
        
        # Extract forecasting frequency
        forecasting_frequency = model_identity.get('forecasting_frequency', 'unknown')
        if forecasting_frequency == 'unknown':
            # Try to detect from name or metadata
            if 'daily' in model_name.lower() or 'daily' in model_dir_name.lower():
                forecasting_frequency = 'daily'
            elif 'monthly' in model_name.lower() or 'monthly' in model_dir_name.lower():
                forecasting_frequency = 'monthly'
            elif 'weekly' in model_name.lower() or 'weekly' in model_dir_name.lower():
                forecasting_frequency = 'weekly'
        
        # Extract performance metrics
        perf_metrics = metadata.get('performance_profile', {}).get('accuracy_metrics', {})
        if not perf_metrics and performance:
            perf_metrics = performance.get('validation_metrics', {})
        
        # Extract features
        required_features = self._extract_required_features(metadata, schema)
        optional_features = self._extract_optional_features(metadata, schema)
        target_variable = self._extract_target_variable(metadata, schema, forecasting_frequency)
        
        # Extract use cases
        use_case = metadata.get('use_case_profile', {})
        best_for = use_case.get('best_for', [])
        not_recommended = use_case.get('not_recommended_for', [])
        
        # Structure final data
        structured_data = {
            'model_name': model_name,
            'model_id': model_id,
            'model_type': model_type,
            'model_path': model_path,
            'model_file': model_file,
            'framework': framework,
            'forecasting_frequency': forecasting_frequency,
            'required_features': required_features,
            'optional_features': optional_features,
            'target_variable': target_variable,
            'performance_metrics': perf_metrics,
            'best_for': best_for,
            'not_recommended_for': not_recommended,
            'full_metadata': metadata,
            'full_schema': schema,
            'full_performance': performance,
            'min_required_rows': self._get_min_rows(forecasting_frequency)
        }
        
        return structured_data
    
    def _detect_model_type(self, model_file: str) -> str:
        """Detect model type from filename"""
        file_lower = model_file.lower()
        
        if 'lightgbm' in file_lower or 'lgbm' in file_lower:
            return 'lightgbm'
        elif 'prophet' in file_lower:
            return 'prophet'
        elif 'arima' in file_lower:
            return 'arima'
        elif 'lstm' in file_lower:
            return 'lstm'
        elif 'tft' in file_lower:
            return 'tft'
        elif 'xgboost' in file_lower or 'xgb' in file_lower:
            return 'xgboost'
        else:
            ext = Path(model_file).suffix
            return f'unknown{ext}'
    
    def _extract_required_features(self, metadata: Dict, schema: Dict) -> List[str]:
        """Extract required features from metadata and schema"""
        features = []
        
        # Try metadata first
        tech_specs = metadata.get('technical_specifications', {})
        input_req = tech_specs.get('input_requirements', {})
        required_cols = input_req.get('required_columns', [])
        
        if required_cols:
            features.extend(required_cols)
        
        # Try schema as fallback
        if schema and not features:
            feature_names = schema.get('feature_names', [])
            features.extend(feature_names)
        
        # Remove duplicates and empty strings
        features = [f for f in features if f and f.strip()]
        
        return list(set(features))
    
    def _extract_optional_features(self, metadata: Dict, schema: Dict) -> List[str]:
        """Extract optional features"""
        features = []
        
        # Try metadata
        tech_specs = metadata.get('technical_specifications', {})
        input_req = tech_specs.get('input_requirements', {})
        optional_regressors = input_req.get('optional_regressors', [])
        
        features.extend(optional_regressors)
        
        # Try schema
        if schema:
            all_features = schema.get('feature_names', [])
            required = self._extract_required_features(metadata, schema)
            optional = [f for f in all_features if f not in required]
            features.extend(optional)
        
        # Remove duplicates and empty strings
        features = [f for f in features if f and f.strip()]
        
        return list(set(features))
    
    def _extract_target_variable(self, metadata: Dict, schema: Dict, frequency: str) -> str:
        """Extract target variable name - FIXED to check correct fields"""
        
        # 1. Try use_case_profile first (most accurate)
        use_case = metadata.get('use_case_profile', {})
        target = use_case.get('target_variable', '')  # ✅ CORRECT FIELD
        
        if target:
            logger.debug(f"Target from use_case_profile: {target}")
            return target
        
        # 2. Try technical_specifications (for complex models like LSTM)
        tech_specs = metadata.get('technical_specifications', {})
        input_req = tech_specs.get('input_requirements', {})
        
        # Check for required_columns that look like targets
        required_cols = input_req.get('required_columns', [])
        target_candidates = [col for col in required_cols 
                            if col.lower() in ['sales', 'revenue', 'totalrevenue', 
                                            'demand', 'numberofpieces', 'pieces', 
                                            'orders', 'quantity', 'y']]
        
        if target_candidates:
            logger.debug(f"Target from required_columns: {target_candidates[0]}")
            return target_candidates[0]
        
        # 3. Try schema
        if schema:
            target = schema.get('target_variable', '')
            if target:
                logger.debug(f"Target from schema: {target}")
                return target
        
        # 4. Fallback based on frequency (legacy logic)
        fallback_map = {
            'daily': 'daily_sales',
            'monthly': 'monthly_sales',
            'weekly': 'weekly_orders'
        }
        fallback = fallback_map.get(frequency, 'target')
        logger.warning(f"⚠️ Using fallback target: {fallback}")
        return fallback
    
    def _get_min_rows(self, frequency: str) -> int:
        """Get minimum required rows for frequency"""
        min_rows_map = {
            'daily': 30,
            'weekly': 13,
            'monthly': 12,
            'quarterly': 8,
            'yearly': 3
        }
        return min_rows_map.get(frequency, 12)


class SmartModelRegistrar:
    """
    Registers discovered models to the database
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def register_all_models(self, discovered_models: List[Dict[str, Any]]):
        """Register all discovered models to database"""
        if not discovered_models:
            logger.warning("⚠️ No models to register")
            return
        
        logger.info(f"\n📦 Registering {len(discovered_models)} models to database...")
        
        successful = 0
        failed = 0
        
        for model_data in discovered_models:
            try:
                self._register_single_model(model_data)
                successful += 1
                logger.info(f"✅ Registered: {model_data['model_name']}")
            except Exception as e:
                failed += 1
                logger.error(f"❌ Failed to register {model_data['model_name']}: {e}")
        
        logger.info(f"\n🎉 Registration complete!")
        logger.info(f"   ✅ Successful: {successful}")
        logger.info(f"   ❌ Failed: {failed}")
    
    def _register_single_model(self, model_data: Dict[str, Any]):
        """Register a single model with all its metadata"""
        
        # 1. Create/Update Dataset Schema
        dataset_id = self._register_dataset_schema(model_data)
        
        # 2. Register ML Model
        self._register_ml_model(model_data, dataset_id)
        
        # 3. Register Business Rules
        self._register_business_rules(model_data)
        
        self.db.conn.commit()
    
    def _register_dataset_schema(self, model_data: Dict[str, Any]) -> int:
        """Register dataset schema - FIXED VERSION"""
        dataset_name = f"{model_data['model_id']}_dataset"
        
        # ✅ FIX: Use correct column names that match your actual database schema
        self.db.conn.execute("""
            INSERT OR REPLACE INTO Dataset_Schemas 
            (dataset_name, description, min_rows, source_path)
            VALUES (?, ?, ?, ?)
        """, (
            dataset_name,
            f"Dataset for {model_data['model_name']} - {model_data['forecasting_frequency']} forecasting",
            model_data['min_required_rows'],
            f"auto_generated_for_{model_data['model_name']}"
        ))
        
        cursor = self.db.conn.execute(
            "SELECT dataset_id FROM Dataset_Schemas WHERE dataset_name = ?",
            (dataset_name,)
        )
        dataset_id = cursor.fetchone()['dataset_id']
        
        # ✅ FIX: Now register column definitions separately
        self._register_column_definitions(dataset_id, model_data)
        
        return dataset_id

    def _register_column_definitions(self, dataset_id: int, model_data: Dict[str, Any]):
        """Register column definitions in the correct table"""
        
        # Clear existing definitions for this dataset
        self.db.conn.execute(
            "DELETE FROM Column_Definitions WHERE dataset_id = ?",
            (dataset_id,)
        )
        
        # Register required features
        for feature in model_data['required_features']:
            self.db.conn.execute("""
                INSERT INTO Column_Definitions 
                (dataset_id, column_name, requirement_level, data_type, role, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                feature,
                'required',
                self._infer_data_type(feature),
                self._infer_feature_role(feature, model_data['target_variable']),
                f"Required feature for {model_data['model_name']}"
            ))
        
        # Register optional features
        for feature in model_data['optional_features']:
            self.db.conn.execute("""
                INSERT INTO Column_Definitions 
                (dataset_id, column_name, requirement_level, data_type, role, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                feature,
                'optional',
                self._infer_data_type(feature),
                self._infer_feature_role(feature, model_data['target_variable']),
                f"Optional feature for {model_data['model_name']}"
            ))
        
        # Register target variable
        self.db.conn.execute("""
            INSERT INTO Column_Definitions 
            (dataset_id, column_name, requirement_level, data_type, role, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            model_data['target_variable'],
            'required',
            'numeric',
            'target',
            f"Target variable for {model_data['model_name']}"
        ))

    def _infer_data_type(self, feature_name: str) -> str:
        """Infer data type from feature name"""
        feature_lower = feature_name.lower()
        
        if any(keyword in feature_lower for keyword in ['date', 'time', 'ds']):
            return 'datetime'
        elif any(keyword in feature_lower for keyword in ['flag', 'is_', 'has_', 'bool']):
            return 'boolean'
        elif any(keyword in feature_lower for keyword in ['id', 'code', 'category']):
            return 'categorical'
        else:
            return 'numeric'

    def _infer_feature_role(self, feature_name: str, target_variable: str) -> str:
        """Infer feature role"""
        if feature_name == target_variable:
            return 'target'
        elif any(keyword in feature_name.lower() for keyword in ['date', 'time', 'ds']):
            return 'timestamp'
        elif any(keyword in feature_name.lower() for keyword in ['id', 'key']):
            return 'identifier'
        else:
            return 'feature'

    def _register_ml_model(self, model_data: Dict[str, Any], dataset_id: int):
        """Register ML model - COMPLETE FIX"""
        
        # ✅ Include ALL columns from ML_Models schema
        self.db.conn.execute("""
            INSERT OR REPLACE INTO ML_Models 
            (model_name, model_type, model_path, required_features, optional_features,
            target_variable, performance_metrics, training_config, dataset_id, 
            hyperparameters, best_for, not_recommended_for, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_data['model_name'],
            model_data['model_type'],
            model_data['model_path'],
            json.dumps(model_data['required_features']),
            json.dumps(model_data['optional_features']),
            model_data['target_variable'],  # ✅ Will be correct after fixing extraction
            json.dumps(model_data['performance_metrics']),
            json.dumps({
                'framework': model_data['framework'],
                'frequency': model_data['forecasting_frequency'],
                'description': model_data.get('description', '')
            }),
            dataset_id,
            json.dumps({}),  # hyperparameters
            json.dumps(model_data['best_for']),  # ✅ NOW INCLUDED
            json.dumps(model_data['not_recommended_for']),  # ✅ NOW INCLUDED
            1  # is_active
        ))

    def _register_business_rules(self, model_data: Dict[str, Any]):
        """Register business rules for model selection"""
        
        # Rule for frequency-based selection
        rule_name = f"select_{model_data['model_id']}"
        
        self.db.conn.execute("""
            INSERT OR IGNORE INTO Rules 
            (name, description, condition, action, rule_type, priority, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            rule_name,
            f"Select {model_data['model_name']} for {model_data['forecasting_frequency']} forecasting",
            f"frequency == '{model_data['forecasting_frequency']}'",
            f"SELECT model_id FROM ML_Models WHERE model_name = '{model_data['model_name']}'",
            'model_selection',
            1,
            '1.0'
        ))


# Main execution script
def discover_and_register_models(
    models_dir: str = "models/forecasting",
    db_path: str = "supply_chain.db"
):
    """
    Main function to discover and register all models
    """
    import sqlite3
    
    print("="*60)
    print("🚀 SMART MODEL DISCOVERY & REGISTRATION")
    print("="*60)
    
    # Step 1: Discover models
    print(f"\n📁 Scanning directory: {models_dir}")
    discovery = SmartModelDiscovery(models_dir)
    discovered_models = discovery.discover_all_models()
    
    if not discovered_models:
        print("\n❌ No models discovered!")
        return
    
    # Step 2: Display discovered models
    print(f"\n📊 DISCOVERED MODELS:")
    print("-" * 60)
    for i, model in enumerate(discovered_models, 1):
        print(f"\n{i}. {model['model_name']}")
        print(f"   Type: {model['model_type']}")
        print(f"   Frequency: {model['forecasting_frequency']}")
        print(f"   Path: {model['model_path']}")
        print(f"   Required Features: {len(model['required_features'])}")
        print(f"   Target: {model['target_variable']}")
        print(f"   Best For: {model['best_for']}")
        print(f"   Performance: R²={model['performance_metrics'].get('r2_score', 'N/A')}")
    
    # Step 3: Register to database
    print(f"\n💾 Connecting to database: {db_path}")
    
    class DBConnection:
        def __init__(self, db_path):
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
    
    db = DBConnection(db_path)
    
    registrar = SmartModelRegistrar(db)
    registrar.register_all_models(discovered_models)
    
    # Step 4: Verify registration
    print(f"\n✅ VERIFICATION:")
    cursor = db.conn.execute("SELECT COUNT(*) as count FROM ML_Models WHERE is_active = 1")
    count = cursor.fetchone()['count']
    print(f"   Active models in database: {count}")
    
    cursor = db.conn.execute("""
        SELECT model_name, model_type, target_variable, best_for 
        FROM ML_Models 
        WHERE is_active = 1
        ORDER BY model_name
    """)
    
    print(f"\n📋 REGISTERED MODELS:")
    for row in cursor.fetchall():
        best_for = json.loads(row['best_for']) if row['best_for'] else []
        print(f"   • {row['model_name']} ({row['model_type']})")
        print(f"     Target: {row['target_variable']}")
        print(f"     Best For: {best_for}")
    
    db.conn.close()
    
    print(f"\n{'='*60}")
    print("🎉 MODEL REGISTRATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    # Run the discovery and registration
    discover_and_register_models(
        models_dir="models/forecasting",
        db_path="supply_chain.db"
    )