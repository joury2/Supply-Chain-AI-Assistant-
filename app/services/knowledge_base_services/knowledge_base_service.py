# storage/knowledge_base_service.py
# This module provides a high-level service class to interact
# app/services/knowledge_base_services/knowledge_base_service.py
# app/services/knowledge_base_services/knowledge_base_service.py
import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# 🔥 FIX: Add the correct import path for SQLiteManager
# Since your SQLiteManager is in app/knowledge_base/relational_kb/sqlite_manager.py
# We need to adjust the Python path

# Get the project root directory (3 levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    # Try importing from the correct path
    from app.knowledge_base.relational_kb.sqlite_manger import SQLiteManager
    print("✅ Successfully imported SQLiteManager from app.knowledge_base.relational_kb")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying direct import...")
    # Fallback: try direct import
    from app.knowledge_base.relational_kb.sqlite_manger import SQLiteManager

class SupplyChainService:
    """
    Main service class for supply chain forecasting operations.
    Works with the exact SQL schema you provided.
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        self.db = SQLiteManager(db_path)
    
    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Get model by ID"""
        cursor = self.db.conn.execute(
            "SELECT * FROM ML_Models WHERE model_id = ?", (model_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all active models"""
        cursor = self.db.conn.execute(
            "SELECT * FROM ML_Models WHERE is_active = TRUE"
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_dataset_schema(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get complete dataset schema with columns"""
        cursor = self.db.conn.execute(
            "SELECT * FROM Dataset_Schemas WHERE dataset_name = ?", (dataset_name,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        schema_data = dict(row)
        
        # Get related columns
        cursor = self.db.conn.execute(
            "SELECT * FROM Column_Definitions WHERE dataset_id = ?", 
            (schema_data['dataset_id'],)
        )
        
        schema_data['columns'] = [dict(col_row) for col_row in cursor.fetchall()]
        return schema_data
    
    def validate_dataset(self, dataset_name: str, provided_columns: List[str]) -> Dict[str, Any]:
        """Validate dataset against schema requirements"""
        schema = self.get_dataset_schema(dataset_name)
        if not schema:
            return {"valid": False, "errors": [f"Dataset '{dataset_name}' not found"]}
        
        required_columns = [
            col['column_name'] for col in schema['columns'] 
            if col['requirement_level'] == 'required'
        ]
        
        errors = []
        for req_col in required_columns:
            if req_col not in provided_columns:
                errors.append(f"Missing required column: {req_col}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "schema_name": schema['dataset_name'],
            "required_columns": required_columns,
            "provided_columns": provided_columns
        }
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Get all active rules"""
        cursor = self.db.conn.execute("""
            SELECT * FROM Rules 
            WHERE is_active = TRUE 
            ORDER BY priority DESC, rule_id
        """)
        return [dict(rule) for rule in cursor.fetchall()]
    
    def get_suitable_models(self, available_features: List[str], target_variable: str) -> List[Dict[str, Any]]:
        """Find models that match available features and target"""
        all_models = self.get_all_models()
        suitable_models = []
        
        for model in all_models:
            required_features = json.loads(model['required_features'])
            
            if (all(feature in available_features for feature in required_features) and
                model['target_variable'] == target_variable):
                suitable_models.append(model)
        
        return suitable_models
    
    def create_forecast_run(self, model_id: int, input_schema: str, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new forecast run"""
        self.db.conn.execute("""
            INSERT INTO Forecast_Runs 
            (model_id, input_dataset_schema, forecast_config, status)
            VALUES (?, ?, ?, ?)
        """, (model_id, input_schema, json.dumps(config), 'pending'))
        
        self.db.conn.commit()
        
        cursor = self.db.conn.execute("SELECT * FROM Forecast_Runs WHERE rowid = last_insert_rowid()")
        return dict(cursor.fetchone())
    
    def close(self):
        """Close database connection"""
        self.db.close()


def run_simple_demo():
    """Simple demo that works with your SQLiteManager"""
    print("🚀 SUPPLY CHAIN FORECASTING SYSTEM")
    print("=" * 50)
    
    try:
        # Use in-memory database for testing
        service = SupplyChainService(":memory:")
        
        print("✅ Service initialized successfully!")
        
        # 1. Show all models
        print("\n📊 ACTIVE MODELS:")
        print("-" * 30)
        models = service.get_all_models()
        print(f"Found {len(models)} models:")
        for model in models:
            metrics = json.loads(model['performance_metrics'])
            print(f"  • {model['model_name']} ({model['model_type']})")
            print(f"    Target: {model['target_variable']}")
            print(f"    MAPE: {metrics.get('MAPE', 'N/A')}")
        
        # 2. Show dataset schema
        print("\n🔍 DATASET SCHEMA:")
        print("-" * 30)
        schema = service.get_dataset_schema("demand_forecasting")
        if schema:
            print(f"Dataset: {schema['dataset_name']}")
            print(f"Description: {schema['description']}")
            print("Columns:")
            for col in schema['columns']:
                print(f"  • {col['column_name']} ({col['data_type']}) - {col['requirement_level']}")
        
        # 3. Test validation
        print("\n✅ DATASET VALIDATION TEST:")
        print("-" * 30)
        validation = service.validate_dataset(
            "demand_forecasting", 
            ["date", "demand", "price"]
        )
        print(f"Validation: {'✅ PASS' if validation['valid'] else '❌ FAIL'}")
        if validation['errors']:
            for error in validation['errors']:
                print(f"  • {error}")
        
        # 4. Test model selection
        print("\n🎯 MODEL SELECTION TEST:")
        print("-" * 30)
        suitable_models = service.get_suitable_models(
            ["date", "demand", "price"], "demand"
        )
        print(f"Found {len(suitable_models)} suitable models for features: ['date', 'demand', 'price']")
        for model in suitable_models:
            print(f"  • {model['model_name']}")
        
        # 5. Show business rules
        print("\n📜 BUSINESS RULES:")
        print("-" * 30)
        rules = service.get_active_rules()
        print(f"Found {len(rules)} active rules:")
        for rule in rules:
            print(f"  • {rule['name']} (Priority: {rule['priority']})")
        
        service.close()
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_simple_demo()