# app/knowledge_base/relational_kb/sqlite_manager.py
# This module defines the SQLite database schema for managing dataset schemas,
# column definitions, machine learning models, performance history, forecast runs,
# and feature importance.
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class SQLiteManager:
    """
    SQLite database manager for supply chain forecasting system.
    Uses sqlite3.Row factory for dictionary-like column access.
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and schema"""
        self.conn = sqlite3.connect(self.db_path)
        
        # 🔥 CRUCIAL: This makes cursor results accessible by column name (like a dictionary)
        self.conn.row_factory = sqlite3.Row
        
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        self._create_tables()
        self._insert_sample_data()  # This should now populate data
        self.conn.commit()
    
    def _create_tables(self):
        """Create all tables using the provided schema"""
        tables_sql = [
            # Dataset_Schemas table
            """
            CREATE TABLE IF NOT EXISTS Dataset_Schemas (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL UNIQUE,
                source_path TEXT,
                description TEXT,
                min_rows INTEGER DEFAULT 30,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Column_Definitions table  
            """
            CREATE TABLE IF NOT EXISTS Column_Definitions (
                column_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER REFERENCES Dataset_Schemas(dataset_id) ON DELETE CASCADE,
                column_name TEXT NOT NULL,
                requirement_level TEXT CHECK (requirement_level IN ('required', 'optional', 'conditional')),
                data_type TEXT NOT NULL,
                role TEXT,
                description TEXT,
                validation_rules TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(dataset_id, column_name)
            )
            """,
            
            # Rules table
            """
            CREATE TABLE IF NOT EXISTS Rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                condition TEXT NOT NULL,
                action TEXT NOT NULL,
                rule_type TEXT CHECK(rule_type IN ('model_selection', 'data_validation', 'business_logic')),
                priority INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT TRUE,
                version TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # ML_Models table
            """
            CREATE TABLE IF NOT EXISTS ML_Models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT CHECK(model_type IN ('time_series', 'regression', 'classification', 'ensemble')),
                model_path TEXT NOT NULL,
                required_features TEXT NOT NULL,
                optional_features TEXT,
                target_variable TEXT NOT NULL,
                performance_metrics TEXT,
                training_config TEXT,
                dataset_id INTEGER REFERENCES Dataset_Schemas(dataset_id),
                hyperparameters TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Model_Performance_History table
            """
            CREATE TABLE IF NOT EXISTS Model_Performance_History (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES ML_Models(model_id) ON DELETE CASCADE,
                training_date DATE NOT NULL,
                dataset_schema TEXT,
                metrics TEXT NOT NULL,
                sample_size INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Forecast_Runs table
            """
            CREATE TABLE IF NOT EXISTS Forecast_Runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER REFERENCES ML_Models(model_id),
                input_dataset_schema TEXT NOT NULL,
                forecast_config TEXT NOT NULL,
                results TEXT,
                validation_issues TEXT,
                llm_interpretation TEXT,
                status TEXT CHECK(status IN ('pending', 'running', 'completed', 'failed')) DEFAULT 'pending',
                executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            )
            """
        ]
        
        for sql in tables_sql:
            self.conn.execute(sql)
    
    def _insert_sample_data(self):
        """Insert sample data for demonstration - COMPLETE VERSION"""
        # Check if data already exists
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM ML_Models")
        count = cursor.fetchone()['count']
        if count > 0:
            print(f"✅ Database already has {count} models. Skipping sample data insertion.")
            return
        
        print("📝 Inserting sample data...")
        
        # 1. Insert Dataset Schema
        self.conn.execute("""
            INSERT OR IGNORE INTO Dataset_Schemas 
            (dataset_name, description, min_rows) 
            VALUES (?, ?, ?)
        """, ("demand_forecasting", "Dataset for demand forecasting with time series data", 30))
        
        # Get the dataset_id
        cursor = self.conn.execute("SELECT dataset_id FROM Dataset_Schemas WHERE dataset_name = ?", 
                                 ("demand_forecasting",))
        result = cursor.fetchone()
        if not result:
            print("❌ Failed to insert dataset schema")
            return
            
        dataset_id = result['dataset_id']
        print(f"✅ Inserted dataset with ID: {dataset_id}")
        
        # 2. Insert Column Definitions
        columns = [
            (dataset_id, 'date', 'required', 'datetime', 'timestamp', 'Date of observation', '{"not_null": true, "is_date": true}'),
            (dataset_id, 'demand', 'required', 'numeric', 'target', 'Demand quantity', '{"not_null": true, "min_value": 0}'),
            (dataset_id, 'price', 'optional', 'numeric', 'feature', 'Product price', '{"min_value": 0}'),
            (dataset_id, 'promotion_flag', 'optional', 'boolean', 'feature', 'Whether promotion was active', '{}')
        ]
        
        for col in columns:
            self.conn.execute("""
                INSERT INTO Column_Definitions 
                (dataset_id, column_name, requirement_level, data_type, role, description, validation_rules)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, col)
        print(f"✅ Inserted {len(columns)} column definitions")
        
        # 3. Insert Rules
        rules = [
            ("large_dataset_tft", "Use TFT for large datasets", 
             "dataset.rows > 10000", "SELECT model_id FROM ML_Models WHERE model_name = 'TFT'",
             "model_selection", 1, "1.0"),
            ("seasonal_data", "Use Prophet for seasonal data",
             "dataset.has_seasonality = True", "SELECT model_id FROM ML_Models WHERE model_name = 'Prophet'", 
             "model_selection", 2, "1.0")
        ]
        
        for rule in rules:
            self.conn.execute("""
                INSERT INTO Rules 
                (name, description, condition, action, rule_type, priority, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rule)
        print(f"✅ Inserted {len(rules)} business rules")
        
        # 4. Insert ML Models
        models = [
            ("TFT", "time_series", "/models/tft.pkl", 
             '["date", "demand"]', '["price", "promotion_flag"]', "demand",
             '{"MAE": 12.5, "RMSE": 18.3, "MAPE": 0.10}', 
             '{"epochs": 100, "learning_rate": 0.001}', dataset_id,
             '{"hidden_size": 64, "attention_heads": 4}'),
            ("Prophet", "time_series", "/models/prophet.pkl",
             '["date", "demand"]', '[]', "demand",
             '{"MAE": 15.2, "RMSE": 22.1, "MAPE": 0.12}',
             '{"seasonality_mode": "multiplicative"}', dataset_id,
             '{"changepoint_prior_scale": 0.05}'),
            ("XGBoost", "regression", "/models/xgboost.pkl",
             '["demand", "price"]', '["promotion_flag"]', "demand", 
             '{"MAE": 11.8, "RMSE": 16.7, "MAPE": 0.09}',
             '{"n_estimators": 100}', dataset_id,
             '{"max_depth": 6, "learning_rate": 0.1}')
        ]
        
        for model in models:
            self.conn.execute("""
                INSERT INTO ML_Models 
                (model_name, model_type, model_path, required_features, optional_features, 
                 target_variable, performance_metrics, training_config, dataset_id, hyperparameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, model)
        print(f"✅ Inserted {len(models)} ML models")
        
        # 5. Insert Performance History
        performance_data = [
            (1, '2024-01-15', 'demand_forecasting', '{"MAE": 12.5, "RMSE": 18.3, "MAPE": 0.10}', 1000),
            (2, '2024-01-10', 'demand_forecasting', '{"MAE": 15.2, "RMSE": 22.1, "MAPE": 0.12}', 800),
            (3, '2024-01-08', 'demand_forecasting', '{"MAE": 11.8, "RMSE": 16.7, "MAPE": 0.09}', 1500)
        ]
        
        for perf in performance_data:
            self.conn.execute("""
                INSERT INTO Model_Performance_History 
                (model_id, training_date, dataset_schema, metrics, sample_size)
                VALUES (?, ?, ?, ?, ?)
            """, perf)
        print(f"✅ Inserted {len(performance_data)} performance records")
        
        print("🎉 Sample data insertion completed successfully!")
    

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return results as dictionaries"""
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_table_counts(self) -> Dict[str, int]:
        """Get record counts for all tables"""
        tables = ['Dataset_Schemas', 'Column_Definitions', 'Rules', 'ML_Models', 'Model_Performance_History', 'Forecast_Runs']
        counts = {}
        for table in tables:
            cursor = self.conn.execute(f"SELECT COUNT(*) as count FROM {table}")
            counts[table] = cursor.fetchone()['count']
        return counts
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()