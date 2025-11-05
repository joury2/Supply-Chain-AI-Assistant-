-- app/knowledge_base/relational_kb/init_kb.sql
-- Fixed version that works with both SQL and Python SQLite

-- Dataset schemas (information about datasets used for modeling)
CREATE TABLE IF NOT EXISTS Dataset_Schemas (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL UNIQUE,
    source_path TEXT,
    description TEXT,
    min_rows INTEGER DEFAULT 30,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

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
);

-- Rules table
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
);

-- ML Models table
CREATE TABLE IF NOT EXISTS ML_Models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT,
    model_path TEXT NOT NULL,
    required_features TEXT NOT NULL,
    optional_features TEXT,
    target_variable TEXT NOT NULL,
    performance_metrics TEXT,
    training_config TEXT,
    dataset_id INTEGER REFERENCES Dataset_Schemas(dataset_id),
    hyperparameters TEXT,
    best_for TEXT, 
    not_recommended_for TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Model_Performance_History (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES ML_Models(model_id) ON DELETE CASCADE,
    training_date DATE NOT NULL,
    dataset_schema TEXT,
    metrics TEXT NOT NULL,
    sample_size INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

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
);

-- Insert sample data
INSERT OR IGNORE INTO Dataset_Schemas (dataset_name, description, min_rows) 
VALUES ('demand_forecasting', 'Dataset for demand forecasting with time series data', 30);

-- Insert column definitions
INSERT OR IGNORE INTO Column_Definitions (dataset_id, column_name, requirement_level, data_type, role, description)
SELECT 
    dataset_id, 'date', 'required', 'datetime', 'timestamp', 'Date of observation'
FROM Dataset_Schemas WHERE dataset_name = 'demand_forecasting';

INSERT OR IGNORE INTO Column_Definitions (dataset_id, column_name, requirement_level, data_type, role, description)
SELECT 
    dataset_id, 'demand', 'required', 'numeric', 'target', 'Demand quantity'
FROM Dataset_Schemas WHERE dataset_name = 'demand_forecasting';

INSERT OR IGNORE INTO Column_Definitions (dataset_id, column_name, requirement_level, data_type, role, description)
SELECT 
    dataset_id, 'price', 'optional', 'numeric', 'feature', 'Product price'
FROM Dataset_Schemas WHERE dataset_name = 'demand_forecasting';

-- Insert rules
INSERT OR IGNORE INTO Rules (name, description, condition, action, rule_type, priority, version)
VALUES 
('large_dataset_tft', 'Use TFT for large datasets', 'dataset.rows > 10000', 'SELECT model_id FROM ML_Models WHERE model_name = ''TFT''', 'model_selection', 1, '1.0'),
('seasonal_data', 'Use Prophet for seasonal data', 'dataset.has_seasonality = True', 'SELECT model_id FROM ML_Models WHERE model_name = ''Prophet''', 'model_selection', 2, '1.0'),
('small_dataset', 'Use ARIMA for small datasets', 'dataset.rows < 1000', 'SELECT model_id FROM ML_Models WHERE model_name = ''ARIMA''', 'model_selection', 3, '1.0');

-- Insert ML models
INSERT OR IGNORE INTO ML_Models (model_name, model_type, model_path, required_features, optional_features, target_variable, performance_metrics, training_config, hyperparameters)
SELECT 
    'TFT', 'time_series', '/models/tft.pkl', '["date", "demand"]', '["price"]', 'demand', 
    '{"MAE": 12.5, "RMSE": 18.3, "MAPE": 0.10}', '{"epochs": 100}', '{"hidden_size": 64}'
WHERE NOT EXISTS (SELECT 1 FROM ML_Models WHERE model_name = 'TFT');

INSERT OR IGNORE INTO ML_Models (model_name, model_type, model_path, required_features, optional_features, target_variable, performance_metrics, training_config, hyperparameters)
SELECT 
    'Prophet', 'time_series', '/models/prophet.pkl', '["date", "demand"]', '[]', 'demand',
    '{"MAE": 15.2, "RMSE": 22.1, "MAPE": 0.12}', '{"seasonality_mode": "multiplicative"}', '{"changepoint_prior_scale": 0.05}'
WHERE NOT EXISTS (SELECT 1 FROM ML_Models WHERE model_name = 'Prophet');

INSERT OR IGNORE INTO ML_Models (model_name, model_type, model_path, required_features, optional_features, target_variable, performance_metrics, training_config, hyperparameters)
SELECT 
    'ARIMA', 'time_series', '/models/arima.pkl', '["date", "demand"]', '["price"]', 'demand',
    '{"MAE": 14.8, "RMSE": 20.5, "MAPE": 0.11}', '{"order": "(1,1,1)"}', '{}'
WHERE NOT EXISTS (SELECT 1 FROM ML_Models WHERE model_name = 'ARIMA');