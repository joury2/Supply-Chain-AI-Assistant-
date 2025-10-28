# demo/sqlite_demo.py
def demonstrate_sqlite_crud():
    """Demonstrate SQLite CRUD operations - MUCH simpler!"""
    
    print("=== SQLite CRUD DEMONSTRATION ===\n")
    
    # Initialize (just one line!)
    service = SQLiteService(":memory:")  # Or use a file path for persistence
    
    # 1. CREATE - Add new schema
    print("1. CREATE Operations")
    print("-" * 40)
    
    new_schema = service.repo.create_dataset_schema({
        'schema_name': 'inventory_forecasting_v2',
        'description': 'Enhanced inventory forecasting schema',
        'min_rows': 100,
        'required_columns': ['date', 'inventory_level', 'sales', 'lead_time']
    })
    print(f"✅ Created schema: {new_schema['schema_name']}")
    
    # 2. READ - Get models and data
    print("\n2. READ Operations")
    print("-" * 40)
    
    # Get all active models with ONE query
    active_models = service.repo.get_active_models()
    print(f"✅ Active models: {len(active_models)}")
    for model in active_models:
        metrics = json.loads(model['performance_metrics'])
        print(f"   - {model['model_name']} (MAPE: {metrics.get('MAPE', 'N/A')})")
    
    # 3. MODEL SELECTION - Complex query made simple
    print("\n3. Model Selection")
    print("-" * 40)
    
    recommendations = service.get_model_recommendations(
        dataset_features=['date', 'demand', 'price'],
        target_variable='demand',
        business_constraints={'min_accuracy': 0.85}
    )
    
    print(f"✅ Recommended models: {len(recommendations['recommended_models'])}")
    for model in recommendations['recommended_models']:
        print(f"   - {model['model_name']}")
    
    # 4. FORECAST EXECUTION
    print("\n4. Forecast Execution Logging")
    print("-" * 40)
    
    if active_models:
        forecast_run = service.log_forecast_execution(
            model_id=active_models[0]['model_id'],
            input_schema='demand_forecasting',
            config={'periods': 30, 'confidence': 0.95},
            results={'forecast': [100, 105, 110, 115], 'metrics': {'MAE': 12.5}},
            validation_issues=['Minor data quality issues']
        )
        print(f"✅ Logged forecast run: {forecast_run['run_id']}")
        print(f"   Status: {forecast_run['status']}")
    
    # 5. ANALYTICS - Complex analytics with simple queries
    print("\n5. Model Analytics")
    print("-" * 40)
    
    if active_models:
        analytics = service.get_model_analytics(active_models[0]['model_id'])
        print(f"✅ Analytics for {active_models[0]['model_name']}:")
        print(f"   - Total runs: {analytics['forecast_stats']['total_runs']}")
        print(f"   - Success rate: {analytics['forecast_stats']['success_rate']:.1%}")
    
    # 6. VALIDATION
    print("\n6. Dataset Validation")
    print("-" * 40)
    
    validation = service.validate_dataset(
        dataset_columns=['date', 'inventory_level', 'sales', 'extra_col'],
        schema_name='inventory_forecasting_v2'
    )
    print(f"✅ Validation result: {'PASS' if validation['valid'] else 'FAIL'}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")
    
    print("\n=== DEMONSTRATION COMPLETE ===")

# Even simpler: Basic usage example
def quick_start_example():
    """Quick start example for the SQLite version"""
    
    # 1. Initialize database
    service = SQLiteService("supply_chain.db")  # Persistent file
    
    # 2. Validate a dataset
    validation = service.validate_dataset(
        ['date', 'demand', 'price'], 
        'demand_forecasting'
    )
    
    # 3. Get model recommendations
    recommendations = service.get_model_recommendations(
        ['date', 'demand', 'price'], 
        'demand'
    )
    
    # 4. Run forecast and log results
    if recommendations['recommended_models']:
        best_model = recommendations['recommended_models'][0]
        forecast_run = service.log_forecast_execution(
            best_model['model_id'],
            'demand_forecasting',
            {'periods': 30},
            results={'forecast': [100, 105, 110]}
        )
        
        print(f"Forecast completed: {forecast_run['run_id']}")

if __name__ == "__main__":
    demonstrate_sqlite_crud()