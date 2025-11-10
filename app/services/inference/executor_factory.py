
# app/services/inference/executor_factory.py
"""
Factory for creating model executors with configuration
"""

from typing import Dict, Any
import yaml
import json
from pathlib import Path

from app.services.inference.base_model_executor import (
    ModelMetadata, ModelType
)
from app.services.inference.model_inference_service import ModelInferenceService


class ModelExecutorFactory:
    """
    Factory for creating and configuring model executors
    Supports configuration from files (YAML/JSON)
    """
    
    @staticmethod
    def create_from_config(config_path: str) -> ModelInferenceService:
        """Create inference service from configuration file"""
        config_file = Path(config_path)
        
        if config_file.suffix == '.yaml':
            with open(config_file) as f:
                config = yaml.safe_load(f)
        elif config_file.suffix == '.json':
            with open(config_file) as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")
        
        # Create service
        service = ModelInferenceService(
            model_storage_path=config.get('model_storage_path', 'models/')
        )
        
        # Load models from config
        for model_config in config.get('models', []):
            metadata = ModelExecutorFactory._config_to_metadata(model_config)
            service.load_model_from_metadata(metadata)
        
        return service
    
    @staticmethod
    def _config_to_metadata(config: Dict[str, Any]) -> ModelMetadata:
        """Convert config dict to ModelMetadata"""
        return ModelMetadata(
            model_name=config['name'],
            model_type=ModelType(config['type']),
            version=config.get('version', '1.0'),
            target_column=config['target_column'],
            required_features=config.get('required_features', []),
            optional_features=config.get('optional_features', []),
            frequency=config.get('frequency', 'daily'),
            supports_multistep=config.get('supports_multistep', True),
            supports_exogenous=config.get('supports_exogenous', False),
            max_horizon=config.get('max_horizon')
        )


# Example configuration file (models_config.yaml):
"""
model_storage_path: "models/"

models:
  - name: "Daily_Shop_Sales_Forecaster"
    type: "lightgbm"
    version: "1.0"
    target_column: "sales"
    required_features:
      - "shop_id"
      - "date"
      - "sales_lag_1"
      - "sales_lag_7"
    optional_features:
      - "sales_lag_30"
    frequency: "daily"
    supports_multistep: true
    max_horizon: 90

  - name: "Supply_Chain_Prophet_Forecaster"
    type: "prophet"
    version: "2.0"
    target_column: "y"
    required_features:
      - "ds"
      - "y"
    optional_features:
      - "is_saturday"
      - "is_sunday"
    frequency: "daily"
    supports_multistep: true
    supports_exogenous: true
    max_horizon: 365
"""