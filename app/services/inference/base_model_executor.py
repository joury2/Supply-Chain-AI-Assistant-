# app/services/inference/base_model_executor.py
"""
Base Model Executor - Plugin Architecture for Extensible Model Support
Solves: Tight coupling, hard-coded model logic, maintainability issues
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union,Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types"""
    LIGHTGBM = "lightgbm"
    PROPHET = "prophet"
    LSTM = "lstm"
    XGBOOST = "xgboost"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Metadata for model identification and configuration"""
    model_name: str
    model_type: ModelType
    version: str
    target_column: str
    required_features: List[str]
    optional_features: List[str]
    frequency: str
    supports_multistep: bool = True
    supports_exogenous: bool = False
    max_horizon: Optional[int] = None


@dataclass
class ForecastResult:
    predictions: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    model_name: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)  # ✅ Now field is defined
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "predictions": self.predictions,
            "confidence_intervals": self.confidence_intervals,
            "model_name": self.model_name,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


class BaseModelExecutor(ABC):
    """
    Abstract base class for all model executors
    Each model type implements its own executor
    """
    
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load the model from disk/registry"""
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data meets model requirements"""
        pass
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame, horizon: int) -> Any:
        """Model-specific preprocessing before inference"""
        pass
    
    @abstractmethod
    def predict(self, preprocessed_data: Any, horizon: int) -> ForecastResult:
        """Generate predictions"""
        pass
    
    @abstractmethod
    def postprocess(self, raw_predictions: Any) -> ForecastResult:
        """Transform raw model output to standardized format"""
        pass
    
    def execute(self, data: pd.DataFrame, horizon: int) -> ForecastResult:
        """
        Main execution pipeline - Template Method Pattern
        This is the ONLY method the supply chain service needs to call
        """
        logger.info(f"🚀 Executing {self.metadata.model_name} for {horizon} periods")
        
        # Validation
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.metadata.model_name} not loaded")
        
        if not self.validate_input(data):
            raise ValueError(f"Input validation failed for {self.metadata.model_name}")
        
        # Execute pipeline
        try:
            # 1. Preprocess
            preprocessed = self.preprocess(data, horizon)
            logger.info(f"   ✅ Preprocessing complete")
            
            # 2. Predict
            raw_predictions = self.predict(preprocessed, horizon)
            logger.info(f"   ✅ Inference complete")
            
            # 3. Postprocess
            result = self.postprocess(raw_predictions)
            logger.info(f"   ✅ Postprocessing complete")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            raise
    
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return model capabilities for selection logic"""
        return {
            'model_name': self.metadata.model_name,
            'model_type': self.metadata.model_type.value,
            'supports_multistep': self.metadata.supports_multistep,
            'supports_exogenous': self.metadata.supports_exogenous,
            'max_horizon': self.metadata.max_horizon,
            'required_features': self.metadata.required_features,
            'frequency': self.metadata.frequency
        }


class ModelExecutorRegistry:
    """
    Registry pattern for model executors
    Automatically discovers and registers available executors
    """
    
    def __init__(self):
        self._executors: Dict[str, type] = {}
        self._instances: Dict[str, BaseModelExecutor] = {}
    
    def register(self, model_type: ModelType, executor_class: type):
        """Register an executor class for a model type"""
        if not issubclass(executor_class, BaseModelExecutor):
            raise TypeError(f"{executor_class} must inherit from BaseModelExecutor")
        
        self._executors[model_type.value] = executor_class
        logger.info(f"✅ Registered executor for {model_type.value}")
    
    def get_executor(self, metadata: ModelMetadata) -> BaseModelExecutor:
        """Get or create executor instance for a model"""
        cache_key = f"{metadata.model_name}_{metadata.version}"
        
        if cache_key in self._instances:
            return self._instances[cache_key]
        
        executor_class = self._executors.get(metadata.model_type.value)
        if not executor_class:
            raise ValueError(f"No executor registered for {metadata.model_type}")
        
        executor = executor_class(metadata)
        self._instances[cache_key] = executor
        return executor
    
    def list_supported_types(self) -> List[str]:
        """List all supported model types"""
        return list(self._executors.keys())


# Global registry instance
executor_registry = ModelExecutorRegistry()


# Helper function for automatic registration
def register_executor(model_type: ModelType):
    """Decorator for automatic executor registration"""
    def decorator(cls):
        executor_registry.register(model_type, cls)
        return cls
    return decorator