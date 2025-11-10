# app/services/inference/model_inference_service.py
"""
Unified Model Inference Service
Uses the plugin architecture to support any model type without code changes
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import os
from pathlib import Path

from app.services.inference.base_model_executor import (
    BaseModelExecutor, ModelMetadata, ForecastResult,
    ModelType, executor_registry
)
from app.services.knowledge_base_services.core.knowledge_base_service import SupplyChainService as KnowledgeBaseService

logger = logging.getLogger(__name__)


class ModelInferenceService:
    """
    Unified service for model inference
    Works with ANY model type through the executor registry
    """
    
    def __init__(self, model_storage_path: str = "models/"):
        self.model_storage_path = Path(model_storage_path)
        self._loaded_executors: Dict[str, BaseModelExecutor] = {}
        self.executor_registry = executor_registry

        logger.info("🚀 Model Inference Service initialized")
        logger.info(f"   Supported model types: {executor_registry.list_supported_types()}")

        # FIX A: Expose global executor_registry to callers/tests
        self.executor_registry = executor_registry

        logger.info("🚀 Model Inference Service initialized")
        logger.info(f"   Supported model types: {executor_registry.list_supported_types()}")

        # 🔧 FIXED: Store knowledge base reference
        self.knowledge_base = KnowledgeBaseService
        


        logger.info("🚀 Model Inference Service initialized")
        logger.info(f"   Model storage: {self.model_storage_path}")
        logger.info(f"   Supported types: {executor_registry.list_supported_types()}")
        logger.info(f"   Knowledge base: {'✅ Connected' if self.knowledge_base else '⚠️ Not connected'}")

    def set_knowledge_base(self, knowledge_base_service):
        """
        Set knowledge base service after initialization
        Allows for deferred dependency injection
        """
        self.knowledge_base = knowledge_base_service
        logger.info("✅ Knowledge base service connected")


    def load_model_from_metadata(self, metadata: ModelMetadata) -> bool:
        """
        Load a model using its metadata
        FIXED: Proper error handling when knowledge_base not available
        """
        try:
            # Get appropriate executor from registry
            executor = executor_registry.get_executor(metadata)
            
            # Determine model path
            # Priority: 1) From metadata, 2) From knowledge base, 3) Construct
            model_path = None
            
            # Try to get path from knowledge base if available
            if self.knowledge_base:
                try:
                    model_info = self.knowledge_base.get_model_by_name(metadata.model_name)
                    if model_info and 'model_path' in model_info:
                        model_path = model_info['model_path']
                        logger.info(f"📂 Using path from knowledge base: {model_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get path from knowledge base: {e}")
            
            # Fallback: construct path
            if not model_path:
                # Try different extensions based on model type
                extensions = {
                    ModelType.LIGHTGBM: ['.txt', '.lgb', '.model'],
                    ModelType.PROPHET: ['.pkl', '.pickle', '.json'],
                    ModelType.LSTM: ['.h5', '.keras', '.pt'],
                    ModelType.XGBOOST: ['.json', '.model', '.ubj']
                }
                
                possible_extensions = extensions.get(metadata.model_type, ['.pkl'])
                
                for ext in possible_extensions:
                    candidate = self.model_storage_path / f"{metadata.model_name}_v{metadata.version}{ext}"
                    if candidate.exists():
                        model_path = str(candidate)
                        logger.info(f"📂 Found model file: {model_path}")
                        break
                
                if not model_path:
                    # Last resort: try without version
                    for ext in possible_extensions:
                        candidate = self.model_storage_path / f"{metadata.model_name}{ext}"
                        if candidate.exists():
                            model_path = str(candidate)
                            logger.info(f"📂 Found model file (no version): {model_path}")
                            break
            
            # Validate path exists
            if not model_path:
                logger.error(f"❌ Model file not found for {metadata.model_name}")
                logger.error(f"   Searched in: {self.model_storage_path}")
                logger.error(f"   Expected patterns: {metadata.model_name}_v{metadata.version}.*")
                return False
            
            # Make path absolute if relative
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.getcwd(), model_path)
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file does not exist: {model_path}")
                return False
            
            # Load the model
            logger.info(f"📥 Loading model from: {model_path}")
            success = executor.load_model(model_path)
            
            if success:
                self._loaded_executors[metadata.model_name] = executor
                logger.info(f"✅ Loaded {metadata.model_name} ({metadata.model_type.value})")
            else:
                logger.error(f"❌ Executor failed to load model: {metadata.model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to load {metadata.model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_model_from_registry(self, model_name: str, 
                                knowledge_base_service=None) -> bool:
        """
        Load model directly from knowledge base information
        FIXED: Can accept knowledge_base_service as parameter or use stored one
        """
        try:
            # Use provided service or stored one
            kb_service = knowledge_base_service or self.knowledge_base
            
            if not kb_service:
                logger.error("❌ Knowledge base service not available")
                logger.error("   Call set_knowledge_base() first or pass knowledge_base_service parameter")
                return False
            
            # Get model info from knowledge base
            model_info = kb_service.get_model_by_name(model_name)
            
            if not model_info:
                logger.error(f"❌ Model {model_name} not found in knowledge base")
                return False
            
            logger.info(f"📋 Model info retrieved: {model_info['model_name']}")
            
            # Convert to metadata
            metadata = self._model_info_to_metadata(model_info)
            
            # Load using metadata
            return self.load_model_from_metadata(metadata)
            
        except Exception as e:
            logger.error(f"❌ Failed to load from registry: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_forecast(self, model_name: str, data: pd.DataFrame, 
                     horizon: int = 30) -> Optional[ForecastResult]:
        """Generate forecast using a loaded model - WITH DETAILED LOGGING"""
        logger.info(f"🎯 INFERENCE_START: model={model_name}, data_shape={data.shape}, horizon={horizon}")
        
        try:
            # Get executor for this model
            executor = self._loaded_executors.get(model_name)
            
            if not executor:
                logger.error(f"❌ INFERENCE_ERROR: Model {model_name} not loaded")
                logger.error(f"   Currently loaded: {self.list_loaded_models()}")
                return None
            
            logger.info(f"📊 INFERENCE_DATA_CHECK: data_type={type(data).__name__}, "
                    f"columns={list(data.columns)[:5] if hasattr(data, 'columns') else 'no_columns'}")
            
            # Check data compatibility
            capabilities = executor.get_capabilities()
            logger.info(f"🔧 MODEL_CAPABILITIES: {capabilities}")
            
            # Execute the forecast
            logger.info(f"🚀 EXECUTING_FORECAST: calling executor.execute()")
            result = executor.execute(data, horizon)
            
            if result is None:
                logger.error("❌ EXECUTION_FAILED: executor.execute() returned None")
                return None
            
            logger.info(f"✅ EXECUTION_SUCCESS: predictions={len(result.predictions)}, "
                    f"type={type(result).__name__}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ INFERENCE_EXCEPTION: {str(e)}")
            import traceback
            logger.error(f"🔍 INFERENCE_TRACEBACK:\n{traceback.format_exc()}")
            return None

    def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get capabilities of a loaded model"""
        executor = self._loaded_executors.get(model_name)
        if executor:
            return executor.get_capabilities()
        return None
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded and ready"""
        return model_name in self._loaded_executors
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self._loaded_executors:
            del self._loaded_executors[model_name]
            logger.info(f"🗑️  Unloaded {model_name}")
    
    def list_loaded_models(self) -> list:
        """List all currently loaded models"""
        return list(self._loaded_executors.keys())
    
    def _model_info_to_metadata(self, model_info: Dict[str, Any]) -> ModelMetadata:
        """Convert knowledge base model info to ModelMetadata"""
        # Parse model type
        model_type_str = model_info.get('model_type', 'custom').lower()
        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            logger.warning(f"Unknown model type: {model_type_str}, using CUSTOM")
            model_type = ModelType.CUSTOM
        
        # Parse required features
        required_features = model_info.get('required_features', [])
        if isinstance(required_features, str):
            import json
            try:
                required_features = json.loads(required_features)
            except:
                try:
                    import ast
                    required_features = ast.literal_eval(required_features)
                except:
                    logger.warning(f"Could not parse required_features: {required_features}")
                    required_features = []
        
        # Parse optional features
        optional_features = model_info.get('optional_features', [])
        if isinstance(optional_features, str):
            import json
            try:
                optional_features = json.loads(optional_features)
            except:
                optional_features = []
        
        return ModelMetadata(
            model_name=model_info['model_name'],
            model_type=model_type,
            version=model_info.get('version', '1.0'),
            target_column=model_info.get('target_variable', 'y'),
            required_features=required_features,
            optional_features=optional_features,
            frequency=model_info.get('frequency', 'daily'),
            supports_multistep=True,
            supports_exogenous=True,
            max_horizon=model_info.get('max_horizon', 365)
        )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for troubleshooting"""
        return {
            'model_storage_path': str(self.model_storage_path),
            'model_storage_exists': self.model_storage_path.exists(),
            'loaded_models': self.list_loaded_models(),
            'loaded_model_count': len(self._loaded_executors),
            'supported_types': self.executor_registry.list_supported_types(),
            'has_knowledge_base': self.knowledge_base is not None,
            'knowledge_base_type': type(self.knowledge_base).__name__ if self.knowledge_base else None
        }    

    # Model-Specific Logging

    # def generate_forecast(self, model_name: str, data: pd.DataFrame, 
    #                     horizon: int = 30) -> Optional[ForecastResult]:
    #     """Generate forecast using a loaded model - WITH DETAILED LOGGING"""
    #     logger.info(f"🎯 INFERENCE_START: model={model_name}, data_shape={data.shape}, horizon={horizon}")
        
    #     try:
    #         # Get executor for this model
    #         executor = self._loaded_executors.get(model_name)
            
    #         if not executor:
    #             logger.error(f"❌ INFERENCE_ERROR: Model {model_name} not loaded")
    #             logger.error(f"   Currently loaded: {self.list_loaded_models()}")
    #             return None
            
    #         logger.info(f"📊 INFERENCE_DATA_CHECK: data_type={type(data).__name__}, "
    #                 f"columns={list(data.columns)[:5] if hasattr(data, 'columns') else 'no_columns'}")
            
    #         # Check data compatibility
    #         capabilities = executor.get_capabilities()
    #         logger.info(f"🔧 MODEL_CAPABILITIES: {capabilities}")
            
    #         # Execute the forecast
    #         logger.info(f"🚀 EXECUTING_FORECAST: calling executor.execute()")
    #         result = executor.execute(data, horizon)
            
    #         if result is None:
    #             logger.error("❌ EXECUTION_FAILED: executor.execute() returned None")
    #             return None
            
    #         logger.info(f"✅ EXECUTION_SUCCESS: predictions={len(result.predictions)}, "
    #                 f"type={type(result).__name__}")
            
    #         return result
            
    #     except Exception as e:
    #         logger.error(f"❌ INFERENCE_EXCEPTION: {str(e)}")
    #         import traceback
    #         logger.error(f"🔍 INFERENCE_TRACEBACK:\n{traceback.format_exc()}")
    #         return None


#--------------------------------
#---------old-------
    # def load_model_from_metadata(self, metadata: ModelMetadata) -> bool:
    #     """Load a model using its metadata - FIXED VERSION"""
    #     try:
    #         # Get appropriate executor from registry
    #         executor = executor_registry.get_executor(metadata)
            
    #         # Get the CORRECT model path from knowledge base
    #         model_info = self.knowledge_base.get_model_by_name(metadata.model_name)
    #         if not model_info:
    #             logger.error(f"Model {metadata.model_name} not found in knowledge base")
    #             return False
                
    #         # Use the ACTUAL path from database, not constructed path
    #         model_path = model_info['model_path']
            
    #         # Load the model
    #         if not os.path.exists(model_path):
    #             logger.error(f"Model file not found: {model_path}")
    #             # Try relative path from current directory
    #             model_path = os.path.join(os.getcwd(), model_path)
    #             if not os.path.exists(model_path):
    #                 logger.error(f"Model file still not found: {model_path}")
    #                 return False
            
    #         success = executor.load_model(model_path)
            
    #         if success:
    #             self._loaded_executors[metadata.model_name] = executor
    #             logger.info(f"✅ Loaded {metadata.model_name} from {model_path}")
            
    #         return success
            
    #     except Exception as e:
    #         logger.error(f"❌ Failed to load {metadata.model_name}: {e}")
    #         return False

    # def load_model_from_registry(self, model_name: str, 
    #                             knowledge_base_service) -> bool:
    #     """
    #     Load model directly from knowledge base information
    #     """
    #     try:
    #         # Get model info from knowledge base
    #         model_info = knowledge_base_service.get_model_by_name(model_name)
            
    #         if not model_info:
    #             logger.error(f"Model {model_name} not found in knowledge base")
    #             return False
            
    #         # Convert to metadata
    #         metadata = self._model_info_to_metadata(model_info)
            
    #         # Load using metadata
    #         return self.load_model_from_metadata(metadata)
            
    #     except Exception as e:
    #         logger.error(f"❌ Failed to load from registry: {e}")
    #         return False
    
    # def generate_forecast(self, model_name: str, data: pd.DataFrame, 
    #                      horizon: int = 30) -> Optional[ForecastResult]:
    #     """
    #     Generate forecast using a loaded model
    #     This is the MAIN method the supply chain service calls
    #     """
    #     logger.info(f"📊 Generating forecast: {model_name} for {horizon} periods")
        
    #     try:
    #         # Get executor for this model
    #         executor = self._loaded_executors.get(model_name)
            
    #         if not executor:
    #             logger.error(f"Model {model_name} not loaded")
    #             return None
            
    #         # Execute the forecast (all model-specific logic is in the executor)
    #         result = executor.execute(data, horizon)
            
    #         logger.info(f"✅ Forecast complete: {len(result.predictions)} predictions")
    #         return result
            
    #     except Exception as e:
    #         logger.error(f"❌ Forecast generation failed: {e}")
    #         import traceback
    #         logger.error(traceback.format_exc())
    #         return None
    
    # def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
    #     """Get capabilities of a loaded model"""
    #     executor = self._loaded_executors.get(model_name)
    #     if executor:
    #         return executor.get_capabilities()
    #     return None
    
    # def is_model_loaded(self, model_name: str) -> bool:
    #     """Check if a model is loaded and ready"""
    #     return model_name in self._loaded_executors
    
    # def unload_model(self, model_name: str):
    #     """Unload a model to free memory"""
    #     if model_name in self._loaded_executors:
    #         del self._loaded_executors[model_name]
    #         logger.info(f"🗑️  Unloaded {model_name}")
    
    # def list_loaded_models(self) -> list:
    #     """List all currently loaded models"""
    #     return list(self._loaded_executors.keys())
    
    # def _model_info_to_metadata(self, model_info: Dict[str, Any]) -> ModelMetadata:
    #     """Convert knowledge base model info to ModelMetadata"""
    #     # Parse model type
    #     model_type_str = model_info.get('model_type', 'custom').lower()
    #     try:
    #         model_type = ModelType(model_type_str)
    #     except ValueError:
    #         model_type = ModelType.CUSTOM
        
    #     # Parse required features
    #     required_features = model_info.get('required_features', [])
    #     if isinstance(required_features, str):
    #         import json
    #         try:
    #             required_features = json.loads(required_features)
    #         except:
    #             required_features = []
        
    #     return ModelMetadata(
    #         model_name=model_info['model_name'],
    #         model_type=model_type,
    #         version=model_info.get('version', '1.0'),
    #         target_column=model_info.get('target_variable', 'y'),
    #         required_features=required_features,
    #         optional_features=model_info.get('optional_features', []),
    #         frequency=model_info.get('frequency', 'daily'),
    #         supports_multistep=True,  # Default
    #         supports_exogenous=True,
    #         max_horizon=model_info.get('max_horizon', 365)
    #     )

