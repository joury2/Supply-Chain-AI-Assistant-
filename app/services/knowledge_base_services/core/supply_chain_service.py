# app/services/knowledge_base_services/core/supply_chain_service.py
# Enhanced Main Supply Chain Forecasting Service with Monitoring, Caching, and Configuration
import os
import sys
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.services.knowledge_base_services.core.knowledge_base_service import SupplyChainService as KnowledgeBaseService
    from app.services.knowledge_base_services.core.rule_engine_service import RuleEngineService
    from app.services.model_serving.model_registry_service import ModelRegistryService
    
    # Try to import LLM service, but provide fallback if not available
    try:
        from app.services.llm.interpretation_service import LLMInterpretationService
        LLM_AVAILABLE = True
    except ImportError:
        logger.warning("LLMInterpretationService not found, using mock implementation")
        LLM_AVAILABLE = False
        
    from app.core.data_processor import DataProcessor
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Create mock classes for demonstration
    class KnowledgeBaseService:
        def __init__(self, db_path):
            logger.info(f"Mock KnowledgeBaseService initialized with {db_path}")
        def get_all_models(self): return []
        def get_dataset_schema(self, name): return None
        def validate_dataset(self, name, columns): return {"valid": True, "errors": []}
        def get_suitable_models(self, features, target): return []
        def get_active_rules(self): return []
        def create_forecast_run(self, *args): return {}
        def close(self): pass
    
    class RuleEngineService:
        def __init__(self): 
            logger.info("Mock RuleEngineService initialized")
        def analyze_dataset(self, dataset): 
            return {
                'validation': {'valid': True},
                'model_selection': {'selected_model': 'Mock_Model', 'confidence': 0.8},
                'summary': {'can_proceed': True}
            }
    
    class ModelRegistryService:
        def __init__(self): 
            logger.info("Mock ModelRegistryService initialized")
        def load_model(self, name): 
            return type('MockModel', (), {'name': name})()
    
    class DataProcessor:
        def __init__(self): 
            logger.info("Mock DataProcessor initialized")
        def prepare_data(self, dataset, model): 
            return {'status': 'processed'}
    
    # Mock LLM service
    class LLMInterpretationService:
        def __init__(self): 
            logger.info("Mock LLMInterpretationService initialized")
        def interpret_forecast(self, forecast, analysis, context): 
            return {'summary': 'Mock interpretation'}


class PipelineMetrics:
    """Track performance metrics for the forecasting pipeline"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_forecasts': 0,
            'validation_failures': 0,
            'model_selection_failures': 0,
            'model_loading_failures': 0,
            'pipeline_errors': 0,
            'average_processing_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'service_health': {
                'knowledge_base': 'healthy',
                'rule_engine': 'healthy', 
                'model_registry': 'healthy',
                'llm_service': 'healthy'
            }
        }
        self.start_time = time.time()
    
    def record_pipeline_execution(self, start_time: float, success: bool, failure_reason: str = None):
        """Record pipeline execution metrics"""
        execution_time = time.time() - start_time
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_forecasts'] += 1
        elif failure_reason:
            failure_key = f'{failure_reason}_failures'
            if failure_key in self.metrics:
                self.metrics[failure_key] += 1
            else:
                self.metrics['pipeline_errors'] += 1
        
        # Update average processing time using running average
        current_avg = self.metrics['average_processing_time']
        total_requests = self.metrics['total_requests']
        self.metrics['average_processing_time'] = (
            (current_avg * (total_requests - 1) + execution_time) / total_requests
        )
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics['cache_misses'] += 1
    
    def update_service_health(self, service: str, status: str):
        """Update health status of a service"""
        if service in self.metrics['service_health']:
            self.metrics['service_health'][service] = status
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        uptime = time.time() - self.start_time
        success_rate = (self.metrics['successful_forecasts'] / self.metrics['total_requests']) * 100 if self.metrics['total_requests'] > 0 else 0
        cache_hit_rate = (self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])) * 100 if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        
        return {
            **self.metrics,
            'uptime_seconds': uptime,
            'success_rate_percent': success_rate,
            'cache_hit_rate_percent': cache_hit_rate,
            'requests_per_minute': (self.metrics['total_requests'] / uptime) * 60 if uptime > 0 else 0
        }


class PipelineConfig:
    """Configurable pipeline settings"""
    
    def __init__(self):
        self.settings = {
            'enable_rule_engine': True,
            'enable_knowledge_base': True,
            'enable_llm_interpretation': True,
            'min_confidence_threshold': 0.6,
            'max_processing_time': 300,  # 5 minutes
            'fallback_strategy': 'best_available',  # or 'fail_fast'
            'log_level': 'INFO',
            'enable_caching': True,
            'cache_ttl_seconds': 300,  # 5 minutes
            'enable_metrics': True,
            'validation_strictness': 'strict'  # or 'lenient'
        }
    
    def update_from_environment(self):
        """Update settings from environment variables"""
        env_mappings = {
            'ENABLE_RULE_ENGINE': ('enable_rule_engine', lambda x: x.lower() == 'true'),
            'ENABLE_KNOWLEDGE_BASE': ('enable_knowledge_base', lambda x: x.lower() == 'true'),
            'ENABLE_LLM': ('enable_llm_interpretation', lambda x: x.lower() == 'true'),
            'MIN_CONFIDENCE': ('min_confidence_threshold', float),
            'MAX_PROCESSING_TIME': ('max_processing_time', int),
            'FALLBACK_STRATEGY': ('fallback_strategy', str),
            'ENABLE_CACHING': ('enable_caching', lambda x: x.lower() == 'true'),
            'CACHE_TTL': ('cache_ttl_seconds', int),
            'VALIDATION_STRICTNESS': ('validation_strictness', str)
        }
        
        for env_var, (setting_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    self.settings[setting_key] = converter(env_value)
                    logger.info(f"📋 Updated {setting_key} from environment: {self.settings[setting_key]}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Failed to parse {env_var}={env_value}: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.settings.get(key, default)


class SupplyChainForecastingService:
    """
    Enhanced Main Supply Chain Forecasting Service
    With monitoring, caching, and configuration
    """
    
    def __init__(self, db_path: str = "supply_chain.db", config: PipelineConfig = None):
        logger.info("🚀 Initializing Enhanced Supply Chain Service...")
        
        # Initialize configuration
        self.config = config or PipelineConfig()
        self.config.update_from_environment()
        
        # Initialize metrics
        self.metrics = PipelineMetrics()
        
        # Initialize all services with database integration
        self.knowledge_base = KnowledgeBaseService(db_path)
        self.rule_engine = RuleEngineService()
        self.model_registry = ModelRegistryService()
        
        # Initialize LLM service with fallback
        if LLM_AVAILABLE and self.config.get('enable_llm_interpretation', True):
            self.llm_service = LLMInterpretationService()
            self.metrics.update_service_health('llm_service', 'healthy')
        else:
            self.llm_service = LLMInterpretationService()  # Use the mock version
            self.metrics.update_service_health('llm_service', 'mock')
            
        self.data_processor = DataProcessor()
        
        # Initialize caching
        self._analysis_cache = {}
        self._model_cache = {}
        self._schema_cache = {}
        
        # State management
        self.current_analysis = None
        self.current_forecast = None
        self.current_interpretation = None
        
        logger.info("✅ Enhanced Supply Chain Service initialized successfully")
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data dictionary"""
        data_str = str(sorted(data.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def validate_forecasting_request(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize forecasting request inputs"""
        validation_errors = []
        warnings = []
        
        # Required fields validation
        required_fields = ['name', 'columns', 'row_count']
        for field in required_fields:
            if field not in dataset_info:
                validation_errors.append(f"Missing required field: {field}")
        
        # Data type validation
        if not isinstance(dataset_info.get('columns', []), list):
            validation_errors.append("Columns must be a list")
        
        if not isinstance(dataset_info.get('row_count', 0), int):
            validation_errors.append("Row count must be an integer")
        
        # Business logic validation
        row_count = dataset_info.get('row_count', 0)
        if row_count < 1:
            validation_errors.append("Row count must be positive")
        elif row_count < 12:
            warnings.append("Limited data points may affect forecast accuracy")
        
        columns = dataset_info.get('columns', [])
        if not columns:
            validation_errors.append("Dataset must have at least one column")
        elif len(columns) < 2:
            warnings.append("Dataset has very few columns, consider adding more features")
        
        # Frequency validation
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'none']
        frequency = dataset_info.get('frequency', 'none')
        if frequency not in valid_frequencies:
            warnings.append(f"Uncommon frequency: {frequency}. Expected: {valid_frequencies}")
        
        # Sanitize inputs
        sanitized_info = dataset_info.copy()
        sanitized_info['name'] = str(sanitized_info.get('name', '')).strip()[:100]
        sanitized_info['columns'] = [str(col).strip() for col in sanitized_info.get('columns', [])]
        
        # Ensure numeric fields are within reasonable bounds
        sanitized_info['missing_percentage'] = max(0.0, min(1.0, float(sanitized_info.get('missing_percentage', 0.0))))
        sanitized_info['row_count'] = max(1, int(sanitized_info.get('row_count', 0)))
        
        return {
            'valid': len(validation_errors) == 0,
            'errors': validation_errors,
            'warnings': warnings,
            'sanitized_data': sanitized_info
        }
    
    def process_forecasting_request(self, 
                                  dataset_info: Dict[str, Any],
                                  forecast_horizon: int = 30,
                                  business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced forecasting pipeline with monitoring and validation
        """
        start_time = time.time()
        logger.info(f"📦 Processing enhanced forecasting request for: {dataset_info.get('name', 'Unknown')}")
        
        try:
            # Step 0: Validate and sanitize request
            validation_result = self.validate_forecasting_request(dataset_info)
            if not validation_result['valid']:
                self.metrics.record_pipeline_execution(start_time, False, 'validation')
                return self._create_validation_failed_response(validation_result)
            
            sanitized_dataset = validation_result['sanitized_data']
            
            # Check processing time limit
            if time.time() - start_time > self.config.get('max_processing_time', 300):
                self.metrics.record_pipeline_execution(start_time, False, 'timeout')
                return self._create_timeout_response()
            
            # Step 1: Comprehensive analysis with caching
            analysis_result = self.analyze_dataset_with_knowledge_base(sanitized_dataset)
            
            # Step 2: Check if we can proceed
            if not analysis_result['combined_summary']['can_proceed']:
                self.metrics.record_pipeline_execution(start_time, False, 'validation')
                return self._create_validation_failed_response_from_analysis(analysis_result)
            
            # Step 3: Select best model (combining rule engine and knowledge base)
            selected_model = self._select_best_model(analysis_result, sanitized_dataset)
            if not selected_model:
                self.metrics.record_pipeline_execution(start_time, False, 'model_selection')
                return self._create_model_selection_failed_response(analysis_result)
            
            # Check confidence threshold
            min_confidence = self.config.get('min_confidence_threshold', 0.6)
            if selected_model.get('confidence', 0) < min_confidence:
                logger.warning(f"⚠️ Model confidence {selected_model['confidence']:.3f} below threshold {min_confidence}")
                if self.config.get('fallback_strategy') == 'fail_fast':
                    self.metrics.record_pipeline_execution(start_time, False, 'low_confidence')
                    return self._create_low_confidence_response(selected_model, analysis_result)
            
            # Step 4: Load and prepare the selected model
            model = self.model_registry.load_model(selected_model['model_name'])
            if model is None:
                self.metrics.record_pipeline_execution(start_time, False, 'model_loading')
                return self._create_model_loading_failed_response(selected_model['model_name'], analysis_result)
            
            # Step 5: Process data using knowledge base schema
            processed_data = self._prepare_data_with_schema(sanitized_dataset, selected_model)
            forecast_result = self._generate_forecast(model, processed_data, forecast_horizon)
            self.current_forecast = forecast_result
            
            # Step 6: Record forecast run in knowledge base
            forecast_run = self._record_forecast_run(selected_model, sanitized_dataset, forecast_result)
            
            # Step 7: Generate plots
            plots = self._generate_plots(forecast_result, sanitized_dataset)
            
            # Step 8: LLM interpretation with business context (if enabled)
            interpretation = None
            if self.config.get('enable_llm_interpretation', True):
                interpretation = self.llm_service.interpret_forecast(
                    forecast_result, 
                    analysis_result,
                    business_context
                )
                self.current_interpretation = interpretation
            
            # Step 9: Compile final response
            result = self._create_success_response(
                analysis_result, 
                selected_model,
                forecast_result, 
                interpretation, 
                plots,
                forecast_run
            )
            
            self.metrics.record_pipeline_execution(start_time, True)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced forecasting pipeline: {str(e)}")
            self.metrics.record_pipeline_execution(start_time, False, 'pipeline_error')
            return self._create_error_response(str(e))
    
    def analyze_dataset_with_knowledge_base(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced dataset analysis using both rule engine and knowledge base with caching
        """
        # Check cache if enabled
        if self.config.get('enable_caching', True):
            cache_key = self._generate_cache_key(dataset_info)
            if cache_key in self._analysis_cache:
                cache_entry = self._analysis_cache[cache_key]
                cache_ttl = self.config.get('cache_ttl_seconds', 300)
                if time.time() - cache_entry['timestamp'] < cache_ttl:
                    logger.info("📊 Returning cached analysis")
                    self.metrics.record_cache_hit()
                    return cache_entry['result']
                else:
                    # Remove expired cache entry
                    del self._analysis_cache[cache_key]
            
            self.metrics.record_cache_miss()
        
        logger.info(f"🔍 Analyzing dataset with knowledge base: {dataset_info.get('name', 'Unknown')}")
        
        # Step 1: Rule-based analysis
        rule_analysis = self.rule_engine.analyze_dataset(dataset_info)
        
        # Step 2: Knowledge base validation
        kb_validation = self._validate_with_knowledge_base(dataset_info)
        
        # Step 3: Model recommendations from knowledge base
        kb_recommendations = self._get_knowledge_base_recommendations(dataset_info)
        
        # Combine results
        combined_analysis = {
            'rule_analysis': rule_analysis,
            'knowledge_base_validation': kb_validation,
            'knowledge_base_recommendations': kb_recommendations,
            'combined_summary': self._combine_analyses(rule_analysis, kb_validation, kb_recommendations)
        }
        
        self.current_analysis = combined_analysis
        
        # Cache the result if enabled
        if self.config.get('enable_caching', True):
            self._analysis_cache[cache_key] = {
                'result': combined_analysis,
                'timestamp': time.time()
            }
        
        return combined_analysis
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics"""
        return self.metrics.get_metrics_summary()
    
    def clear_cache(self):
        """Clear all caches"""
        self._analysis_cache.clear()
        self._model_cache.clear()
        self._schema_cache.clear()
        logger.info("🧹 Cleared all service caches")
    
    def update_config(self, new_settings: Dict[str, Any]):
        """Update pipeline configuration"""
        for key, value in new_settings.items():
            if key in self.config.settings:
                old_value = self.config.settings[key]
                self.config.settings[key] = value
                logger.info(f"⚙️ Updated config {key}: {old_value} → {value}")
            else:
                logger.warning(f"⚠️ Unknown config key: {key}")
    
    # Keep all existing methods from previous implementation
    def _validate_with_knowledge_base(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset against knowledge base schemas"""
        dataset_name = dataset_info.get('name', 'custom_dataset')
        provided_columns = dataset_info.get('columns', [])
        
        # Try to find matching schema in knowledge base
        schema = self.knowledge_base.get_dataset_schema(dataset_name)
        
        if schema:
            # Validate against known schema
            return self.knowledge_base.validate_dataset(dataset_name, provided_columns)
        else:
            # No matching schema found, use generic validation
            return {
                "valid": True,
                "errors": [],
                "schema_name": "custom_schema",
                "required_columns": [],
                "provided_columns": provided_columns,
                "note": "Using custom dataset schema"
            }
    
    def _get_knowledge_base_recommendations(self, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get model recommendations from knowledge base"""
        available_features = dataset_info.get('columns', [])
        target_variable = self._infer_target_variable(dataset_info)
        
        # Get suitable models from knowledge base
        suitable_models = self.knowledge_base.get_suitable_models(available_features, target_variable)
        
        recommendations = []
        for model in suitable_models:
            recommendations.append({
                'source': 'knowledge_base',
                'model_name': model['model_name'],
                'model_type': model['model_type'],
                'target_variable': model['target_variable'],
                'confidence': self._calculate_kb_confidence(model, dataset_info),
                'reason': f"Matches available features and target '{target_variable}'"
            })
        
        return recommendations
    
    def _infer_target_variable(self, dataset_info: Dict[str, Any]) -> str:
        """Infer target variable from dataset columns"""
        columns = dataset_info.get('columns', [])
        
        # Common target variable names
        target_candidates = ['sales', 'demand', 'quantity', 'value', 'target', 'y']
        
        for candidate in target_candidates:
            if candidate in [col.lower() for col in columns]:
                return candidate
        
        # Default to first numeric-looking column
        return columns[0] if columns else 'unknown'
    
    def _calculate_kb_confidence(self, model: Dict[str, Any], dataset_info: Dict[str, Any]) -> float:
        """Calculate confidence score for knowledge base recommendations"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on feature match
        required_features = model.get('required_features', [])
        if isinstance(required_features, str):
            try:
                required_features = eval(required_features)
            except:
                required_features = []
        
        available_features = dataset_info.get('columns', [])
        feature_match_ratio = len(set(required_features) & set(available_features)) / max(len(required_features), 1)
        confidence += feature_match_ratio * 0.3
        
        # Adjust based on data size
        row_count = dataset_info.get('row_count', 0)
        if row_count > 1000:
            confidence += 0.1
        elif row_count < 50:
            confidence -= 0.2
        
        return min(confidence, 1.0)
    
    def _select_best_model(self, analysis_result: Dict[str, Any], dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select best model combining rule engine and knowledge base recommendations"""
        rule_model = analysis_result['rule_analysis']['model_selection']
        kb_recommendations = analysis_result['knowledge_base_recommendations']
        
        # If rule engine selected a model, use it (highest priority)
        if rule_model.get('selected_model'):
            return {
                'model_name': rule_model['selected_model'],
                'source': 'rule_engine',
                'confidence': rule_model.get('confidence', 0.0),
                'reason': rule_model.get('reason', 'Rule-based selection')
            }
        
        # Otherwise, use knowledge base recommendation
        if kb_recommendations:
            best_kb = max(kb_recommendations, key=lambda x: x.get('confidence', 0))
            return {
                'model_name': best_kb['model_name'],
                'source': 'knowledge_base',
                'confidence': best_kb.get('confidence', 0.0),
                'reason': best_kb.get('reason', 'Knowledge base recommendation')
            }
        
        return None
    
    def _prepare_data_with_schema(self, dataset_info: Dict[str, Any], selected_model: Dict[str, Any]) -> Any:
        """Prepare data using knowledge base schema information"""
        model_name = selected_model['model_name']
        
        # Get model requirements from knowledge base
        all_models = self.knowledge_base.get_all_models()
        model_details = next((m for m in all_models if m['model_name'] == model_name), None)
        
        if model_details:
            required_features = model_details.get('required_features', [])
            if isinstance(required_features, str):
                try:
                    required_features = eval(required_features)
                except:
                    required_features = []
            
            logger.info(f"Model {model_name} requires features: {required_features}")
        
        # Use data processor with model-specific preparation
        return self.data_processor.prepare_data(dataset_info, model_name)
    
    def _record_forecast_run(self, selected_model: Dict[str, Any], 
                           dataset_info: Dict[str, Any],
                           forecast_result: Dict[str, Any]) -> Dict[str, Any]:
        """Record forecast run in knowledge base"""
        try:
            # Get model ID from knowledge base
            all_models = self.knowledge_base.get_all_models()
            model_details = next((m for m in all_models if m['model_name'] == selected_model['model_name']), None)
            
            if model_details:
                forecast_config = {
                    'horizon': forecast_result.get('horizon', 30),
                    'model_source': selected_model['source'],
                    'confidence': selected_model['confidence'],
                    'dataset_characteristics': {
                        'row_count': dataset_info.get('row_count'),
                        'columns': dataset_info.get('columns', []),
                        'frequency': dataset_info.get('frequency')
                    }
                }
                
                forecast_run = self.knowledge_base.create_forecast_run(
                    model_id=model_details['model_id'],
                    input_schema=dataset_info.get('name', 'custom_dataset'),
                    config=forecast_config
                )
                
                logger.info(f"📝 Recorded forecast run ID: {forecast_run.get('run_id')}")
                return forecast_run
        
        except Exception as e:
            logger.warning(f"Could not record forecast run in knowledge base: {e}")
        
        return {}
    
    def _combine_analyses(self, rule_analysis: Dict, kb_validation: Dict, kb_recommendations: List) -> Dict[str, Any]:
        """Combine results from rule engine and knowledge base"""
        rule_summary = rule_analysis.get('summary', {})
        kb_valid = kb_validation.get('valid', True)
        
        can_proceed = (
            rule_summary.get('can_proceed', False) and 
            kb_valid and
            (rule_analysis['model_selection'].get('selected_model') or kb_recommendations)
        )
        
        # Combine recommendations
        all_recommendations = []
        if rule_analysis.get('recommendations'):
            all_recommendations.extend(rule_analysis['recommendations'])
        
        if kb_validation.get('errors'):
            all_recommendations.extend([{'message': error, 'priority': 'high'} for error in kb_validation['errors']])
        
        # Calculate combined confidence
        rule_confidence = rule_analysis['model_selection'].get('confidence', 0.0)
        kb_confidence = max([r.get('confidence', 0.0) for r in kb_recommendations] or [0.0])
        combined_confidence = max(rule_confidence, kb_confidence)
        
        return {
            'can_proceed': can_proceed,
            'confidence': combined_confidence,
            'primary_issue': self._identify_primary_issue(rule_summary, kb_validation),
            'recommendation_count': len(all_recommendations),
            'sources_used': ['rule_engine', 'knowledge_base']
        }
    
    def _identify_primary_issue(self, rule_summary: Dict, kb_validation: Dict) -> Optional[str]:
        """Identify the most critical issue"""
        if rule_summary.get('primary_issue'):
            return rule_summary['primary_issue']
        elif not kb_validation.get('valid'):
            return "Knowledge base validation failed"
        return None
    
    def get_available_schemas(self) -> List[Dict[str, Any]]:
        """Get all available dataset schemas from knowledge base"""
        # This would query your knowledge base for all schemas
        # For now, return mock data
        return [
            {
                'name': 'demand_forecasting',
                'description': 'Standard demand forecasting dataset',
                'required_columns': ['date', 'demand', 'product_id']
            },
            {
                'name': 'inventory_optimization', 
                'description': 'Inventory management dataset',
                'required_columns': ['date', 'inventory_level', 'product_id', 'lead_time']
            }
        ]
    
    def get_business_rules(self) -> List[Dict[str, Any]]:
        """Get all active business rules from knowledge base"""
        return self.knowledge_base.get_active_rules()
    
    # Keep all the existing helper methods from previous version
    def _generate_forecast(self, model, processed_data: Any, horizon: int) -> Dict[str, Any]:
        """Generate forecasts using the selected model"""
        logger.info(f"📊 Generating forecast for {horizon} periods")
        
        try:
            # Mock forecast generation
            forecast_values = np.random.normal(100, 10, horizon).tolist()
            confidence_intervals = {
                'lower': [v * 0.9 for v in forecast_values],
                'upper': [v * 1.1 for v in forecast_values]
            }
            
            return {
                'values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'horizon': horizon,
                'model_used': getattr(model, 'name', 'unknown'),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Forecast generation failed: {str(e)}")
            raise Exception(f"Forecast generation failed: {str(e)}")
    
    def _generate_plots(self, forecast_result: Dict[str, Any], dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization plots for the forecast"""
        return {
            'forecast_plot': {
                'type': 'line',
                'title': f"Forecast for {dataset_info.get('name', 'Dataset')}",
                'description': 'Forecast with confidence intervals'
            },
            'components_plot': {
                'type': 'decomposition', 
                'title': 'Forecast Components',
                'description': 'Trend, seasonality, and residuals'
            }
        }
    
    def _create_success_response(self, analysis: Dict, selected_model: Dict, forecast: Dict, 
                               interpretation: Dict, plots: Dict, forecast_run: Dict) -> Dict[str, Any]:
        """Create success response"""
        return {
            'status': 'success',
            'analysis': analysis,
            'selected_model': selected_model,
            'forecast': forecast,
            'interpretation': interpretation,
            'visualizations': plots,
            'forecast_run_id': forecast_run.get('run_id'),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _create_validation_failed_response(self, validation_result: Dict) -> Dict[str, Any]:
        """Create validation failed response"""
        return {
            'status': 'validation_failed',
            'error': 'Request validation failed',
            'validation_errors': validation_result['errors'],
            'warnings': validation_result['warnings'],
            'recommendations': ['Fix the validation errors above and retry']
        }
    
    def _create_validation_failed_response_from_analysis(self, analysis: Dict) -> Dict[str, Any]:
        """Create validation failed response from analysis"""
        return {
            'status': 'validation_failed',
            'analysis': analysis,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': analysis['combined_summary']['primary_issue'],
            'recommendations': analysis.get('rule_analysis', {}).get('recommendations', [])
        }
    
    
    def _create_model_selection_failed_response(self, analysis: Dict) -> Dict[str, Any]:
        """Create model selection failed response"""
        return {
            'status': 'model_selection_failed',
            'analysis': analysis,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': "No suitable model found",
            'recommendations': ["Consider adding more features", "Check data quality requirements"]
        }
    
    def _create_model_loading_failed_response(self, model_name: str, analysis: Dict) -> Dict[str, Any]:
        """Create model loading failed response"""
        return {
            'status': 'model_loading_failed',
            'analysis': analysis,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': f"Failed to load model: {model_name}",
            'recommendations': [
                "Check if the model is properly registered",
                "Verify model dependencies are installed",
                "Contact system administrator"
            ]
        }
    
    def _create_low_confidence_response(self, selected_model: Dict, analysis: Dict) -> Dict[str, Any]:
        """Create low confidence response"""
        return {
            'status': 'low_confidence',
            'analysis': analysis,
            'selected_model': selected_model,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': f"Model confidence {selected_model['confidence']:.3f} below threshold {self.config.get('min_confidence_threshold', 0.6)}",
            'recommendations': [
                "Consider collecting more data",
                "Try a different model configuration",
                "Adjust confidence threshold if appropriate"
            ]
        }
    
    def _create_timeout_response(self) -> Dict[str, Any]:
        """Create timeout response"""
        return {
            'status': 'timeout',
            'error': f"Processing exceeded maximum time limit of {self.config.get('max_processing_time', 300)} seconds",
            'recommendations': [
                "Try with a smaller dataset",
                "Reduce forecast horizon",
                "Contact system administrator for performance tuning"
            ]
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create generic error response"""
        return {
            'status': 'error',
            'analysis': None,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': error_message,
            'recommendations': [
                "Check the dataset format and quality",
                "Verify all required columns are present", 
                "Contact support if the issue persists"
            ]
        }
    
    def close(self):
        """Close all services"""
        self.knowledge_base.close()
        logger.info("🔚 Supply Chain Service closed")


# Enhanced Demo function
def demo_enhanced_supply_chain_service():
    """Demonstrate with all new features"""
    print("🚀 ENHANCED SUPPLY CHAIN SERVICE WITH MONITORING & CACHING")
    print("=" * 60)
    
    # Initialize service with custom config
    config = PipelineConfig()
    config.settings['min_confidence_threshold'] = 0.5  # Lower threshold for demo
    config.settings['enable_caching'] = True
    
    service = SupplyChainForecastingService(config=config)
    
    test_datasets = [
        {
            'name': 'simple_demand_forecasting',
            'frequency': 'monthly',
            'granularity': 'product_level', 
            'row_count': 120,
            'columns': ['date', 'demand'],
            'missing_percentage': 0.03,
            'description': 'Simple demand data for time series models'
        },
        {
            'name': 'invalid_dataset',  # This should fail validation
            'row_count': 5,  # Too few rows
            'columns': [],  # No columns
            'description': 'Invalid dataset for testing validation'
        }
    ]
    
    for i, test_dataset in enumerate(test_datasets, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}: {test_dataset['description']}")
        print(f"{'='*50}")
        
        print("📊 Testing enhanced forecasting pipeline...")
        
        # Process forecasting request
        result = service.process_forecasting_request(
            dataset_info=test_dataset,
            forecast_horizon=12,
            business_context={'industry': 'retail'}
        )
        
        # Display results
        print(f"\n🎯 Result Status: {result['status']}")
        
        if result['status'] == 'success':
            print("✅ Enhanced forecasting pipeline completed successfully!")
            print(f"🤖 Model Used: {result['selected_model']['model_name']}")
            print(f"📊 Source: {result['selected_model']['source']}")
            print(f"🎯 Confidence: {result['selected_model']['confidence']:.1%}")
            
        elif result['status'] == 'validation_failed':
            print("❌ Validation failed")
            print(f"🚨 Errors: {result.get('validation_errors', [result.get('error', 'Unknown error')])}")
        
        elif result['status'] == 'low_confidence':
            print("⚠️ Low confidence warning")
            print(f"📉 Confidence: {result['selected_model']['confidence']:.1%}")
    
    # Show metrics and cache info
    print(f"\n📈 PIPELINE METRICS:")
    print("-" * 40)
    metrics = service.get_pipeline_metrics()
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Success Rate: {metrics['success_rate_percent']:.1f}%")
    print(f"Cache Hit Rate: {metrics['cache_hit_rate_percent']:.1f}%")
    print(f"Avg Processing Time: {metrics['average_processing_time']:.2f}s")
    
    # Show configuration
    print(f"\n⚙️  PIPELINE CONFIGURATION:")
    print("-" * 40)
    for key, value in service.config.settings.items():
        print(f"  {key}: {value}")
    
    # Test cache functionality
    print(f"\n💾 CACHE TEST:")
    print("-" * 40)
    test_dataset = test_datasets[0]  # Use the valid dataset
    print("First analysis (should cache):")
    result1 = service.analyze_dataset_with_knowledge_base(test_dataset)
    print("Second analysis (should use cache):")
    result2 = service.analyze_dataset_with_knowledge_base(test_dataset)
    
    # Clear cache and test again
    service.clear_cache()
    print("After cache clear:")
    result3 = service.analyze_dataset_with_knowledge_base(test_dataset)
    
    service.close()
    print(f"\n🎉 Enhanced Supply Chain Service Demo Completed!")

if __name__ == "__main__":
    demo_enhanced_supply_chain_service()