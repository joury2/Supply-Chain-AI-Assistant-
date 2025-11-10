# --------------------------------------------------
# ---------------- NEW verion of the code ---------
# --------------------------------------------------
# app/services/knowledge_base_services/core/supply_chain_service.py
# REFACTORED VERSION - Using Plugin Architecture
"""
Supply Chain Forecasting Service
Now uses the unified inference service - NO model-specific code needed!
"""

import logging
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import hashlib
import json  # Added missing import

from app.services.knowledge_base_services.core.knowledge_base_service import SupplyChainService as KnowledgeBaseService
from app.services.knowledge_base_services.core.rule_engine_service import RuleEngineService
from app.services.llm.interpretation_service import LLMInterpretationService
from app.services.transformation.feature_engineering import ForecastDataPreprocessor

# NEW: Import the unified inference service
from app.services.inference.model_inference_service import ModelInferenceService

logger = logging.getLogger(__name__)


class SupplyChainForecastingService:
    """
    Supply Chain Forecasting Service - Clean Architecture Version
    Separates concerns: business logic vs model inference
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        logger.info("🚀 Initializing Supply Chain Service (Clean Architecture)...")
        
        # Configuration
        self.enable_caching = True
        self.cache_ttl_seconds = 300
        self.min_confidence_threshold = 0.6
        
        # Business logic services
        self.knowledge_base = KnowledgeBaseService(db_path)
        self.rule_engine = RuleEngineService()
        self.llm_service = LLMInterpretationService()
        
        # NEW: Unified inference service (handles ALL model types)
        # 🔧 FIXED: Pass knowledge_base to inference service
        # self.inference_service = ModelInferenceService(
        #     model_storage_path="models/",
        #     knowledge_base_service=self.knowledge_base  # Inject dependency
        # )

        self.inference_service = ModelInferenceService(
        model_storage_path="models/"
        )
        
        # Set the knowledge base separately
        self.inference_service.set_knowledge_base(self.knowledge_base)
        
        
        # Data preprocessing service
        self.data_preprocessor = ForecastDataPreprocessor()
        
        # Caching
        self._analysis_cache = {}
        
        # State management
        self.current_analysis = None
        self.current_forecast = None
        self.current_interpretation = None
        
        logger.info("✅ Supply Chain Service initialized successfully")
    

    def process_forecasting_request(self, 
                                  dataset_info: Dict[str, Any],
                                  forecast_horizon: int = 30,
                                  business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        REFACTORED forecasting pipeline
        Now all model-specific code is in the inference service!
        """
        start_time = time.time()
        logger.info(f"📦 Processing forecasting request for: {dataset_info.get('name', 'Unknown')}")
        
        try:
            # Step 1: Validate request
            validation_result = self.validate_forecasting_request(dataset_info)
            if not validation_result['valid']:
                return self._create_validation_failed_response(validation_result)
            
            sanitized_dataset = validation_result['sanitized_data']
            
            # Step 2: Analyze dataset (business logic)
            analysis_result = self.analyze_dataset_with_knowledge_base(sanitized_dataset)
            
            if not analysis_result['combined_summary']['can_proceed']:
                return self._create_validation_failed_response_from_analysis(analysis_result)
            
            # Step 3: Select best model (business logic)
            selected_model = self._select_best_model(analysis_result, sanitized_dataset)
            if not selected_model:
                return self._create_model_selection_failed_response(analysis_result)
            
            if selected_model.get('confidence', 0) < self.min_confidence_threshold:
                logger.warning(f"⚠️ Model confidence {selected_model['confidence']:.3f} below threshold")
                return self._create_low_confidence_response(selected_model, analysis_result)
            
            # Step 4: Load model using inference service
            model_name = selected_model['model_name']
            if not self.inference_service.is_model_loaded(model_name):
                logger.info(f"📥 Loading model: {model_name}")
                loaded = self.inference_service.load_model_from_registry(
                    model_name, 
                    self.knowledge_base
                )
                if not loaded:
                    return self._create_model_loading_failed_response(model_name, analysis_result)
            
            # Step 5: Prepare data (feature engineering)
            user_data = sanitized_dataset.get('data')
            if user_data is None:
                return self._create_error_response("No data provided for forecasting")
            
            processed_data = self.data_preprocessor.prepare_data_for_model(
                user_data=user_data,
                model_name=model_name,
                forecast_horizon=forecast_horizon
            )
            
            if processed_data is None:
                return self._create_error_response(f"Failed to prepare data for {model_name}")
            
            logger.info(f"✅ Data prepared: {processed_data.shape}")
            
            # Step 6: Generate forecast using inference service
            # THIS IS WHERE THE MAGIC HAPPENS - ONE LINE FOR ANY MODEL!
            forecast_result = self.inference_service.generate_forecast(
                model_name=model_name,
                data=processed_data,
                horizon=forecast_horizon
            )
            
            if forecast_result is None:
                return self._create_error_response("Forecast generation failed")
            
            # Convert ForecastResult to dict format
            forecast_dict = self._forecast_result_to_dict(forecast_result)
            self.current_forecast = forecast_dict
            
            # Step 7: LLM interpretation
            interpretation = self.llm_service.interpret_forecast(
                forecast_dict, 
                analysis_result,
                business_context
            )
            self.current_interpretation = interpretation
            
            # Step 8: Compile final response
            result = self._create_success_response(
                analysis_result, 
                selected_model,
                forecast_dict, 
                interpretation
            )
            
            logger.info(f"✅ Forecast completed in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in forecasting pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_error_response(str(e))

    def _forecast_result_to_dict(self, forecast_result) -> Dict[str, Any]:
        """Convert ForecastResult object to dict format"""
        return {
            'values': forecast_result.predictions.tolist(),
            'confidence_intervals': {
                'lower': forecast_result.confidence_lower.tolist(),
                'upper': forecast_result.confidence_upper.tolist()
            },
            'horizon': len(forecast_result.predictions),
            'model_used': forecast_result.metadata.get('model_name', 'unknown'),
            'model_type': forecast_result.metadata.get('model_type', 'unknown'),
            'timestamp': forecast_result.metadata.get('timestamp', 
                                                     pd.Timestamp.now().isoformat()),
            'metadata': forecast_result.metadata
        }

    # ============================================================================
    # MAIN BUSINESS LOGIC METHODS (KEEP ONLY ONE VERSION OF EACH)
    # ============================================================================

    def validate_forecasting_request(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize forecasting request inputs - SINGLE VERSION"""
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

    def analyze_dataset_with_knowledge_base(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dataset analysis - BUSINESS LOGIC ONLY - SINGLE VERSION
        No model inference code here!
        """
        # Extract DataFrame before caching
        dataset_copy = dataset_info.copy()
        dataset_df = dataset_copy.pop('data', None)
        
        # Check cache
        if self.enable_caching:
            cache_key = self._generate_cache_key(dataset_info)
            if cache_key in self._analysis_cache:
                cache_entry = self._analysis_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl_seconds:
                    logger.info("📊 Returning cached analysis")
                    return cache_entry['result']
        
        logger.info(f"🔍 Analyzing dataset: {dataset_info.get('name', 'Unknown')}")
        
        if dataset_df is not None:
            dataset_info['data'] = dataset_df
        
        try:
            # Business logic: validation
            validation_result = self.rule_engine.validate_dataset(dataset_info)
            
            # Business logic: get available models
            available_models = self.knowledge_base.get_all_models()
            active_models = [m for m in available_models if m.get('is_active', True)]
            
            if not active_models:
                return self._create_no_models_response(validation_result, dataset_info)
            
            # Business logic: model selection
            selected_model_result = self.rule_engine.rule_engine.select_model(dataset_info)
            
            if selected_model_result and selected_model_result.get('selected_model'):
                primary_model_name = selected_model_result['selected_model']
                primary_model = next((m for m in active_models 
                                    if m.get('model_name') == primary_model_name), None)
                
                if primary_model:
                    model_recommendations = self._create_recommendations_with_primary(
                        primary_model, active_models, selected_model_result
                    )
                else:
                    model_recommendations = self._create_fallback_recommendations(
                        active_models, dataset_info
                    )
            else:
                model_recommendations = self._create_fallback_recommendations(
                    active_models, dataset_info
                )
            
            # Business logic: validation and recommendations
            kb_validation = self._validate_with_knowledge_base(dataset_info)
            kb_recommendations = self._get_knowledge_base_recommendations(dataset_info)
            
            rule_analysis = {
                'validation': validation_result,
                'model_selection': self._format_model_selection(model_recommendations),
                'summary': self._create_rule_summary(validation_result, model_recommendations)
            }
            
            combined_analysis = {
                'rule_analysis': rule_analysis,
                'knowledge_base_validation': kb_validation,
                'knowledge_base_recommendations': kb_recommendations,
                'combined_summary': self._combine_analyses(rule_analysis, kb_validation, kb_recommendations)
            }
            
            self.current_analysis = combined_analysis
            
            # Cache result
            if self.enable_caching:
                cache_data = {k: v for k, v in combined_analysis.items() if k != 'data'}
                self._analysis_cache[cache_key] = {
                    'result': cache_data,
                    'timestamp': time.time()
                }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._create_fallback_analysis(dataset_info)

    # Add this to your supply_chain_service.py in the forecasting pipeline

    def _validate_model_requirements(self, model_name: str, data: pd.DataFrame, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that data meets model requirements and provide clear guidance"""
        missing_required = []
        missing_optional = []
        
        # Check required columns
        for col in requirements.get('required_columns', []):
            if col not in data.columns:
                missing_required.append(col)
        
        # Check required regressors  
        for regressor in requirements.get('required_regressors', []):
            if regressor not in data.columns:
                missing_required.append(regressor)
        
        # Check optional regressors
        for regressor in requirements.get('optional_regressors', []):
            if regressor not in data.columns:
                missing_optional.append(regressor)
        
        if missing_required:
            return {
                'valid': False,
                'error_type': 'MISSING_REQUIRED_DATA',
                'missing_columns': missing_required,
                'missing_optional': missing_optional,
                'guidance': self._generate_data_guidance(model_name, missing_required, missing_optional)
            }
        
        return {'valid': True}

    def _generate_data_guidance(self, model_name: str, missing_required: list, missing_optional: list) -> str:
        """Generate clear user guidance for missing data"""
        guidance = []
        
        if missing_required:
            guidance.append("❌ **Missing Required Data:**")
            for col in missing_required:
                if col in ['unique_customers', 'avg_revenue']:
                    guidance.append(f"   - '{col}': This column is required for the trained model but not in your dataset.")
                    guidance.append(f"     → **Solution**: Add a '{col}' column to your data or use a different model.")
                elif col in ['ds', 'y']:
                    guidance.append(f"   - '{col}': This is a core Prophet column.")
                    guidance.append(f"     → **Solution**: Ensure your data has date and target columns.")
        
        if missing_optional:
            guidance.append("⚠️ **Missing Optional Data:**")
            for col in missing_optional:
                guidance.append(f"   - '{col}': This may improve forecast accuracy.")
        
        # Model-specific guidance
        if model_name == "Supply_Chain_Prophet_Forecaster":
            guidance.append("\n💡 **Prophet Model Notes:**")
            guidance.append("   - This model was trained with additional business metrics")
            guidance.append("   - For best results, include: unique_customers, avg_revenue")
            guidance.append("   - Alternatively, use Daily_Shop_Sales_Forecaster for basic sales data")
        
        return "\n".join(guidance)

    # ============================================================================
    # HELPER METHODS (ADD MISSING ONES)
    # ============================================================================

    def _generate_cache_key(self, dataset_info: Dict[str, Any]) -> str:
        """Generate reproducible cache key for analysis requests."""
        # Use name + columns + row_count + missing_percentage for simplicity
        name = dataset_info.get('name', 'dataset')
        cols = ",".join(sorted([str(c) for c in dataset_info.get('columns', [])]))
        rows = str(dataset_info.get('row_count', 0))
        miss = str(round(float(dataset_info.get('missing_percentage', 0.0)), 4))
        return f"{name}::cols={cols}::rows={rows}::miss={miss}"

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Standardized error response for forecasting requests."""
        logger.error(f" Creating error response: {message}")
        return {
            'status': 'error',
            'error': message,
            'analysis': None,
            'selected_model': None,
            'forecast': None,
            'interpretation': None,
            'timestamp': pd.Timestamp.now().isoformat()
        }

    # ============================================================================
    # EXISTING HELPER METHODS (KEEP THESE)
    # ============================================================================

    def _create_fallback_analysis(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback analysis when real analysis fails"""
        logger.warning("🔄 Using fallback analysis")
        
        return {
            'rule_analysis': {
                'validation': {'valid': True, 'errors': [], 'warnings': []},
                'model_selection': {
                    'selected_model': 'lightgbm_demand_forecaster',
                    'confidence': 0.5,
                    'reason': 'Fallback selection'
                },
                'summary': {'can_proceed': True, 'primary_issue': None}
            },
            'knowledge_base_validation': {'valid': True, 'errors': []},
            'knowledge_base_recommendations': [],
            'combined_summary': {
                'can_proceed': True,
                'confidence': 0.5,
                'primary_issue': 'Fallback analysis used',
                'sources_used': ['fallback']
            }
        }

    def _create_recommendations_with_primary(self, primary_model: Dict[str, Any], 
                                        all_models: List[Dict[str, Any]],
                                        selection_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create recommendations with a primary model selected by RuleEngine"""
        recommendations = []
        
        # Add primary model first
        primary_recommendation = {
            'model_name': primary_model['model_name'],
            'model_type': primary_model.get('model_type', 'unknown'),
            'score': 95,  # High score for rule-selected model
            'confidence': selection_result.get('confidence', 0.8),
            'reasons': [selection_result.get('reason', 'Rule-based selection')],
            'status': 'primary',
            'rule_name': selection_result.get('rule_name'),
            'source': 'rule_engine'
        }
        recommendations.append(primary_recommendation)
        
        # Add other active models as alternatives
        for model in all_models:
            if model['model_name'] != primary_model['model_name']:
                alternative_recommendation = {
                    'model_name': model['model_name'],
                    'model_type': model.get('model_type', 'unknown'),
                    'score': 50,  # Medium score for alternatives
                    'confidence': 0.5,
                    'reasons': ['Active alternative model'],
                    'status': 'alternative',
                    'source': 'knowledge_base'
                }
                recommendations.append(alternative_recommendation)
        
        return recommendations

    def _calculate_compatibility_score(self, model: Dict[str, Any], 
                                    dataset_info: Dict[str, Any]) -> int:
        """Calculate compatibility score (0-100) between model and dataset"""
        score = 50  # Base score
        
        model_name = model.get('model_name', '').lower()
        dataset_columns = dataset_info.get('columns', [])
        frequency = dataset_info.get('frequency', '')
        row_count = dataset_info.get('row_count', 0)
        
        # Check required features compatibility
        required_features = model.get('required_features', [])
        if isinstance(required_features, str):
            try:
                required_features = eval(required_features)
            except:
                required_features = []
        
        # Calculate feature match ratio
        if required_features:
            matched_features = len(set(required_features) & set(dataset_columns))
            feature_ratio = matched_features / len(required_features)
            score += int(feature_ratio * 30)  # Up to 30 points for feature matching
        
        # Check target variable
        target_variable = model.get('target_variable', '')
        if target_variable and target_variable in dataset_columns:
            score += 20  # Bonus for target variable match
        
        # Model-specific compatibility checks
        if 'prophet' in model_name and frequency in ['daily', 'weekly', 'monthly']:
            score += 15
        elif 'lightgbm' in model_name and len(dataset_columns) > 3:
            score += 10
        elif 'arima' in model_name and frequency != 'none':
            score += 10
        
        return min(score, 100)

    def _find_most_compatible_model(self, models: List[Dict[str, Any]], 
                                dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the most compatible model based on dataset characteristics"""
        if not models:
            return None
        
        best_model = None
        best_score = 0
        
        for model in models:
            compatibility_score = self._calculate_compatibility_score(model, dataset_info)
            
            if compatibility_score > best_score:
                best_score = compatibility_score
                best_model = model
        
        # Only return if we found a reasonably compatible model
        return best_model if best_score >= 50 else None

    # ... [KEEP ALL OTHER EXISTING HELPER METHODS - they're unique]
    # _get_model_performance_score, get_suitable_models, _parse_required_features,
    def _get_model_performance_score(self, model: Dict[str, Any]) -> float:
        """Extract performance score from model metrics (lower is better)"""
        try:
            metrics = json.loads(model['performance_metrics'])
            # Prefer MAPE (Mean Absolute Percentage Error) - lower is better
            return metrics.get('MAPE', float('inf'))
        except:
            return float('inf')
    
    def get_suitable_models(self, available_features: List[str], target_variable: str, 
                       sort_by_performance: bool = True) -> List[Dict[str, Any]]:
        """
        Find models compatible with available features - FIXED VERSION
        """
        try:
            # Get ALL models from knowledge base (data access)
            all_models = self.knowledge_base.get_all_models(use_cache=True)
            
            suitable_models = []
            for model in all_models:
                # FIXED: Better feature parsing
                required_features = self._parse_required_features(model['required_features'])
                
                # Check compatibility (business logic)
                if (all(feature in available_features for feature in required_features) and
                    model['target_variable'] == target_variable):
                    suitable_models.append(model)
            
            # Sort by performance if requested (business logic)
            if sort_by_performance and suitable_models:
                suitable_models.sort(
                    key=lambda x: self._get_model_performance_score(x)
                )
                logger.info(f"✅ Found {len(suitable_models)} suitable models (sorted by performance)")
            else:
                logger.info(f"✅ Found {len(suitable_models)} suitable models")
            
            return suitable_models
            
        except Exception as e:
            logger.error(f"❌ Error finding suitable models: {e}")
            return []

    def _parse_required_features(self, features_data) -> List[str]:
        """Parse required features from various formats"""
        if isinstance(features_data, list):
            return features_data
        elif isinstance(features_data, str):
            try:
                # Try JSON parsing first
                import json
                return json.loads(features_data)
            except json.JSONDecodeError:
                try:
                    # Try Python literal eval as fallback
                    import ast
                    return ast.literal_eval(features_data)
                except:
                    # Last resort: split by comma
                    return [f.strip() for f in features_data.split(',')]
        else:
            return []
    
    # _create_fallback_recommendations, _validate_with_knowledge_base, 
    def _create_fallback_recommendations(self, active_models: List[Dict[str, Any]],
                                    dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback recommendations when RuleEngine doesn't select a model"""
        if not active_models:
            return []
        
        recommendations = []
        
        # Try to find the most compatible model based on dataset characteristics
        best_model = self._find_most_compatible_model(active_models, dataset_info)
        
        if best_model:
            # Found a compatible model
            recommendations.append({
                'model_name': best_model['model_name'],
                'model_type': best_model.get('model_type', 'unknown'),
                'score': 70,  # Good score for compatible model
                'confidence': 0.7,
                'reasons': ['Most compatible model based on dataset characteristics'],
                'status': 'compatible',
                'source': 'fallback_analysis'
            })
            
            # Add other models as lower-priority alternatives
            for model in active_models:
                if model['model_name'] != best_model['model_name']:
                    recommendations.append({
                        'model_name': model['model_name'],
                        'model_type': model.get('model_type', 'unknown'),
                        'score': 40,  # Lower score for other models
                        'confidence': 0.4,
                        'reasons': ['Available alternative'],
                        'status': 'alternative',
                        'source': 'knowledge_base'
                    })
        else:
            # No compatible models found - return all as low-confidence options
            for model in active_models:
                recommendations.append({
                    'model_name': model['model_name'],
                    'model_type': model.get('model_type', 'unknown'),
                    'score': 30,  # Low score when no good match
                    'confidence': 0.3,
                    'reasons': ['Limited compatibility - manual review recommended'],
                    'status': 'low_confidence',
                    'source': 'fallback'
                })
        
        return recommendations

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

    # _calculate_kb_confidence, _infer_target_variable, _get_knowledge_base_recommendations,
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
    
    # need better logic - letter on fix the logic of _infer_target_variable
    def _infer_target_variable(self, dataset_info: Dict[str, Any]) -> str:
        """Intelligently infer target variable from dataset columns"""
        columns = dataset_info.get('columns', [])
        
        # Priority order for target detection
        target_priority = ['sales', 'demand', 'quantity', 'value', 'target', 'revenue', 'volume']
        
        for target in target_priority:
            if target in columns:
                return target
        
        # If no standard target found, look for numeric columns
        numeric_indicators = ['amount', 'count', 'total', 'qty']
        for indicator in numeric_indicators:
            for column in columns:
                if indicator in column.lower():
                    return column
        
        # Fallback: use first column that's not a date
        for column in columns:
            if 'date' not in column.lower() and 'time' not in column.lower():
                return column
        
        return columns[0] if columns else 'unknown'

    def _get_knowledge_base_recommendations(self, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get model recommendations from knowledge base"""
        available_features = dataset_info.get('columns', [])
        target_variable = self._infer_target_variable(dataset_info)
        
        # Get suitable models from knowledge base
        suitable_models = self.get_suitable_models(available_features, target_variable)
        
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

    # _format_model_selection, _create_rule_summary, _identify_primary_issue,
    def _format_model_selection(self, model_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format model recommendations for the expected structure"""
        if not model_recommendations:
            return {'selected_model': None, 'confidence': 0.0, 'reason': 'No recommendations'}
        
        # Get the top recommendation
        top_model = model_recommendations[0]
        
        return {
            'selected_model': top_model['model_name'],
            'confidence': top_model.get('confidence', 0.0),
            'reason': ', '.join(top_model.get('reasons', ['Rule-based selection'])),
            'all_recommendations': model_recommendations
        }
    
    def _create_rule_summary(self, validation_result: Dict[str, Any], model_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary from validation and model recommendations"""
        can_proceed = validation_result.get('valid', False) and len(model_recommendations) > 0
        
        primary_issue = None
        if not validation_result.get('valid', False):
            primary_issue = "Dataset validation failed"
        elif not model_recommendations:
            primary_issue = "No suitable models found"
        
        return {
            'can_proceed': can_proceed,
            'primary_issue': primary_issue,
            'validation_score': len(validation_result.get('applied_rules', [])),
            'model_count': len(model_recommendations)
        }
    
    def _identify_primary_issue(self, rule_summary: Dict, kb_validation: Dict) -> Optional[str]:
        """Identify the main issue preventing forecasting"""
        if not rule_summary.get('can_proceed', False):
            return rule_summary.get('primary_issue', 'Rule engine validation failed')
        
        if not kb_validation.get('valid', True):
            errors = kb_validation.get('errors', [])
            return f"Knowledge base validation failed: {errors[0] if errors else 'Unknown error'}"
        
        return None
    
    # _combine_analyses, _select_best_model, _create_*_response methods,
    def _combine_analyses(self, rule_analysis: Dict, kb_validation: Dict, kb_recommendations: List) -> Dict[str, Any]:
        """Combine results from rule engine and knowledge base"""
        rule_summary = rule_analysis.get('summary', {})
        kb_valid = kb_validation.get('valid', True)
        
        can_proceed = (
            rule_summary.get('can_proceed', False) and 
            kb_valid and
            (rule_analysis['model_selection'].get('selected_model') or kb_recommendations)
        )
        
        # Calculate combined confidence
        rule_confidence = rule_analysis['model_selection'].get('confidence', 0.0)
        kb_confidence = max([r.get('confidence', 0.0) for r in kb_recommendations] or [0.0])
        combined_confidence = max(rule_confidence, kb_confidence)
        
        return {
            'can_proceed': can_proceed,
            'confidence': combined_confidence,
            'primary_issue': self._identify_primary_issue(rule_summary, kb_validation),
            'recommendation_count': len(kb_recommendations),
            'sources_used': ['rule_engine', 'knowledge_base']
        }

    def _select_best_model(self, analysis_result: Dict[str, Any], dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Select the best model from analysis results
        """
        try:
            selection = analysis_result['rule_analysis']['model_selection']
            if selection.get('selected_model'):
                return {
                    'model_name': selection['selected_model'],
                    'confidence': selection.get('confidence', 0.0),
                    'reason': selection.get('reason', 'Selected by rule engine')
                }
            
            # Fallback: use first compatible model
            compatible_models = analysis_result.get('rule_analysis', {}).get('selection_analysis', {}).get('analysis', {}).get('compatible_models', [])
            if compatible_models:
                best_model = compatible_models[0]
                return {
                    'model_name': best_model.get('model_name', 'Unknown'),
                    'confidence': best_model.get('compatibility_score', 50) / 100,
                    'reason': 'Best compatible model'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
            return None

    # - _create_*_response methods
    def _create_low_confidence_response(self, selected_model: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when model confidence is too low"""
        return {
            'status': 'low_confidence',
            'analysis': analysis_result,
            'selected_model': selected_model,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': f"Model confidence {selected_model['confidence']:.1%} below threshold {self.min_confidence_threshold:.1%}",
            'recommendations': [
                "Consider collecting more historical data",
                "Try uploading a dataset with more features",
                "Adjust business constraints to allow more model options",
                "Contact support for model customization"
            ]
        }

    def _create_validation_failed_response(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when validation fails"""
        return {
            'status': 'validation_failed',
            'analysis': None,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': f"Dataset validation failed: {', '.join(validation_result['errors'])}",
            'recommendations': [
                "Fix the validation errors listed above",
                "Ensure all required columns are present",
                "Check data quality and formatting",
                "Re-upload the corrected dataset"
            ]
        }

    def _create_validation_failed_response_from_analysis(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when analysis validation fails"""
        return {
            'status': 'validation_failed',
            'analysis': analysis_result,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': "Dataset analysis failed validation checks",
            'recommendations': [
                "Review the analysis results for specific issues",
                "Check if your data meets minimum requirements",
                "Consider using a different dataset structure",
                "Contact support for data preparation guidance"
            ]
        }

    def _create_model_selection_failed_response(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when no suitable model is found"""
        return {
            'status': 'no_suitable_model',
            'analysis': analysis_result,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': "No suitable forecasting model found for your dataset",
            'recommendations': [
                "Check if your dataset has the required features",
                "Ensure your target variable is properly identified",
                "Try uploading data with different characteristics",
                "Contact support for model compatibility analysis"
            ]
        }

    def _create_model_loading_failed_response(self, model_name: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when model fails to load"""
        return {
            'status': 'model_loading_failed',
            'analysis': analysis_result,
            'forecast': None,
            'interpretation': None,
            'visualizations': None,
            'error': f"Failed to load model: {model_name}",
            'recommendations': [
                f"Check if model {model_name} is properly installed",
                "Verify model registry configuration",
                "Try selecting a different model",
                "Contact administrator for model deployment"
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
    
    def _create_no_models_response(self, validation_result: Dict[str, Any], 
                                dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when no models are available"""
        return {
            'rule_analysis': {
                'validation': validation_result,
                'model_selection': {
                    'selected_model': None,
                    'confidence': 0.0,
                    'reason': 'No active models available in knowledge base'
                },
                'summary': {
                    'can_proceed': False,
                    'primary_issue': 'NO_ACTIVE_MODELS',
                    'confidence': 0.0
                }
            },
            'knowledge_base_validation': {'valid': False, 'errors': ['No active models available']},
            'knowledge_base_recommendations': [],
            'combined_summary': {
                'can_proceed': False,
                'confidence': 0.0,
                'primary_issue': 'No forecasting models available',
                'recommendations': [
                    'Check if any models are marked as active in the knowledge base',
                    'Contact administrator to activate forecasting models',
                    'Verify model registry configuration'
                ]
            }
        }
    
    def _create_success_response(self, analysis_result: Dict, selected_model: Dict, 
                            forecast_result: Dict, interpretation: Dict) -> Dict[str, Any]:
        """Create success response - SIMPLIFIED VERSION to match actual usage"""
        return {
            'status': 'success',
            'analysis': analysis_result,
            'selected_model': selected_model,
            'forecast': forecast_result,
            'interpretation': interpretation,
            'visualizations': {'available': True, 'type': 'forecast_plot'},
            'timestamp': pd.Timestamp.now().isoformat() if 'pd' in globals() else '2024-01-01T00:00:00'
        }

    # Execution Flow Logging
    def _log_forecasting_execution_flow(self, step: str, details: Dict[str, Any] = None):
        """Comprehensive logging for forecasting execution flow"""
        logger.info(f"🔍 FORECASTING_FLOW [{step}]")
        
        if details:
            for key, value in details.items():
                if key == 'data' and value is not None:
                    logger.info(f"   📊 {key}: shape={getattr(value, 'shape', 'No shape')}, "
                            f"type={type(value).__name__}")
                elif isinstance(value, (list, tuple)) and len(value) > 5:
                    logger.info(f"   📊 {key}: len={len(value)}, first_3={value[:3]}")
                elif isinstance(value, dict) and len(value) > 3:
                    logger.info(f"   📊 {key}: keys={list(value.keys())[:5]}")
                else:
                    logger.info(f"   📊 {key}: {value}")

    def process_forecasting_request(self, 
                                dataset_info: Dict[str, Any],
                                forecast_horizon: int = 30,
                                business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        REFACTORED forecasting pipeline WITH COMPREHENSIVE LOGGING
        """
        start_time = time.time()
        logger.info(f"🚀 START Forecasting Pipeline")
        self._log_forecasting_execution_flow("START_REQUEST", {
            'dataset_name': dataset_info.get('name'),
            'horizon': forecast_horizon,
            'dataset_keys': list(dataset_info.keys())
        })
        
        try:
            # Step 1: Validate request
            self._log_forecasting_execution_flow("VALIDATION_START")
            validation_result = self.validate_forecasting_request(dataset_info)
            self._log_forecasting_execution_flow("VALIDATION_COMPLETE", {
                'valid': validation_result['valid'],
                'errors': validation_result['errors'],
                'warnings': validation_result['warnings']
            })
            
            if not validation_result['valid']:
                return self._create_validation_failed_response(validation_result)
            
            sanitized_dataset = validation_result['sanitized_data']
            
            # Step 2: Analyze dataset
            self._log_forecasting_execution_flow("ANALYSIS_START")
            analysis_result = self.analyze_dataset_with_knowledge_base(sanitized_dataset)
            self._log_forecasting_execution_flow("ANALYSIS_COMPLETE", {
                'can_proceed': analysis_result['combined_summary']['can_proceed'],
                'confidence': analysis_result['combined_summary'].get('confidence'),
                'selected_model': analysis_result['rule_analysis']['model_selection'].get('selected_model')
            })
            
            if not analysis_result['combined_summary']['can_proceed']:
                return self._create_validation_failed_response_from_analysis(analysis_result)
            
            # Step 3: Select best model
            self._log_forecasting_execution_flow("MODEL_SELECTION_START")
            selected_model = self._select_best_model(analysis_result, sanitized_dataset)
            self._log_forecasting_execution_flow("MODEL_SELECTED", {
                'model_name': selected_model['model_name'],
                'confidence': selected_model['confidence'],
                'reason': selected_model.get('reason')
            })
            
            if not selected_model:
                return self._create_model_selection_failed_response(analysis_result)
            
            if selected_model.get('confidence', 0) < self.min_confidence_threshold:
                logger.warning(f"⚠️ Model confidence {selected_model['confidence']:.3f} below threshold")
                return self._create_low_confidence_response(selected_model, analysis_result)
            
            # Step 4: Load model
            model_name = selected_model['model_name']
            self._log_forecasting_execution_flow("MODEL_LOADING_START", {'model_name': model_name})
            
            if not self.inference_service.is_model_loaded(model_name):
                logger.info(f"📥 Loading model: {model_name}")
                loaded = self.inference_service.load_model_from_registry(
                    model_name, 
                    self.knowledge_base
                )
                self._log_forecasting_execution_flow("MODEL_LOADING_RESULT", {
                    'loaded': loaded,
                    'loaded_models': self.inference_service.list_loaded_models()
                })
                if not loaded:
                    return self._create_model_loading_failed_response(model_name, analysis_result)
            else:
                self._log_forecasting_execution_flow("MODEL_ALREADY_LOADED")
            
            # Step 5: Prepare data
            self._log_forecasting_execution_flow("DATA_PREPARATION_START")
            user_data = sanitized_dataset.get('data')
            if user_data is None:
                logger.error("❌ CRITICAL: No data in sanitized_dataset")
                return self._create_error_response("No data provided for forecasting")
            
            self._log_forecasting_execution_flow("DATA_BEFORE_PREPROCESSING", {
                'data_type': type(user_data).__name__,
                'data_shape': getattr(user_data, 'shape', 'No shape attribute'),
                'data_columns': getattr(user_data, 'columns', 'No columns attribute')[:5] if hasattr(user_data, 'columns') else 'No columns'
            })
            
            processed_data = self.data_preprocessor.prepare_data_for_model(
                user_data=user_data,
                model_name=model_name,
                forecast_horizon=forecast_horizon
            )
            
            self._log_forecasting_execution_flow("DATA_AFTER_PREPROCESSING", {
                'processed_data_type': type(processed_data).__name__,
                'processed_shape': getattr(processed_data, 'shape', 'No shape'),
                'is_none': processed_data is None
            })
            
            if processed_data is None:
                return self._create_error_response(f"Failed to prepare data for {model_name}")
            
            logger.info(f"✅ Data prepared: {processed_data.shape}")
            
            # Step 6: Generate forecast - THIS IS THE CRITICAL PART!
            self._log_forecasting_execution_flow("FORECAST_GENERATION_START", {
                'model_name': model_name,
                'data_shape': processed_data.shape,
                'horizon': forecast_horizon
            })
            
            # THIS IS WHERE THE MAGIC HAPPENS - ONE LINE FOR ANY MODEL!
            forecast_result = self.inference_service.generate_forecast(
                model_name=model_name,
                data=processed_data,
                horizon=forecast_horizon
            )
            
            self._log_forecasting_execution_flow("FORECAST_GENERATION_RESULT", {
                'result_type': type(forecast_result).__name__,
                'is_none': forecast_result is None,
                'has_predictions': hasattr(forecast_result, 'predictions') if forecast_result else False,
                'prediction_count': len(forecast_result.predictions) if forecast_result and hasattr(forecast_result, 'predictions') else 0
            })
            
            if forecast_result is None:
                logger.error("❌ CRITICAL: generate_forecast returned None")
                return self._create_error_response("Forecast generation failed")
            
            # Step 7: Convert and store results
            forecast_dict = self._forecast_result_to_dict(forecast_result)
            self.current_forecast = forecast_dict
            
            self._log_forecasting_execution_flow("FORECAST_CONVERSION_COMPLETE", {
                'values_count': len(forecast_dict.get('values', [])),
                'has_confidence': 'confidence_intervals' in forecast_dict
            })
            
            # Step 8: LLM interpretation
            self._log_forecasting_execution_flow("INTERPRETATION_START")
            interpretation = self.llm_service.interpret_forecast(
                forecast_dict, 
                analysis_result,
                business_context
            )
            self.current_interpretation = interpretation
            self._log_forecasting_execution_flow("INTERPRETATION_COMPLETE")
            
            # Step 9: Compile final response
            result = self._create_success_response(
                analysis_result, 
                selected_model,
                forecast_dict, 
                interpretation
            )
            
            total_time = time.time() - start_time
            self._log_forecasting_execution_flow("PIPELINE_COMPLETE", {
                'total_time_seconds': total_time,
                'final_status': 'success'
            })
            
            logger.info(f"✅ Forecast completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ PIPELINE_ERROR: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"🔍 FULL_TRACEBACK:\n{error_trace}")
            return self._create_error_response(str(e))
    
    # Diagnostic Method for Testing "run_forecast_diagnostic"
    def run_forecast_diagnostic(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a quick diagnostic to identify where the forecasting breaks
        """
        diagnostic = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'steps': {},
            'issues': []
        }
        
        logger.info("🩺 RUNNING FORECAST DIAGNOSTIC")
        
        # Step 1: Check service initialization
        diagnostic['steps']['service_initialization'] = {
            'knowledge_base': self.knowledge_base is not None,
            'rule_engine': self.rule_engine is not None,
            'inference_service': self.inference_service is not None,
            'data_preprocessor': self.data_preprocessor is not None
        }
        
        # Step 2: Check model availability
        try:
            models = self.knowledge_base.get_all_models()
            diagnostic['steps']['model_availability'] = {
                'total_models': len(models),
                'active_models': len([m for m in models if m.get('is_active', True)]),
                'model_names': [m['model_name'] for m in models if m.get('is_active', True)]
            }
        except Exception as e:
            diagnostic['issues'].append(f"Model availability check failed: {e}")
        
        # Step 3: Check data preprocessing
        try:
            user_data = dataset_info.get('data')
            if user_data is not None:
                processed = self.data_preprocessor.prepare_data_for_model(
                    user_data=user_data,
                    model_name="test_model",  # Use any model name
                    forecast_horizon=30
                )
                diagnostic['steps']['data_preprocessing'] = {
                    'input_type': type(user_data).__name__,
                    'input_shape': getattr(user_data, 'shape', 'No shape'),
                    'output_type': type(processed).__name__ if processed else None,
                    'output_shape': getattr(processed, 'shape', 'No shape') if processed else None,
                    'success': processed is not None
                }
            else:
                diagnostic['issues'].append("No data in dataset_info")
        except Exception as e:
            diagnostic['issues'].append(f"Data preprocessing failed: {e}")
        
        # Step 4: Check inference service
        try:
            loaded_models = self.inference_service.list_loaded_models()
            diagnostic['steps']['inference_service'] = {
                'loaded_models': loaded_models,
                'loaded_count': len(loaded_models),
                'diagnostics': self.inference_service.get_diagnostics()
            }
        except Exception as e:
            diagnostic['issues'].append(f"Inference service check failed: {e}")
        
        logger.info(f"📋 DIAGNOSTIC_COMPLETE: {len(diagnostic['issues'])} issues found")
        return diagnostic

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models"""
        return {
            'loaded_models': self.inference_service.list_loaded_models(),
            'model_count': len(self.inference_service.list_loaded_models()),
            'supported_types': self.inference_service.executor_registry.list_supported_types()
        }
    
    def preload_models(self, model_names: List[str]):
        """Preload models for faster inference"""
        for model_name in model_names:
            if not self.inference_service.is_model_loaded(model_name):
                logger.info(f"📥 Preloading {model_name}")
                self.inference_service.load_model_from_registry(
                    model_name,
                    self.knowledge_base
                )
    
    def clear_cache(self):
        """Clear all caches"""
        self._analysis_cache.clear()
        logger.info("🧹 Cleared analysis cache")
    
    def close(self):
        """Close all services"""
        self.knowledge_base.close()
        # Unload all models
        for model_name in self.inference_service.list_loaded_models():
            self.inference_service.unload_model(model_name)
        logger.info("🔚 Supply Chain Service closed")


    
    # def analyze_dataset_with_knowledge_base(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Dataset analysis - BUSINESS LOGIC ONLY
    #     No model inference code here!
    #     """
    #     # Extract DataFrame before caching
    #     dataset_copy = dataset_info.copy()
    #     dataset_df = dataset_copy.pop('data', None)
        
    #     # Check cache
    #     if self.enable_caching:
    #         cache_key = self._generate_cache_key(dataset_info)
    #         if cache_key in self._analysis_cache:
    #             cache_entry = self._analysis_cache[cache_key]
    #             if time.time() - cache_entry['timestamp'] < self.cache_ttl_seconds:
    #                 logger.info("📊 Returning cached analysis")
    #                 return cache_entry['result']
        
    #     logger.info(f"🔍 Analyzing dataset: {dataset_info.get('name', 'Unknown')}")
        
    #     if dataset_df is not None:
    #         dataset_info['data'] = dataset_df
        
    #     try:
    #         # Business logic: validation
    #         validation_result = self.rule_engine.validate_dataset(dataset_info)
            
    #         # Business logic: get available models
    #         available_models = self.knowledge_base.get_all_models()
    #         active_models = [m for m in available_models if m.get('is_active', True)]
            
    #         if not active_models:
    #             return self._create_no_models_response(validation_result, dataset_info)
            
    #         # Business logic: model selection
    #         selected_model_result = self.rule_engine.rule_engine.select_model(dataset_info)
            
    #         if selected_model_result and selected_model_result.get('selected_model'):
    #             primary_model_name = selected_model_result['selected_model']
    #             primary_model = next((m for m in active_models 
    #                                 if m.get('model_name') == primary_model_name), None)
                
    #             if primary_model:
    #                 model_recommendations = self._create_recommendations_with_primary(
    #                     primary_model, active_models, selected_model_result
    #                 )
    #             else:
    #                 model_recommendations = self._create_fallback_recommendations(
    #                     active_models, dataset_info
    #                 )
    #         else:
    #             model_recommendations = self._create_fallback_recommendations(
    #                 active_models, dataset_info
    #             )
            
    #         # Business logic: validation and recommendations
    #         kb_validation = self._validate_with_knowledge_base(dataset_info)
    #         kb_recommendations = self._get_knowledge_base_recommendations(dataset_info)
            
    #         rule_analysis = {
    #             'validation': validation_result,
    #             'model_selection': self._format_model_selection(model_recommendations),
    #             'summary': self._create_rule_summary(validation_result, model_recommendations)
    #         }
            
    #         combined_analysis = {
    #             'rule_analysis': rule_analysis,
    #             'knowledge_base_validation': kb_validation,
    #             'knowledge_base_recommendations': kb_recommendations,
    #             'combined_summary': self._combine_analyses(rule_analysis, kb_validation, kb_recommendations)
    #         }
            
    #         self.current_analysis = combined_analysis
            
    #         # Cache result
    #         if self.enable_caching:
    #             cache_data = {k: v for k, v in combined_analysis.items() if k != 'data'}
    #             self._analysis_cache[cache_key] = {
    #                 'result': cache_data,
    #                 'timestamp': time.time()
    #             }
            
    #         return combined_analysis
            
    #     except Exception as e:
    #         logger.error(f"❌ Analysis failed: {e}")
    #         return self._create_fallback_analysis(dataset_info)
    
    # def validate_forecasting_request(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """Validate and sanitize forecasting request inputs"""
    #     validation_errors = []
    #     warnings = []
        
    #     required_fields = ['name', 'columns', 'row_count']
    #     for field in required_fields:
    #         if field not in dataset_info:
    #             validation_errors.append(f"Missing required field: {field}")
        
    #     if not isinstance(dataset_info.get('columns', []), list):
    #         validation_errors.append("Columns must be a list")
        
    #     if not isinstance(dataset_info.get('row_count', 0), int):
    #         validation_errors.append("Row count must be an integer")
        
    #     row_count = dataset_info.get('row_count', 0)
    #     if row_count < 1:
    #         validation_errors.append("Row count must be positive")
    #     elif row_count < 12:
    #         warnings.append("Limited data points may affect forecast accuracy")
        
    #     columns = dataset_info.get('columns', [])
    #     if not columns:
    #         validation_errors.append("Dataset must have at least one column")
    #     elif len(columns) < 2:
    #         warnings.append("Dataset has very few columns")
        
    #     sanitized_info = dataset_info.copy()
    #     sanitized_info['name'] = str(sanitized_info.get('name', '')).strip()[:100]
    #     sanitized_info['columns'] = [str(col).strip() for col in sanitized_info.get('columns', [])]
    #     sanitized_info['missing_percentage'] = max(0.0, min(1.0, 
    #         float(sanitized_info.get('missing_percentage', 0.0))))
    #     sanitized_info['row_count'] = max(1, int(sanitized_info.get('row_count', 0)))
        
    #     return {
    #         'valid': len(validation_errors) == 0,
    #         'errors': validation_errors,
    #         'warnings': warnings,
    #         'sanitized_data': sanitized_info
    #     }

    # def validate_forecasting_request(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """Validate and sanitize forecasting request inputs"""
    #     validation_errors = []
    #     warnings = []
        
    #     # Required fields validation
    #     required_fields = ['name', 'columns', 'row_count']
    #     for field in required_fields:
    #         if field not in dataset_info:
    #             validation_errors.append(f"Missing required field: {field}")
        
    #     # Data type validation
    #     if not isinstance(dataset_info.get('columns', []), list):
    #         validation_errors.append("Columns must be a list")
        
    #     if not isinstance(dataset_info.get('row_count', 0), int):
    #         validation_errors.append("Row count must be an integer")
        
    #     # Business logic validation
    #     row_count = dataset_info.get('row_count', 0)
    #     if row_count < 1:
    #         validation_errors.append("Row count must be positive")
    #     elif row_count < 12:
    #         warnings.append("Limited data points may affect forecast accuracy")
        
    #     columns = dataset_info.get('columns', [])
    #     if not columns:
    #         validation_errors.append("Dataset must have at least one column")
    #     elif len(columns) < 2:
    #         warnings.append("Dataset has very few columns, consider adding more features")
        
    #     # Frequency validation
    #     valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'none']
    #     frequency = dataset_info.get('frequency', 'none')
    #     if frequency not in valid_frequencies:
    #         warnings.append(f"Uncommon frequency: {frequency}. Expected: {valid_frequencies}")
        
    #     # Sanitize inputs
    #     sanitized_info = dataset_info.copy()
    #     sanitized_info['name'] = str(sanitized_info.get('name', '')).strip()[:100]
    #     sanitized_info['columns'] = [str(col).strip() for col in sanitized_info.get('columns', [])]
        
    #     # Ensure numeric fields are within reasonable bounds
    #     sanitized_info['missing_percentage'] = max(0.0, min(1.0, float(sanitized_info.get('missing_percentage', 0.0))))
    #     sanitized_info['row_count'] = max(1, int(sanitized_info.get('row_count', 0)))
        
    #     return {
    #         'valid': len(validation_errors) == 0,
    #         'errors': validation_errors,
    #         'warnings': warnings,
    #         'sanitized_data': sanitized_info
    #     }
    
    # def analyze_dataset_with_knowledge_base(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     REAL dataset analysis using actual dataset with proper fallback handling
    #     """
    #     # Extract DataFrame BEFORE caching
    #     dataset_copy = dataset_info.copy()
    #     dataset_df = dataset_copy.pop('data', None) 
        
    #     # Check cache if enabled
    #     if self.enable_caching:
    #         cache_key = self.knowledge_base._generate_cache_key(dataset_info)
    #         if cache_key in self._analysis_cache:
    #             cache_entry = self._analysis_cache[cache_key]
    #             if time.time() - cache_entry['timestamp'] < self.cache_ttl_seconds:
    #                 logger.info("📊 Returning cached analysis")
    #                 return cache_entry['result']
    #             else:
    #                 del self._analysis_cache[cache_key]
        
    #     logger.info(f"🔍 Analyzing dataset: {dataset_info.get('name', 'Unknown')}")
        
    #     # Restore DataFrame for processing
    #     if dataset_df is not None:
    #         dataset_info['data'] = dataset_df
        
    #     try:
    #         # Step 1: Rule-based validation using REAL RuleEngineService
    #         validation_result = self.rule_engine.validate_dataset(dataset_info)
            
    #         # Step 2: Get ALL available models (no filtering here)
    #         available_models = self.knowledge_base.get_all_models()
    #         active_models = [m for m in available_models if m.get('is_active', True)]
            
    #         if not active_models:
    #             # No models available - return early with clear error
    #             return self._create_no_models_response(validation_result, dataset_info)
            
    #         # Step 3: Use RuleEngine to select the BEST model for this actual dataset
    #         selected_model_result = self.rule_engine.rule_engine.select_model(dataset_info)
            
    #         if selected_model_result and selected_model_result.get('selected_model'):
    #             # RuleEngine found a model! Use it as primary
    #             primary_model_name = selected_model_result['selected_model']
    #             primary_model = next((m for m in active_models if m.get('model_name') == primary_model_name), None)
                
    #             if primary_model:
    #                 # Found the selected model in our available models
    #                 model_recommendations = self._create_recommendations_with_primary(
    #                     primary_model, active_models, selected_model_result
    #                 )
    #             else:
    #                 # Selected model not in available models - use fallback
    #                 model_recommendations = self._create_fallback_recommendations(active_models, dataset_info)
    #         else:
    #             # RuleEngine didn't select any model - use fallback
    #             model_recommendations = self._create_fallback_recommendations(active_models, dataset_info)
            
    #         # Step 4: Knowledge base validation
    #         kb_validation = self._validate_with_knowledge_base(dataset_info)
            
    #         # Step 5: Model recommendations from knowledge base
    #         kb_recommendations = self._get_knowledge_base_recommendations(dataset_info)
            
    #         # Format rule analysis
    #         rule_analysis = {
    #             'validation': validation_result,
    #             'model_selection': self._format_model_selection(model_recommendations),
    #             'summary': self._create_rule_summary(validation_result, model_recommendations)
    #         }
            
    #         # Combine results
    #         combined_analysis = {
    #             'rule_analysis': rule_analysis,
    #             'knowledge_base_validation': kb_validation,
    #             'knowledge_base_recommendations': kb_recommendations,
    #             'combined_summary': self._combine_analyses(rule_analysis, kb_validation, kb_recommendations)
    #         }
            
    #         self.current_analysis = combined_analysis
            
    #         # Cache the result if enabled
    #         if self.enable_caching:
    #             cache_data = {k: v for k, v in combined_analysis.items() if k != 'data'}
    #             self._analysis_cache[cache_key] = {
    #                 'result': cache_data,
    #                 'timestamp': time.time()
    #             }
            
    #         return combined_analysis
            
    #     except Exception as e:
    #         logger.error(f"❌ Analysis failed: {e}")
    #         # Fallback to basic analysis
    #         return self._create_fallback_analysis(dataset_info)
    
    # def validate_forecasting_request(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """Validate and sanitize forecasting request inputs"""
    #     validation_errors = []
    #     warnings = []
        
    #     required_fields = ['name', 'columns', 'row_count']
    #     for field in required_fields:
    #         if field not in dataset_info:
    #             validation_errors.append(f"Missing required field: {field}")
        
    #     if not isinstance(dataset_info.get('columns', []), list):
    #         validation_errors.append("Columns must be a list")
        
    #     if not isinstance(dataset_info.get('row_count', 0), int):
    #         validation_errors.append("Row count must be an integer")
        
    #     row_count = dataset_info.get('row_count', 0)
    #     if row_count < 1:
    #         validation_errors.append("Row count must be positive")
    #     elif row_count < 12:
    #         warnings.append("Limited data points may affect forecast accuracy")
        
    #     columns = dataset_info.get('columns', [])
    #     if not columns:
    #         validation_errors.append("Dataset must have at least one column")
    #     elif len(columns) < 2:
    #         warnings.append("Dataset has very few columns")
        
    #     sanitized_info = dataset_info.copy()
    #     sanitized_info['name'] = str(sanitized_info.get('name', '')).strip()[:100]
    #     sanitized_info['columns'] = [str(col).strip() for col in sanitized_info.get('columns', [])]
    #     sanitized_info['missing_percentage'] = max(0.0, min(1.0, 
    #         float(sanitized_info.get('missing_percentage', 0.0))))
    #     sanitized_info['row_count'] = max(1, int(sanitized_info.get('row_count', 0)))
        
    #     return {
    #         'valid': len(validation_errors) == 0,
    #         'errors': validation_errors,
    #         'warnings': warnings,
    #         'sanitized_data': sanitized_info
    #     }
    
    # def analyze_dataset_with_knowledge_base(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Dataset analysis - BUSINESS LOGIC ONLY
    #     No model inference code here!
    #     """
    #     # Extract DataFrame before caching
    #     dataset_copy = dataset_info.copy()
    #     dataset_df = dataset_copy.pop('data', None)
        
    #     # Check cache
    #     if self.enable_caching:
    #         cache_key = self.knowledge_base._generate_cache_key(dataset_info)
    #         if cache_key in self._analysis_cache:
    #             cache_entry = self._analysis_cache[cache_key]
    #             if time.time() - cache_entry['timestamp'] < self.cache_ttl_seconds:
    #                 logger.info("📊 Returning cached analysis")
    #                 return cache_entry['result']
        
    #     logger.info(f"🔍 Analyzing dataset: {dataset_info.get('name', 'Unknown')}")
        
    #     if dataset_df is not None:
    #         dataset_info['data'] = dataset_df
        
    #     try:
    #         # Business logic: validation
    #         validation_result = self.rule_engine.validate_dataset(dataset_info)
            
    #         # Business logic: get available models
    #         available_models = self.knowledge_base.get_all_models()
    #         active_models = [m for m in available_models if m.get('is_active', True)]
            
    #         if not active_models:
    #             return self._create_no_models_response(validation_result, dataset_info)
            
    #         # Business logic: model selection
    #         selected_model_result = self.rule_engine.rule_engine.select_model(dataset_info)
            
    #         if selected_model_result and selected_model_result.get('selected_model'):
    #             primary_model_name = selected_model_result['selected_model']
    #             primary_model = next((m for m in active_models 
    #                                 if m.get('model_name') == primary_model_name), None)
                
    #             if primary_model:
    #                 model_recommendations = self._create_recommendations_with_primary(
    #                     primary_model, active_models, selected_model_result
    #                 )
    #             else:
    #                 model_recommendations = self._create_fallback_recommendations(
    #                     active_models, dataset_info
    #                 )
    #         else:
    #             model_recommendations = self._create_fallback_recommendations(
    #                 active_models, dataset_info
    #             )
            
    #         # Business logic: validation and recommendations
    #         kb_validation = self._validate_with_knowledge_base(dataset_info)
    #         kb_recommendations = self._get_knowledge_base_recommendations(dataset_info)
            
    #         rule_analysis = {
    #             'validation': validation_result,
    #             'model_selection': self._format_model_selection(model_recommendations),
    #             'summary': self._create_rule_summary(validation_result, model_recommendations)
    #         }
            
    #         combined_analysis = {
    #             'rule_analysis': rule_analysis,
    #             'knowledge_base_validation': kb_validation,
    #             'knowledge_base_recommendations': kb_recommendations,
    #             'combined_summary': self._combine_analyses(rule_analysis, kb_validation, kb_recommendations)
    #         }
            
    #         self.current_analysis = combined_analysis
            
    #         # Cache result
    #         if self.enable_caching:
    #             cache_data = {k: v for k, v in combined_analysis.items() if k != 'data'}
    #             self._analysis_cache[cache_key] = {
    #                 'result': cache_data,
    #                 'timestamp': time.time()
    #             }
            
    #         return combined_analysis
            
    #     except Exception as e:
    #         logger.error(f"❌ Analysis failed: {e}")
    #         return self._create_fallback_analysis(dataset_info)
    
    #     # ---------------------------
