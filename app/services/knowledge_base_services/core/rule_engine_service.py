# app/services/knowledge_base_services/core/rule_engine_service.py
# Enhanced Rule Engine Service with Caching, Performance Metrics, and Export Capabilities
import sys
import os
import time
import json
import hashlib
from typing import Dict, List, Any, Optional
import logging

# In rule_engine_service.py
from app.services.shared_utils import (
    parse_features,
    safe_json_load,
    sanitize_dataset_info,
    calculate_model_compatibility_score,
    generate_compatibility_reasons,
    generate_cache_key
)


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RuleEngine with proper error handling
ActualRuleEngine = None
USE_REAL_ENGINE = False

try:
    # Try to import the actual RuleEngine class (not RuleEngineService)
    from app.knowledge_base.rule_layer.rule_engine import RuleEngine as ActualRuleEngine
    USE_REAL_ENGINE = True
    logger.info("✅ Using REAL RuleEngine with YAML rules")
except ImportError as e:
    logger.warning(f"⚠️ Could not import RuleEngine directly: {e}")
    try:
        # Fallback: try importing from different path
        from knowledge_base.rule_layer.rule_engine import RuleEngine as ActualRuleEngine
        USE_REAL_ENGINE = True
        logger.info("✅ Using REAL RuleEngine (alternative path)")
    except ImportError as e2:
        logger.error(f"❌ Could not import real RuleEngine from any path: {e2}")
        USE_REAL_ENGINE = False


class RuleEngineService:
    """
    Enhanced service wrapper for the REAL Rule Engine with caching, performance metrics, and export capabilities
    """
    
    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        self._model_cache = None
        self._analysis_cache = {}
        # self._compatibility_cache = {}
        
        

        if USE_REAL_ENGINE and ActualRuleEngine is not None:
            try:
                self.rule_engine = ActualRuleEngine(db_connection)
                logger.info("✅ Rule Engine Service initialized with REAL rule processing")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RuleEngine: {e}")
                self.rule_engine = MinimalRuleEngine()
                logger.info("⚠️ Using minimal rule engine fallback")
        else:
            # Use minimal fallback
            self.rule_engine = MinimalRuleEngine()
            logger.info("⚠️ Using minimal rule engine fallback")


    # use shared_utils.py
    # def _generate_dataset_hash(self, dataset_info: Dict[str, Any]) -> str:
    #     """Generate unique hash for dataset to enable caching"""
    #     # Remove any DataFrame objects before hashing to avoid serialization issues
    #     sanitized_info = dataset_info.copy()
    #     if 'data' in sanitized_info:
    #         del sanitized_info['data']
    #     dataset_str = json.dumps(sanitized_info, sort_keys=True, default=str)
    #     return hashlib.md5(dataset_str.encode()).hexdigest()


    def analyze_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple dataset analysis - validates data and suggests models
        Returns basic information needed for the forecasting workflow
        """
        logger.info(f"Analyzing dataset: {dataset_info.get('name', 'Unknown')}")
        
        try:
            # Step 1: Basic validation using rule engine
            validation_result = self.rule_engine.validate_dataset(dataset_info)
            
            # Step 2: Get available models
            available_models = self._get_available_models()

            # Step 3: Get model recommendations if validation passed
            if validation_result['valid']:
                # This returns a LIST of recommendations
                recommendations_list = self.get_intelligent_model_recommendations(
                    dataset_info=dataset_info,
                    validation_result=validation_result,      
                    available_models=available_models
                )
                
                # Convert the LIST to the expected DICTIONARY structure
                if recommendations_list:
                    top_recommendation = recommendations_list[0]  # Get the highest scored model
                    model_recommendations = {
                        'ranked_models': recommendations_list,
                        'top_recommendation': top_recommendation['model_name'],
                        'selection_confidence': top_recommendation.get('confidence', 0.0)
                    }
                else:
                    model_recommendations = {
                        'ranked_models': [],
                        'top_recommendation': None,
                        'selection_confidence': 0.0
                    }
            else:
                model_recommendations = {
                    'ranked_models': [],
                    'top_recommendation': None,
                    'selection_confidence': 0.0
                }
            
            # Step 4: Determine if we can proceed
            can_proceed = (
                validation_result['valid'] and 
                model_recommendations.get('top_recommendation') is not None
            )
            
            # Step 5: Create the summary that the demo expects
            if can_proceed:
                summary_message = f"✅ Ready for forecasting with {model_recommendations['top_recommendation']}"
            else:
                summary_message = "❌ Cannot proceed - check validation results"
            
            # ✅ FIX: Return the complete structure with 'summary' key
            return {
                'validation': {
                    'valid': validation_result['valid'],
                    'errors': validation_result.get('errors', []),
                    'warnings': validation_result.get('warnings', [])
                },
                'model_recommendations': {
                    'top_recommendation': model_recommendations.get('top_recommendation'),
                    'ranked_models': model_recommendations.get('ranked_models', []),
                    'total_models_found': len(model_recommendations.get('ranked_models', [])),
                    'selection_confidence': model_recommendations.get('selection_confidence', 0.0)
                },
                'can_proceed': can_proceed,
                'status': 'success',
                # ✅ CRITICAL: Add the missing 'summary' key that the demo expects
                'summary': {
                    'status_message': summary_message,
                    'can_proceed': can_proceed,
                    'confidence': model_recommendations.get('selection_confidence', 0.0),
                    'selected_model': model_recommendations.get('top_recommendation'),
                    'validation_passed': validation_result['valid']
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'can_proceed': False,
                'validation': {'valid': False, 'errors': [f'Analysis error: {str(e)}'], 'warnings': []},
                'model_recommendations': {'top_recommendation': None, 'ranked_models': [], 'total_models_found': 0, 'selection_confidence': 0.0},
                # ✅ CRITICAL: Even in error, include the 'summary' key
                'summary': {
                    'status_message': f'Analysis failed: {str(e)}',
                    'can_proceed': False,
                    'confidence': 0.0,
                    'selected_model': None,
                    'validation_passed': False
                }
            }
    # use shared_utils.py
    # def _sanitize_dataset_info(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """Remove any non-serializable objects from dataset info"""
    #     sanitized = dataset_info.copy()
        
    #     # Remove DataFrame objects and other non-serializable items
    #     keys_to_remove = []
    #     for key, value in sanitized.items():
    #         if hasattr(value, '__class__') and 'DataFrame' in str(value.__class__):
    #             keys_to_remove.append(key)
    #         elif hasattr(value, 'dtypes'):  # Likely a DataFrame
    #             keys_to_remove.append(key)
        
    #     for key in keys_to_remove:
    #         if key in sanitized:
    #             logger.debug(f"Removing non-serializable key: {key}")
    #             del sanitized[key]
        
    #     return sanitized


    def _analyze_model_selection(self, dataset_info: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze why models are or aren't being selected with detailed reasons
        """
        if not validation_result.get('valid', False):
            return {
                'result': {
                    'selected_model': None,
                    'confidence': 0.0,
                    'reason': 'Dataset validation failed',
                    'model_type': None
                },
                'analysis': {
                    'can_proceed': False,
                    'rejection_reason': 'DATA_VALIDATION_FAILED',
                    'failed_checks': validation_result.get('errors', []),
                    'compatible_models': [],
                    'missing_requirements': validation_result.get('errors', [])
                }
            }
        
        # Try to select a model
        try:
            selection_result = self.rule_engine.select_model(dataset_info)
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            selection_result = {
                'selected_model': None,
                'confidence': 0.0,
                'reason': f'Model selection error: {str(e)}'
            }
        
        if selection_result.get('selected_model'):
            # Model was selected successfully
            return {
                'result': selection_result,
                'analysis': {
                    'can_proceed': True,
                    'rejection_reason': None,
                    'failed_checks': [],
                    'compatible_models': [{
                        'model_name': selection_result['selected_model'],
                        'match_reason': selection_result.get('reason', 'Rule matched'),
                        'confidence': selection_result.get('confidence', 0.0),
                        'status': 'selected'
                    }],
                    'missing_requirements': []
                }
            }
        else:
            # No model selected - analyze why
            compatibility_analysis = self._analyze_model_compatibility(dataset_info)
            
            return {
                'result': selection_result,
                'analysis': {
                    'can_proceed': False,
                    'rejection_reason': 'NO_COMPATIBLE_MODELS',
                    'failed_checks': [],
                    'compatible_models': compatibility_analysis['compatible_models'],
                    'missing_requirements': compatibility_analysis['missing_requirements']
                }
            }

    # Add this to RuleEngineService class in rule_engine_service.py

    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models from knowledge base"""
        try:
            # Get all models from knowledge base
            from app.repositories.model_repository import get_model_repository
            repository = get_model_repository()
            return repository.get_all_active_models()
            # all_models = kb_service.get_all_models()
            # kb_service.close()
            
            # if not all_models:
            #     logger.warning("No models found in knowledge base")
            #     return []
            
            # # Filter to active models only
            # active_models = [model for model in all_models if model.get('is_active', True)]
            
            # if not active_models:
            #     logger.warning("No active models found")
            #     return []
            
            # logger.info(f"📊 Found {len(active_models)} active models for evaluation")
            # return active_models
            
        except Exception as e:
            logger.error(f"❌ Error getting available models: {e}")
            return []  # Empty list - let calling function handle fallback
        

    # def _analyze_model_compatibility(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Enhanced compatibility analysis with scoring and caching
    #     """
    #     dataset_hash = self._generate_dataset_hash(dataset_info)
        
    #     # Check cache first
    #     if dataset_hash in self._compatibility_cache:
    #         logger.debug("Returning cached compatibility analysis")
    #         return self._compatibility_cache[dataset_hash]
        
    #     try:
    #         from app.services.knowledge_base_services.core.knowledge_base_service import SupplyChainService as KnowledgeBaseService
    #         kb_service = KnowledgeBaseService("supply_chain.db")
    #         all_models = kb_service.get_all_models()
    #         kb_service.close()
    #     except Exception as e:
    #         logger.error(f"❌ Could not load models from knowledge base: {e}")
    #         # Fallback if KB service is not available
    #         all_models = []
        
    #     compatible_models = []
    #     missing_requirements = []
    #     dataset_columns = set(dataset_info.get('columns', []))
        
    #     for model in all_models:
    #         model_name = model.get('model_name', 'Unknown')
    #         required_features = self._parse_required_features(model.get('required_features', []))
    #         optional_features = self._parse_required_features(model.get('optional_features', []))
    #         target_variable = model.get('target_variable', 'unknown')
            
    #         # ENHANCED: Calculate compatibility score (0-100)
    #         required_match = sum(1 for f in required_features if f in dataset_columns)
    #         optional_match = sum(1 for f in optional_features if f in dataset_columns)
            
    #         # Calculate compatibility score with weights
    #         required_score = (required_match / max(len(required_features), 1)) * 70  # 70% weight
    #         optional_score = (optional_match / max(len(optional_features), 1)) * 30  # 30% weight
    #         compatibility_score = required_score + optional_score
            
    #         # Check target variable - use flexible matching
    #         has_target = (
    #             target_variable in dataset_columns or 
    #             any(target in dataset_columns for target in ['sales', 'demand', 'value', 'target', 'quantity', 'revenue', 'volume'])
    #         )
            
    #         # Adjust score based on target availability
    #         if not has_target:
    #             compatibility_score *= 0.5  # Halve score if target missing
            
    #         if compatibility_score >= 90 and has_target:
    #             # Model is fully compatible
    #             compatible_models.append({
    #                 'model_name': model_name,
    #                 'model_type': model.get('model_type', 'unknown'),
    #                 'compatibility_score': compatibility_score,
    #                 'status': 'fully_compatible',
    #                 'match_reason': "All required features and target available",
    #                 'matching_features': [f for f in required_features if f in dataset_columns],
    #                 'target_variable': target_variable,
    #                 'performance_metrics': self._safe_json_load(model.get('performance_metrics', '{}'))
    #             })
    #         elif compatibility_score >= 60 and has_target:
    #             # Model is partially compatible
    #             compatible_models.append({
    #                 'model_name': model_name,
    #                 'model_type': model.get('model_type', 'unknown'),
    #                 'compatibility_score': compatibility_score,
    #                 'status': 'partially_compatible',
    #                 'match_reason': f"Partially compatible ({compatibility_score:.1f}%)",
    #                 'matching_features': [f for f in required_features if f in dataset_columns],
    #                 'missing_required': [f for f in required_features if f not in dataset_columns],
    #                 'target_variable': target_variable,
    #                 'performance_metrics': self._safe_json_load(model.get('performance_metrics', '{}'))
    #             })
    #         else:
    #             # Model is incompatible - record why
    #             requirement_issues = []
    #             missing_req_features = [f for f in required_features if f not in dataset_columns]
                
    #             if missing_req_features:
    #                 if len(missing_req_features) > 3:
    #                     shown_features = missing_req_features[:3]
    #                     requirement_issues.append(f"Missing features: {', '.join(shown_features)} + {len(missing_req_features) - 3} more")
    #                 else:
    #                     requirement_issues.append(f"Missing features: {', '.join(missing_req_features)}")
                
    #             if not has_target:
    #                 requirement_issues.append(f"Missing target variable: {target_variable}")
                
    #             missing_requirements.append({
    #                 'model_name': model_name,
    #                 'model_type': model.get('model_type', 'unknown'),
    #                 'compatibility_score': compatibility_score,
    #                 'issues': requirement_issues,
    #                 'required_features': required_features,
    #                 'target_variable': target_variable,
    #                 'missing_required_count': len(missing_req_features),
    #                 'has_target': has_target
    #             })
        
    #     # Sort compatible models by score (descending)
    #     compatible_models.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
    #     result = {
    #         'compatible_models': compatible_models,
    #         'missing_requirements': missing_requirements,
    #         'total_models_analyzed': len(all_models),
    #         'compatibility_summary': {
    #             'fully_compatible': len([m for m in compatible_models if m['status'] == 'fully_compatible']),
    #             'partially_compatible': len([m for m in compatible_models if m['status'] == 'partially_compatible']),
    #             'incompatible': len(missing_requirements),
    #             'average_compatibility_score': sum(m['compatibility_score'] for m in compatible_models) / max(len(compatible_models), 1) if compatible_models else 0
    #         }
    #     }
        
    #     # Cache the result
    #     self._compatibility_cache[dataset_hash] = result
    #     return result

    def get_intelligent_model_recommendations(
        self,
        dataset_info: Dict[str, Any],
        validation_result: Dict[str, Any],
        available_models: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get intelligent model recommendations based on dataset characteristics
        
        Args:
            dataset_info: Dataset information
            validation_result: Validation results from rule engine
            available_models: List of available models from KB
        
        Returns:
            List of recommended models with scores
        """
        try:
            recommendations = []
            
            # Extract dataset characteristics
            row_count = dataset_info.get('row_count', 0)
            frequency = dataset_info.get('frequency', 'unknown')
            columns = dataset_info.get('columns', [])
            
            # Score each model
            for model in available_models:
                score = 0
                reasons = []
                
                model_name = model.get('model_name', 'Unknown')
                
                # Rule 1: Prophet for time series with seasonality
                if model_name == 'Prophet':
                    if frequency in ['daily', 'weekly', 'monthly']:
                        score += 30
                        reasons.append("Good for seasonal patterns")
                    if row_count >= 50:
                        score += 20
                        reasons.append("Sufficient data for Prophet")
                
                # Rule 3: LightGBM for feature-rich datasets
                elif model_name == 'LightGBM':
                    if len(columns) > 3:
                        score += 30
                        reasons.append("Leverages multiple features")
                    if row_count >= 100:
                        score += 20
                        reasons.append("Sufficient data for tree models")
                
                # Rule 4: TFT for complex patterns
                elif 'TFT' in model_name or 'Temporal' in model_name:
                    if row_count >= 500:
                        score += 35
                        reasons.append("Sufficient data for deep learning")
                    if len(columns) > 5:
                        score += 15
                        reasons.append("Can handle complex feature sets")
                
                # General rules
                if validation_result.get('valid', False):
                    score += 10
                    reasons.append("Data passed validation")
                
                # Add to recommendations
                recommendations.append({
                    'model_name': model_name,
                    'model_type': model.get('model_type', 'Unknown'),
                    'score': score,
                    'confidence': min(score / 100, 0.95),
                    'reasons': reasons,
                    'status': 'recommended' if score >= 40 else 'compatible' if score >= 20 else 'not_recommended'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Generated {len(recommendations)} model recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def validate_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dataset using the underlying rule engine
        This method was MISSING causing the error
        """
        try:
            # Delegate to the actual rule engine
            return self.rule_engine.validate_dataset(dataset_info)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}'],
                'warnings': []
            }

    def _analyze_model_compatibility(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple model compatibility check
        Returns which models work with the dataset
        """
        try:
            # Get models from repository (not separate service)
            all_models = self.repository.get_active_models()
            dataset_columns = set(dataset_info.get('columns', []))
            
            compatible_models = []
            
            for model in all_models:
                model_name = model['model_name']
                required_features = json.loads(model['required_features'])
                target_variable = model['target_variable']
                
                # Simple check: are all required features present?
                missing_features = [f for f in required_features if f not in dataset_columns]
                has_target = target_variable in dataset_columns
                
                if not missing_features and has_target:
                    # Model is compatible
                    compatible_models.append({
                        'model_id': model['model_id'],
                        'model_name': model_name,
                        'model_type': model['model_type'],
                        'target_variable': target_variable,
                        'performance_metrics': json.loads(model.get('performance_metrics', '{}'))
                    })
            
            # Sort by performance (MAPE lower is better)
            compatible_models.sort(
                key=lambda x: x['performance_metrics'].get('MAPE', float('inf'))
            )
            
            return {
                'compatible_models': compatible_models,
                'total_compatible': len(compatible_models),
                'total_checked': len(all_models)
            }
            
        except Exception as e:
            logger.error(f"Compatibility analysis failed: {e}")
            return {
                'compatible_models': [],
                'total_compatible': 0,
                'total_checked': 0,
                'error': str(e)
            }
    
    # use shared_utils.py 
    # def _safe_json_load(self, json_str: str) -> Dict[str, Any]:
    #     """Safely load JSON string with error handling"""
    #     try:
    #         if isinstance(json_str, str):
    #             return json.loads(json_str)
    #         elif isinstance(json_str, dict):
    #             return json_str
    #         else:
    #             return {}
    #     except:
    #         return {}


    # use shared_utils.py 
    # def _parse_required_features(self, features_data) -> List[str]:
    #     """Parse required features from various formats"""
    #     if isinstance(features_data, str):
    #         try:
    #             # Handle string representation of list
    #             if features_data.startswith('[') and features_data.endswith(']'):
    #                 return eval(features_data)
    #             else:
    #                 return [features_data]
    #         except:
    #             return []
    #     elif isinstance(features_data, list):
    #         return features_data
    #     else:
    #         return []


    def _generate_comprehensive_recommendations(self, validation_result: Dict[str, Any],
                                            selection_analysis: Dict[str, Any],
                                            dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive, actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        if validation_result.get('errors'):
            for error in validation_result['errors']:
                recommendations.append({
                    'type': 'critical',
                    'priority': 'high',
                    'category': 'data_quality',
                    'message': error,
                    'action': 'Must be fixed before forecasting',
                    'impact': 'Prevents forecasting entirely'
                })
        
        if validation_result.get('warnings'):
            for warning in validation_result['warnings']:
                recommendations.append({
                    'type': 'warning',
                    'priority': 'medium',
                    'category': 'data_quality',
                    'message': warning,
                    'action': 'Consider addressing for better results',
                    'impact': 'May affect forecast accuracy'
                })
        
        # Model compatibility recommendations
        analysis = selection_analysis.get('analysis', {})
        if not analysis.get('can_proceed') and analysis.get('missing_requirements'):
            for req in analysis['missing_requirements']:
                if isinstance(req, dict) and 'issues' in req:
                    for issue in req.get('issues', []):
                        recommendations.append({
                            'type': 'requirement',
                            'priority': 'high' if req.get('compatibility_score', 0) < 50 else 'medium',
                            'category': 'model_compatibility',
                            'message': f"{req.get('model_name', 'Unknown model')}: {issue}",
                            'action': f"Add missing features or choose different model",
                            'impact': f"Cannot use {req.get('model_name', 'this model')} for forecasting",
                            'compatibility_score': req.get('compatibility_score', 0)
                        })
        
        # Data collection recommendations
        row_count = dataset_info.get('row_count', 0)
        if row_count < 50:
            recommendations.append({
                'type': 'suggestion',
                'priority': 'medium',
                'category': 'data_volume',
                'message': f'Limited data ({row_count} points) may affect forecast accuracy',
                'action': 'Collect more historical data',
                'impact': 'Improved model performance and reliability'
            })
        
        # Feature engineering recommendations
        columns = dataset_info.get('columns', [])
        if 'date' not in columns and dataset_info.get('frequency') != 'none':
            recommendations.append({
                'type': 'suggestion',
                'priority': 'medium',
                'category': 'feature_engineering',
                'message': 'Time series data detected but no date column',
                'action': 'Add a date column for time series forecasting',
                'impact': 'Enable time series models like Prophet'
            })
        
        return recommendations

    def _generate_detailed_summary(self, validation_result: Dict[str, Any],
                                 selection_analysis: Dict[str, Any],
                                 recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        analysis = selection_analysis.get('analysis', {})
        can_proceed = analysis.get('can_proceed', False)
        
        # Count recommendation priorities and categories
        high_priority = len([r for r in recommendations if r.get('priority') == 'high'])
        medium_priority = len([r for r in recommendations if r.get('priority') == 'medium'])
        
        categories = {}
        for rec in recommendations:
            cat = rec.get('category', 'other')
            categories[cat] = categories.get(cat, 0) + 1
        
        # Generate detailed status message
        if can_proceed:
            status_message = f"✅ Ready for forecasting with {selection_analysis['result'].get('selected_model', 'selected model')}"
            next_steps = [
                f"Proceed with {selection_analysis['result'].get('selected_model', 'selected model')}",
                "Review model configuration",
                "Generate forecast"
            ]
        else:
            rejection_reason = analysis.get('rejection_reason', 'UNKNOWN')
            if rejection_reason == 'DATA_VALIDATION_FAILED':
                status_message = "❌ Cannot proceed: Dataset validation failed"
                next_steps = [
                    "Address validation errors above",
                    "Re-upload corrected dataset",
                    "Re-analyze after fixes"
                ]
            elif rejection_reason == 'NO_COMPATIBLE_MODELS':
                compatible_count = len(analysis.get('compatible_models', []))
                status_message = f"❌ Cannot proceed: No compatible models found ({compatible_count} models analyzed)"
                next_steps = [
                    "Add required features based on recommendations",
                    "Consider different dataset structure",
                    "Contact support for model customization"
                ]
            else:
                status_message = "❌ Cannot proceed: Unknown issue"
                next_steps = ["Contact system administrator"]
        
        return {
            'can_proceed': can_proceed,
            'status_message': status_message,
            'confidence': selection_analysis['result'].get('confidence', 0.0),
            'rejection_reason': analysis.get('rejection_reason'),
            'compatible_models_count': len(analysis.get('compatible_models', [])),
            'incompatible_models_count': len(analysis.get('missing_requirements', [])),
            'recommendation_summary': {
                'total': len(recommendations),
                'high_priority': high_priority,
                'medium_priority': medium_priority,
                'categories': categories
            },
            'next_steps': next_steps
        }

    def export_analysis_report(self, analysis_result: Dict[str, Any], format: str = 'markdown') -> str:
        """
        Export analysis as formatted report
        """
        if format == 'markdown':
            return self._generate_markdown_report(analysis_result)
        elif format == 'json':
            return json.dumps(analysis_result, indent=2, ensure_ascii=False, default=str)
        elif format == 'html':
            return self._generate_html_report(analysis_result)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report"""
        summary = analysis_result.get('summary', {})
        performance = analysis_result.get('performance_metrics', {})
        
        report = [
            "# Supply Chain Forecasting Analysis Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Analysis Time:** {performance.get('analysis_time_seconds', 0):.3f}s",
            "",
            f"## 📊 Executive Summary",
            f"**Status:** {summary.get('status_message', 'Unknown')}",
            f"**Confidence:** {summary.get('confidence', 0):.1%}",
            f"**Can Proceed:** {'✅ Yes' if summary.get('can_proceed') else '❌ No'}",
            "",
            f"## 🔍 Model Compatibility",
            f"**Compatible Models:** {summary.get('compatible_models_count', 0)}",
            f"**Incompatible Models:** {summary.get('incompatible_models_count', 0)}",
            ""
        ]
        
        # Add selected model information
        if analysis_result.get('model_selection', {}).get('selected_model'):
            model_info = analysis_result['model_selection']
            report.extend([
                "## 🎯 Selected Model",
                f"**Model:** {model_info.get('selected_model')}",
                f"**Reason:** {model_info.get('reason', 'N/A')}",
                f"**Confidence:** {model_info.get('confidence', 0):.1%}",
                ""
            ])
        
        # Add recommendations
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            report.extend([
                "## 💡 Recommendations",
                ""
            ])
            
            high_priority = [r for r in recommendations if r.get('priority') == 'high']
            medium_priority = [r for r in recommendations if r.get('priority') == 'medium']
            
            if high_priority:
                report.append("### 🔴 High Priority")
                for rec in high_priority:
                    report.extend([
                        f"#### {rec.get('message', 'No message')}",
                        f"- **Action:** {rec.get('action', 'N/A')}",
                        f"- **Impact:** {rec.get('impact', 'N/A')}",
                        f"- **Category:** {rec.get('category', 'N/A')}",
                        ""
                    ])
            
            if medium_priority:
                report.append("### 🟡 Medium Priority")
                for rec in medium_priority:
                    report.extend([
                        f"#### {rec.get('message', 'No message')}",
                        f"- **Action:** {rec.get('action', 'N/A')}",
                        f"- **Impact:** {rec.get('impact', 'N/A')}",
                        f"- **Category:** {rec.get('category', 'N/A')}",
                        ""
                    ])
        
        # Add performance metrics
        if performance:
            report.extend([
                "## ⚡ Performance Metrics",
                f"- **Analysis Time:** {performance.get('analysis_time_seconds', 0):.3f}s",
                f"- **Models Analyzed:** {performance.get('models_analyzed', 0)}",
                f"- **Recommendations Generated:** {performance.get('recommendations_generated', 0)}",
                f"- **Throughput:** {performance.get('throughput_models_per_second', 0):.1f} models/second",
                f"- **Cached:** {'Yes' if performance.get('cached') else 'No'}",
                ""
            ])
        
        # Add next steps
        next_steps = summary.get('next_steps', [])
        if next_steps:
            report.extend([
                "## 👣 Next Steps",
                ""
            ])
            for step in next_steps:
                report.append(f"- {step}")
            report.append("")
        
        return "\n".join(report)

    def _generate_html_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate HTML report (simplified version)"""
        markdown_report = self._generate_markdown_report(analysis_result)
        
        # Simple markdown to HTML conversion
        html_report = markdown_report.replace('\n## ', '\n<h2>').replace('## ', '<h2>')
        html_report = html_report.replace('\n### ', '\n<h3>').replace('### ', '<h3>')
        html_report = html_report.replace('**', '<strong>').replace('**', '</strong>')
        html_report = html_report.replace('\n- ', '\n<li>').replace('- ', '<li>')
        html_report = html_report.replace('\n', '<br>\n')
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Supply Chain Forecasting Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                .critical {{ color: #e74c3c; }}
                .warning {{ color: #f39c12; }}
                .success {{ color: #27ae60; }}
            </style>
        </head>
        <body>
            {html_report}
        </body>
        </html>
        """

    def verify_return_structure(self, dataset_info: Dict[str, Any] = None):
        """Verify that analyze_dataset returns the expected structure"""
        if dataset_info is None:
            dataset_info = {
                'name': 'test_dataset',
                'columns': ['date', 'sales', 'shop_id'],
                'row_count': 100,
                'frequency': 'monthly'
            }
        
        result = self.analyze_dataset(dataset_info)
        
        print("\n🔍 VERIFYING RETURN STRUCTURE:")
        print("=" * 40)
        
        # Check required keys
        required_keys = ['validation', 'model_recommendations', 'can_proceed', 'status', 'summary']
        for key in required_keys:
            if key in result:
                print(f"✅ {key}: PRESENT")
            else:
                print(f"❌ {key}: MISSING")
        
        # Show summary content
        if 'summary' in result:
            print(f"\n📋 SUMMARY CONTENT:")
            for k, v in result['summary'].items():
                print(f"   {k}: {v}")
        
        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return {
                'analysis_cache_size': len(self._analysis_cache) if hasattr(self, '_analysis_cache') else 0,
                'compatibility_cache_size': len(self._compatibility_cache) if hasattr(self, '_compatibility_cache') else 0,
                'model_cache_available': self._model_cache is not None if hasattr(self, '_model_cache') else False,
                'total_cached_items': (
                    (len(self._analysis_cache) if hasattr(self, '_analysis_cache') else 0) +
                    (len(self._compatibility_cache) if hasattr(self, '_compatibility_cache') else 0) +
                    (1 if hasattr(self, '_model_cache') and self._model_cache is not None else 0)
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}

    def clear_cache(self):
        """Clear all caches"""
        try:
            # Clear analysis cache
            if hasattr(self, '_analysis_cache'):
                self._analysis_cache.clear()
                logger.info("🧹 Cleared analysis cache")
            
            # Clear compatibility cache  
            if hasattr(self, '_compatibility_cache'):
                self._compatibility_cache.clear()
                logger.info("🧹 Cleared compatibility cache")
            
            # Clear model cache
            if hasattr(self, '_model_cache'):
                self._model_cache = None
                logger.info("🧹 Cleared model cache")
                
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")


class MinimalRuleEngine:
    """
    Minimal fallback rule engine that works without dependencies
    """
    def __init__(self):
        self.rules_loaded = False
        self._load_rules()
    
    def _load_rules(self):
        """Try to load rules directly"""
        try:
            import yaml
            # Try multiple possible paths
            possible_paths = [
                'app/knowledge_base/rule_layer/model_selection_rules.yaml',
                './app/knowledge_base/rule_layer/model_selection_rules.yaml',
                '../knowledge_base/rule_layer/model_selection_rules.yaml'
            ]
            
            for rules_path in possible_paths:
                if os.path.exists(rules_path):
                    with open(rules_path, 'r') as f:
                        self.rules = yaml.safe_load(f)
                    self.rules_loaded = True
                    logger.info(f"✅ Minimal engine loaded YAML rules from: {rules_path}")
                    break
            else:
                logger.warning("❌ No YAML rules file found")
                self.rules = {'rules': []}
                
        except Exception as e:
            logger.error(f"❌ Minimal engine failed to load rules: {e}")
            self.rules = {'rules': []}
    
    def validate_dataset(self, dataset_info):
        """Basic validation - accepts only metadata"""
        errors = []
        warnings = []
        
        # Check for required metadata
        if dataset_info.get('row_count', 0) < 12:
            errors.append("Insufficient data points (minimum 12 required)")
        
        if dataset_info.get('missing_percentage', 0) > 0.3:
            errors.append("Too many missing values (>30%)")
        
        # Check for required columns
        columns = dataset_info.get('columns', [])
        if not any(col in columns for col in ['sales', 'demand', 'quantity']):
            warnings.append("No obvious target variable found (sales, demand, quantity)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def select_model(self, dataset_info):
        """Basic model selection using simple rules - metadata only"""
        columns = dataset_info.get('columns', [])
        frequency = dataset_info.get('frequency', '')
        row_count = dataset_info.get('row_count', 0)
        
        # Simple rule-based selection
        if self.rules_loaded and self.rules.get('rules'):
            # Try to use YAML rules if available
            for rule in self.rules['rules']:
                if self._evaluate_simple_condition(rule.get('condition', ''), dataset_info):
                    model_name = self._extract_model_name(rule.get('action', ''))
                    if model_name:
                        return {
                            'selected_model': model_name,
                            'confidence': rule.get('priority', 5) / 10.0,
                            'reason': rule.get('message', 'Rule matched')
                        }
        
        # Fallback to simple logic
        if frequency == 'monthly' and any(col in columns for col in ['shop_id', 'store_id']) and row_count >= 12:
            return {
                'selected_model': 'Monthly_Shop_Sales_Forecaster',
                'confidence': 0.8,
                'reason': 'Monthly shop data detected'
            }
        elif frequency == 'daily' and any(col in columns for col in ['shop_id', 'store_id']) and row_count >= 30:
            return {
                'selected_model': 'Daily_Shop_Sales_Forecaster', 
                'confidence': 0.8,
                'reason': 'Daily shop data detected'
            }
        elif 'date' in columns and any(col in columns for col in ['demand', 'sales']) and row_count >= 50:
            return {
                'selected_model': 'Prophet',
                'confidence': 0.7,
                'reason': 'Time series data detected'
            }
        
        return {
            'selected_model': None,
            'confidence': 0.0,
            'reason': 'No matching model found'
        }
    

    def _evaluate_simple_condition(self, condition: str, dataset_info: Dict[str, Any]) -> bool:
        """Simple condition evaluator"""
        if not condition:
            return False
            
        try:
            # Replace common patterns
            condition = condition.replace('dataset.', 'dataset_info.get(')
            condition = condition.replace(' in dataset.columns', " in dataset_info.get('columns', [])")
            
            # Add closing parentheses
            import re
            condition = re.sub(r'dataset_info\.get\((\w+)(?=[),])', r"dataset_info.get('\1', None)", condition)
            
            # Safe evaluation
            result = eval(condition, {'dataset_info': dataset_info})
            return bool(result)
        except Exception as e:
            logger.debug(f"Condition evaluation failed: {e}")
            return False
    # use shared_utils.py
    # def _extract_model_name(self, action: str) -> str:
    #     """Extract model name from action string"""
    #     if "model_name =" in action:
    #         import re
    #         match = re.search(r"model_name\s*=\s*['\"]([^'\"]+)['\"]", action)
    #         if match:
    #             return match.group(1)
    #     return ""


def demo_enhanced_rule_engine():
    """Test the enhanced rule engine with all new features"""
    print("🚀 ENHANCED RULE ENGINE WITH CACHING & EXPORT CAPABILITIES")
    print("=" * 60)
    
    service = RuleEngineService()
    
    # Test different scenarios
    test_scenarios = [
        {
            'name': '✅ Compatible Dataset',
            'dataset': {
                'name': 'compatible_shop_data',
                'frequency': 'monthly',
                'granularity': 'shop_level',
                'row_count': 48,
                'columns': ['shop_id', 'date', 'sales'],
                'missing_percentage': 0.02
            }
        },
        {
            'name': '❌ Missing Target Variable',
            'dataset': {
                'name': 'no_target_data',
                'frequency': 'monthly',
                'granularity': 'shop_level',
                'row_count': 48,
                'columns': ['shop_id', 'date', 'price'],  # Missing sales/demand
                'missing_percentage': 0.02
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*50}")
        
        try:
            # First analysis (uncached)
            print("📊 First Analysis (Uncached):")
            start_time = time.time()
            result = service.analyze_dataset(scenario['dataset'])
            first_analysis_time = time.time() - start_time
            
            # Second analysis (cached)
            print("📊 Second Analysis (Cached):")
            start_time = time.time()
            cached_result = service.analyze_dataset(scenario['dataset'])
            cached_analysis_time = time.time() - start_time
            
            # ✅ FIX: Use the correct key 'model_recommendations' instead of 'selection_analysis'
            summary = result['summary']
            
            print(f"\n📋 SUMMARY: {summary['status_message']}")
            print(f"🎯 Confidence: {summary['confidence']:.1%}")
            print(f"⏱️ Performance: {first_analysis_time:.3f}s (uncached) vs {cached_analysis_time:.3f}s (cached)")
            print(f"📈 Speed Improvement: {first_analysis_time/max(cached_analysis_time, 0.001):.1f}x faster")
            
            if not summary['can_proceed']:
                print(f"🚫 Rejection Reason: {summary.get('rejection_reason', 'Unknown')}")
            
            # ✅ FIX: Update this section to use the correct structure
            model_recs = result['model_recommendations']
            print(f"🔍 Model Recommendations: {len(model_recs.get('ranked_models', []))} models ranked")
            print(f"📊 Top Model: {model_recs.get('top_recommendation', 'None')}")
            
            # Test export functionality
            print(f"\n📄 EXPORT TEST:")
            markdown_report = service.export_analysis_report(result, 'markdown')
            print(f"📝 Markdown Report: {len(markdown_report)} characters")
            
            json_report = service.export_analysis_report(result, 'json')
            print(f"📊 JSON Report: {len(json_report)} characters")
            
            # html_report = service.export_analysis_report(result, 'html')
            # print(f"🌐 HTML Report: {len(html_report)} characters")
            
        except Exception as e:
            print(f"❌ Error in scenario: {e}")
            import traceback
            traceback.print_exc()
    
    # Show cache statistics
    cache_stats = service.get_cache_stats()
    print(f"\n💾 CACHE STATISTICS:")
    print(f"   Analysis Cache: {cache_stats['analysis_cache_size']} items")
    print(f"   Compatibility Cache: {cache_stats['compatibility_cache_size']} items")
    print(f"   Total Cached: {cache_stats['total_cached_items']} items")
    
    # Clear cache and show stats again
    service.clear_cache()
    cache_stats_after = service.get_cache_stats()
    print(f"   After Clear: {cache_stats_after['total_cached_items']} items")
    
    print("\n🎉 ENHANCED DEMO COMPLETED SUCCESSFULLY!")
if __name__ == "__main__":
    demo_enhanced_rule_engine()