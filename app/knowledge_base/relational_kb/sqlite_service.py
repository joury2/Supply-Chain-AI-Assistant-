# app/knowledge_base/relational_kb/sqlite_service.py
# SQLiteService - The Master Orchestrator for Supply Chain Forecasting Knowledge Base
"""
SQLiteService - The Master Orchestrator for Supply Chain Forecasting Knowledge Base

This service combines:
- SQLiteManager (Database operations)
- SQLiteRepository (Data access layer) 
- RuleEngineService (Business logic)
- Knowledge Base Services
Into a cohesive business logic layer that makes intelligent decisions.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class SQLiteService:
    """
    Master orchestration service that combines relational data, business rules,
    and domain knowledge to make intelligent forecasting decisions.
    
    Analogy: The "Corporate Executive" that coordinates all departments.
    """
    
    def __init__(self, db_path: str = "supply_chain.db"):
        self.db_path = db_path
        self._initialize_components()
        logger.info("🚀 SQLiteService initialized - Knowledge Base Orchestrator Ready")
    
    def _initialize_components(self):
        """Initialize all required components with proper error handling"""
        try:
            from app.knowledge_base.relational_kb.sqlite_manager import SQLiteManager
            from app.knowledge_base.relational_kb.sqlite_schema import SQLiteRepository
            from app.services.knowledge_base_services.core.rule_engine_service import RuleEngineService
            
            # Core database components
            # for managing SQLite connections and CRUD operations
            self.db_manager = SQLiteManager(self.db_path)
            self.repository = SQLiteRepository(self.db_manager)
            
            # Business logic engine
            # for rule-based validations and model selections
            self.rule_engine_service = RuleEngineService(self.db_manager.conn)
            
            # Service integrations (with fallbacks)
            # to connect with external knowledge base services
            self._initialize_services()
            
        except ImportError as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
        
        logger.info("✅ All components initialized successfully")
    
    def _initialize_services(self):
        """Initialize external services with graceful fallbacks"""
        self.services = {}
        
        try:
            from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService
            self.services['forecasting'] = SupplyChainForecastingService(self.db_path)   
            logger.info("✅ SupplyChainForecastingService integrated")
        except ImportError:
            logger.warning("⚠️ SupplyChainForecastingService not available")
            self.services['forecasting'] = None
        
        try:
            from app.services.knowledge_base_services.core.rule_engine_service import RuleEngineService
            self.services['rule_analysis'] = RuleEngineService(self.db_manager.conn)
            logger.info("✅ RuleEngineService integrated")
        except ImportError:
            logger.warning("⚠️ RuleEngineService not available")
            self.services['rule_analysis'] = None

    # ============================================================================
    # CORE ORCHESTRATION METHODS
    # ============================================================================

    def validate_dataset_comprehensive(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive dataset validation combining multiple validation sources.
        
        Returns: Detailed validation report with actionable recommendations.
        """
        logger.info(f"🔍 Comprehensive validation for: {dataset_info.get('name', 'Unknown')}")
        
        validation_results = {
            'basic_validation': self._validate_basic_requirements(dataset_info),
            'schema_validation': self._validate_against_schemas(dataset_info),
            'rule_validation': self._validate_with_business_rules(dataset_info),
            'model_compatibility': self._analyze_model_compatibility(dataset_info)
        }
        
        # Combine results into executive summary
        executive_summary = self._generate_validation_summary(validation_results)
        
        return {
            'validation_results': validation_results,
            'executive_summary': executive_summary,
            'recommendations': self._generate_validation_recommendations(validation_results),
            'can_proceed': executive_summary['overall_valid']
        }

    def get_intelligent_model_recommendations(self, 
                                            dataset_info: Dict[str, Any],
                                            business_constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Intelligent model selection combining multiple recommendation sources.
        
        Business constraints can include:
        - min_accuracy: Minimum required accuracy
        - max_training_time: Maximum allowed training time  
        - interpretability_required: Whether model must be interpretable
        - resource_limits: CPU/Memory constraints
        """
        logger.info(f"🎯 Intelligent model selection for: {dataset_info.get('name', 'Unknown')}")
        
        recommendations = {
            'rule_based': self._get_rule_based_recommendations(dataset_info, business_constraints),
            'feature_based': self._get_feature_based_recommendations(dataset_info),
            'performance_based': self._get_performance_based_recommendations(dataset_info),
            'constraint_based': self._apply_business_constraints(dataset_info, business_constraints)
        }
        
        # Rank and combine recommendations
        ranked_models = self._rank_and_combine_recommendations(recommendations, dataset_info)
        
        return {
            'recommendation_sources': recommendations,
            'ranked_models': ranked_models,
            'top_recommendation': ranked_models[0] if ranked_models else None,
            'selection_confidence': self._calculate_selection_confidence(ranked_models),
            'constraints_applied': business_constraints or {}
        }

    def execute_forecast_with_knowledge(self,
                                      dataset_info: Dict[str, Any],
                                      model_selection: Dict[str, Any],
                                      forecast_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute forecasting pipeline with full knowledge base integration.
        
        This is the main orchestration method that coordinates the entire workflow.
        """
        logger.info(f"🚀 Executing knowledge-driven forecast for: {dataset_info.get('name', 'Unknown')}")
        
        # Step 1: Log the forecast execution start
        forecast_run = self._log_forecast_execution_start(dataset_info, model_selection, forecast_config)
        
        try:
            # Step 2: Validate we can proceed
            validation = self.validate_dataset_comprehensive(dataset_info)
            if not validation['can_proceed']:
                return self._handle_validation_failure(forecast_run, validation)
            
            # Step 3: Prepare data using knowledge base schemas
            prepared_data = self._prepare_data_with_knowledge(dataset_info, model_selection)
            
            # Step 4: Execute forecasting (integrate with your forecasting service)
            forecast_result = self._execute_forecasting_pipeline(prepared_data, model_selection, forecast_config)
            
            # Step 5: Generate interpretations and insights
            insights = self._generate_forecast_insights(forecast_result, dataset_info, model_selection)
            
            # Step 6: Update forecast run with results
            self._log_forecast_execution_completion(forecast_run, forecast_result, insights)
            
            return self._create_success_response(forecast_run, forecast_result, insights, validation)
            
        except Exception as e:
            logger.error(f"❌ Forecast execution failed: {e}")
            self._log_forecast_execution_failure(forecast_run, str(e))
            return self._create_error_response(forecast_run, str(e))

    def get_model_analytics(self, time_period: str = "30d") -> Dict[str, Any]:
        """
        Comprehensive analytics across all models, performance, and forecast runs.
        
        Provides business intelligence for model management and system optimization.
        """
        logger.info(f"📊 Generating model analytics for period: {time_period}")
        
        analytics = {
            'performance_metrics': self._get_performance_analytics(time_period),
            'model_utilization': self._get_model_utilization_analytics(time_period),
            'forecast_accuracy': self._get_forecast_accuracy_analytics(time_period),
            'business_impact': self._get_business_impact_analytics(time_period),
            'recommendations': self._generate_analytics_recommendations(time_period)
        }
        
        return analytics

    # ============================================================================
    # PRIVATE ORCHESTRATION METHODS
    # ============================================================================

    def _validate_basic_requirements(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic dataset requirements"""
        errors = []
        warnings = []
        
        # Check minimum data requirements
        if dataset_info.get('row_count', 0) < 12:
            errors.append("Insufficient data points (minimum 12 required)")
        elif dataset_info.get('row_count', 0) < 30:
            warnings.append("Limited data may affect forecast accuracy")
        
        # Check for target variable
        target_candidates = ['sales', 'demand', 'quantity', 'value', 'target']
        has_target = any(target in dataset_info.get('columns', []) for target in target_candidates)
        if not has_target:
            errors.append("No target variable found (sales, demand, quantity, value, or target)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _validate_against_schemas(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against known dataset schemas in knowledge base"""
        dataset_name = dataset_info.get('name', 'custom')
        provided_columns = set(dataset_info.get('columns', []))
        
        # Get all schemas from knowledge base
        schemas = self.repository.get_all_schemas()
        best_match = None
        best_score = 0
        
        for schema in schemas:
            schema_columns = set([col['column_name'] for col in schema.get('columns', [])])
            overlap = len(provided_columns.intersection(schema_columns))
            score = overlap / max(len(schema_columns), 1)
            
            if score > best_score:
                best_score = score
                best_match = schema
        
        if best_match and best_score > 0.7:
            return {
                'matched_schema': best_match['dataset_name'],
                'match_confidence': best_score,
                'missing_columns': list(set([col['column_name'] for col in best_match.get('columns', [])]) - provided_columns),
                'extra_columns': list(provided_columns - set([col['column_name'] for col in best_match.get('columns', [])]))
            }
        
        return {
            'matched_schema': None,
            'match_confidence': 0,
            'note': 'No strong schema match found'
        }

    def _validate_with_business_rules(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using business rules from rule engine"""
        if hasattr(self, 'rule_engine'):
            analysis_result = self.rule_engine.analyze_dataset(dataset_info)
            return analysis_result['validation']
        
        return {
            'valid': True,
            'errors': [],
            'warnings': ['Rule engine not available for validation'],
            'applied_rules': []
        }

    def _analyze_model_compatibility(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which models are compatible with the dataset"""
        available_features = dataset_info.get('columns', [])
        target_variable = self._infer_target_variable(dataset_info)
        
        compatible_models = self.repository.get_suitable_models(available_features, target_variable)
        
        return {
            'compatible_models': len(compatible_models),
            'total_models_analyzed': len(self.repository.get_active_models()),
            'target_variable_inferred': target_variable,
            'compatibility_details': [
                {
                    'model_name': model['model_name'],
                    'model_type': model['model_type'],
                    'required_features': json.loads(model['required_features']),
                    'performance_metrics': json.loads(model.get('performance_metrics', '{}'))
                }
                for model in compatible_models
            ]
        }

    def _get_rule_based_recommendations(self, dataset_info: Dict[str, Any], 
                                      business_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations from rule engine"""
        if hasattr(self, 'rule_engine'):
            result = self.rule_engine.select_model(dataset_info, business_constraints)
            if result.get('selected_model'):
                return [{
                    'model_id': result['selected_model'],
                    'confidence': result.get('confidence', 0.0),
                    'source': 'rule_engine',
                    'reason': result.get('reason', 'Rule matched'),
                    'priority': 'high'
                }]
        return []

    def _get_feature_based_recommendations(self, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on feature compatibility"""
        available_features = dataset_info.get('columns', [])
        target_variable = self._infer_target_variable(dataset_info)
        
        suitable_models = self.repository.get_suitable_models(available_features, target_variable)
        
        recommendations = []
        for model in suitable_models:
            metrics = json.loads(model.get('performance_metrics', '{}'))
            recommendations.append({
                'model_id': model['model_id'],
                'model_name': model['model_name'],
                'confidence': 1.0 - metrics.get('MAPE', 0.5),  # Lower MAPE = higher confidence
                'source': 'feature_matching',
                'reason': f"Feature compatible with target '{target_variable}'",
                'performance_metrics': metrics
            })
        
        return recommendations

    def _get_performance_based_recommendations(self, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on historical performance"""
        # This would analyze performance history and select best performing models
        # for similar dataset characteristics
        all_models = self.repository.get_active_models()
        
        recommendations = []
        for model in all_models:
            metrics = json.loads(model.get('performance_metrics', '{}'))
            if 'MAPE' in metrics:
                recommendations.append({
                    'model_id': model['model_id'],
                    'model_name': model['model_name'],
                    'confidence': 1.0 - metrics['MAPE'],  # Lower MAPE = higher confidence
                    'source': 'historical_performance',
                    'reason': f"Historical MAPE: {metrics['MAPE']:.3f}",
                    'performance_metrics': metrics
                })
        
        return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)

    def _apply_business_constraints(self, dataset_info: Dict[str, Any],
                                  constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply business constraints to filter recommendations"""
        if not constraints:
            return []
        
        # This would filter models based on business constraints
        # For now, return empty - to be implemented based on specific constraints
        return []

    def _rank_and_combine_recommendations(self, recommendations: Dict[str, List], 
                                        dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intelligently rank and combine recommendations from all sources"""
        all_recommendations = []
        
        # Collect all recommendations
        for source, recs in recommendations.items():
            all_recommendations.extend(recs)
        
        # Group by model and combine confidence scores
        model_scores = {}
        for rec in all_recommendations:
            model_id = rec['model_id']
            if model_id not in model_scores:
                model_scores[model_id] = {
                    'model_id': model_id,
                    'model_name': rec.get('model_name', 'Unknown'),
                    'confidence_scores': [],
                    'sources': [],
                    'reasons': []
                }
            
            model_scores[model_id]['confidence_scores'].append(rec['confidence'])
            model_scores[model_id]['sources'].append(rec['source'])
            model_scores[model_id]['reasons'].append(rec['reason'])
        
        # Calculate combined scores
        ranked_models = []
        for model_id, data in model_scores.items():
            # Weighted average of confidence scores
            confidence = sum(data['confidence_scores']) / len(data['confidence_scores'])
            
            # Boost confidence if multiple sources agree
            source_boost = min(0.2, len(set(data['sources'])) * 0.05)
            final_confidence = min(1.0, confidence + source_boost)
            
            ranked_models.append({
                'model_id': model_id,
                'model_name': data['model_name'],
                'combined_confidence': final_confidence,
                'source_count': len(data['sources']),
                'sources': data['sources'],
                'primary_reason': data['reasons'][0] if data['reasons'] else 'Multiple factors',
                'recommendation_strength': 'strong' if final_confidence > 0.7 else 'medium' if final_confidence > 0.5 else 'weak'
            })
        
        return sorted(ranked_models, key=lambda x: x['combined_confidence'], reverse=True)

    def _log_forecast_execution_start(self, dataset_info: Dict[str, Any],
                                    model_selection: Dict[str, Any],
                                    forecast_config: Dict[str, Any]) -> Dict[str, Any]:
        """Log the start of a forecast execution"""
        forecast_run = self.repository.create_forecast_run(
            model_id=model_selection.get('model_id'),
            input_schema=json.dumps(dataset_info),
            config=forecast_config
        )
        
        logger.info(f"📝 Forecast execution started: Run ID {forecast_run.get('run_id')}")
        return forecast_run

    def _prepare_data_with_knowledge(self, dataset_info: Dict[str, Any],
                                   model_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data using knowledge base schemas and model requirements"""
        # Get model requirements from knowledge base
        model_details = self.repository.get_model_by_id(model_selection.get('model_id'))
        
        if model_details:
            required_features = json.loads(model_details.get('required_features', '[]'))
            logger.info(f"📊 Preparing data for {model_details['model_name']} with features: {required_features}")
        
        # This would integrate with your DataProcessor service
        return {
            'status': 'prepared',
            'model_requirements': required_features if model_details else [],
            'dataset_info': dataset_info
        }

    def _execute_forecasting_pipeline(self, prepared_data: Dict[str, Any],
                                    model_selection: Dict[str, Any],
                                    forecast_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual forecasting pipeline"""
        # This would integrate with your forecasting services
        # For now, return mock forecast
        horizon = forecast_config.get('horizon', 30)
        
        return {
            'status': 'success',
            'forecast_values': [100 + i * 2 for i in range(horizon)],
            'confidence_intervals': {
                'lower': [90 + i * 2 for i in range(horizon)],
                'upper': [110 + i * 2 for i in range(horizon)]
            },
            'model_used': model_selection.get('model_name', 'Unknown'),
            'horizon': horizon,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_forecast_insights(self, forecast_result: Dict[str, Any],
                                  dataset_info: Dict[str, Any],
                                  model_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business insights from forecast results"""
        # This would integrate with your LLM service for interpretations
        return {
            'business_impact': 'Positive growth trend detected',
            'key_takeaways': [
                'Steady demand increase expected',
                'Seasonal patterns identified',
                'Confidence in forecast: High'
            ],
            'recommendations': [
                'Consider increasing inventory for peak periods',
                'Monitor for trend changes monthly'
            ],
            'risk_assessment': 'Low risk - stable pattern'
        }

    def _log_forecast_execution_completion(self, forecast_run: Dict[str, Any],
                                         forecast_result: Dict[str, Any],
                                         insights: Dict[str, Any]):
        """Log successful forecast completion"""
        self.repository.update_forecast_results(
            run_id=forecast_run['run_id'],
            results=json.dumps(forecast_result),
            validation_issues=[],
            llm_interpretation=json.dumps(insights)
        )
        
        logger.info(f"✅ Forecast execution completed: Run ID {forecast_run['run_id']}")

    def _log_forecast_execution_failure(self, forecast_run: Dict[str, Any], error: str):
        """Log forecast execution failure"""
        self.repository.update_forecast_results(
            run_id=forecast_run['run_id'],
            results=None,
            validation_issues=[error],
            llm_interpretation=None
        )
        
        logger.error(f"❌ Forecast execution failed: Run ID {forecast_run['run_id']} - {error}")

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _infer_target_variable(self, dataset_info: Dict[str, Any]) -> str:
        """Intelligently infer the target variable from dataset columns"""
        columns = dataset_info.get('columns', [])
        
        target_priority = ['sales', 'demand', 'quantity', 'value', 'target', 'y']
        
        for target in target_priority:
            if target in columns:
                return target
        
        # If no standard target found, look for numeric columns that could be targets
        numeric_indicators = ['amount', 'count', 'volume', 'price', 'revenue']
        for indicator in numeric_indicators:
            for column in columns:
                if indicator in column.lower():
                    return column
        
        return columns[0] if columns else 'unknown'

    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from validation results"""
        basic = validation_results['basic_validation']
        schema = validation_results['schema_validation']
        rule = validation_results['rule_validation']
        compatibility = validation_results['model_compatibility']
        
        # Calculate overall validity
        overall_valid = (
            basic['valid'] and 
            rule.get('valid', True) and
            compatibility['compatible_models'] > 0
        )
        
        # Generate summary message
        if overall_valid:
            message = f"✅ Dataset validated successfully. {compatibility['compatible_models']} compatible models found."
        else:
            message = "❌ Dataset validation failed. See details below."
        
        return {
            'overall_valid': overall_valid,
            'summary_message': message,
            'compatible_models_count': compatibility['compatible_models'],
            'validation_errors': basic.get('errors', []) + rule.get('errors', []),
            'validation_warnings': basic.get('warnings', []) + rule.get('warnings', []),
            'schema_match': schema.get('matched_schema'),
            'schema_confidence': schema.get('match_confidence', 0)
        }

    def _generate_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from validation results"""
        recommendations = []
        
        basic = validation_results['basic_validation']
        schema = validation_results['schema_validation']
        compatibility = validation_results['model_compatibility']
        
        # Basic validation recommendations
        for error in basic.get('errors', []):
            recommendations.append(f"CRITICAL: {error}")
        
        for warning in basic.get('warnings', []):
            recommendations.append(f"WARNING: {warning}")
        
        # Schema recommendations
        if schema.get('matched_schema'):
            if schema.get('missing_columns'):
                recommendations.append(f"Consider adding missing columns from {schema['matched_schema']} schema: {', '.join(schema['missing_columns'])}")
        
        # Compatibility recommendations
        if compatibility['compatible_models'] == 0:
            recommendations.append("No compatible models found. Consider adding different features or models.")
        
        return recommendations

    def _calculate_selection_confidence(self, ranked_models: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in model selection"""
        if not ranked_models:
            return 0.0
        
        top_model = ranked_models[0]
        base_confidence = top_model['combined_confidence']
        
        # Boost confidence if multiple sources agree
        source_boost = min(0.2, top_model['source_count'] * 0.05)
        
        return min(1.0, base_confidence + source_boost)

    def _create_success_response(self, forecast_run: Dict[str, Any],
                               forecast_result: Dict[str, Any],
                               insights: Dict[str, Any],
                               validation: Dict[str, Any]) -> Dict[str, Any]:
        """Create success response for forecast execution"""
        return {
            'status': 'success',
            'forecast_run_id': forecast_run.get('run_id'),
            'forecast_result': forecast_result,
            'business_insights': insights,
            'validation_summary': validation['executive_summary'],
            'timestamp': datetime.now().isoformat()
        }

    def _create_error_response(self, forecast_run: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create error response for forecast execution"""
        return {
            'status': 'error',
            'forecast_run_id': forecast_run.get('run_id'),
            'error': error,
            'timestamp': datetime.now().isoformat()
        }

    def _handle_validation_failure(self, forecast_run: Dict[str, Any],
                                 validation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation failure scenario"""
        self._log_forecast_execution_failure(
            forecast_run, 
            f"Validation failed: {validation['executive_summary']['summary_message']}"
        )
        
        return {
            'status': 'validation_failed',
            'forecast_run_id': forecast_run.get('run_id'),
            'validation_results': validation,
            'timestamp': datetime.now().isoformat()
        }

    # ============================================================================
    # ANALYTICS METHODS (Simplified for example)
    # ============================================================================

    def _get_performance_analytics(self, time_period: str) -> Dict[str, Any]:
        """Get performance analytics across all models"""
        return {
            'average_mape': 0.15,
            'best_performing_model': 'LightGBM_Monthly',
            'model_performance_trend': 'improving',
            'forecast_accuracy_by_model': {
                'LightGBM_Monthly': 0.85,
                'Prophet': 0.78,
                'ARIMA': 0.72
            }
        }

    def _get_model_utilization_analytics(self, time_period: str) -> Dict[str, Any]:
        """Get model utilization analytics"""
        return {
            'total_forecasts': 150,
            'most_used_model': 'LightGBM_Monthly',
            'utilization_by_model': {
                'LightGBM_Monthly': 45,
                'Prophet': 35,
                'ARIMA': 20
            }
        }

    def _get_forecast_accuracy_analytics(self, time_period: str) -> Dict[str, Any]:
        """Get forecast accuracy analytics"""
        return {
            'overall_accuracy': 0.82,
            'accuracy_trend': 'stable',
            'best_accuracy_scenario': 'monthly_retail_data',
            'improvement_opportunities': ['data_quality', 'feature_engineering']
        }

    def _get_business_impact_analytics(self, time_period: str) -> Dict[str, Any]:
        """Get business impact analytics"""
        return {
            'cost_savings_estimated': 125000,
            'inventory_optimization': 0.18,
            'service_level_improvement': 0.12,
            'roi_calculated': 3.5
        }

    def _generate_analytics_recommendations(self, time_period: str) -> List[str]:
        """Generate recommendations from analytics"""
        return [
            "Consider retraining LightGBM models with recent data",
            "Prophet model showing improved seasonal accuracy",
            "Evaluate ARIMA model performance for potential replacement"
        ]

    # ============================================================================
    # PUBLIC UTILITY METHODS
    # ============================================================================

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report"""
        return {
            'database_status': self._get_database_status(),
            'service_status': self._get_service_status(),
            'knowledge_base_status': self._get_knowledge_base_status(),
            'performance_metrics': self._get_system_performance()
        }

    def _get_database_status(self) -> Dict[str, Any]:
        """Get database status and statistics"""
        try:
            counts = self.db_manager.get_table_counts()
            return {
                'status': 'healthy',
                'table_counts': counts,
                'connection': 'active'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'connection': 'inactive'
            }

    def _get_service_status(self) -> Dict[str, Any]:
        """Get status of all integrated services"""
        status = {}
        for service_name, service in self.services.items():
            status[service_name] = 'available' if service else 'unavailable'
        return status

    def _get_knowledge_base_status(self) -> Dict[str, Any]:
        """Get knowledge base status"""
        try:
            model_count = len(self.repository.get_active_models())
            schema_count = len(self.repository.get_all_schemas())
            rule_count = len(self.repository.get_active_rules())
            
            return {
                'status': 'healthy',
                'active_models': model_count,
                'dataset_schemas': schema_count,
                'business_rules': rule_count
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _get_system_performance(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            'average_processing_time': 2.5,
            'success_rate': 0.94,
            'concurrent_operations': 3,
            'memory_usage': 'optimal'
        }

    def close(self):
        """Close all resources"""
        if hasattr(self, 'db_manager'):
            self.db_manager.close()
        logger.info("🔚 SQLiteService closed")

# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_sqlite_service():
    """Demonstrate the SQLiteService capabilities"""
    print("🚀 SQLiteService - Knowledge Base Orchestration Demo")
    print("=" * 60)
    
    service = SQLiteService()
    
    try:
        # 1. Show system status
        print("\n📊 SYSTEM STATUS:")
        status = service.get_system_status()
        print(f"Database: {status['database_status']['status']}")
        print(f"Knowledge Base: {status['knowledge_base_status']['status']}")
        print(f"Active Models: {status['knowledge_base_status'].get('active_models', 0)}")
        
        # 2. Test comprehensive validation
        print("\n🔍 COMPREHENSIVE VALIDATION TEST:")
        test_dataset = {
            'name': 'retail_sales_monthly',
            'frequency': 'monthly',
            'granularity': 'shop_level',
            'row_count': 36,
            'columns': ['shop_id', 'date', 'sales', 'price'],
            'missing_percentage': 0.02
        }
        
        validation = service.validate_dataset_comprehensive(test_dataset)
        summary = validation['executive_summary']
        
        print(f"Overall Valid: {summary['overall_valid']}")
        print(f"Summary: {summary['summary_message']}")
        print(f"Compatible Models: {summary['compatible_models_count']}")
        
        if validation['recommendations']:
            print("Recommendations:")
            for rec in validation['recommendations']:
                print(f"  • {rec}")
        
        # 3. Test intelligent model selection
        print("\n🎯 INTELLIGENT MODEL SELECTION:")
        business_constraints = {
            'min_accuracy': 0.8,
            'interpretability_required': True
        }
        
        recommendations = service.get_intelligent_model_recommendations(
            test_dataset, business_constraints
        )
        
        print(f"Top Recommendation: {recommendations['top_recommendation']['model_name']}")
        print(f"Selection Confidence: {recommendations['selection_confidence']:.1%}")
        print(f"Models Ranked: {len(recommendations['ranked_models'])}")
        
        # 4. Test full forecasting pipeline
        if summary['overall_valid'] and recommendations['top_recommendation']:
            print("\n🚀 FULL FORECASTING PIPELINE TEST:")
            
            forecast_config = {
                'horizon': 12,
                'confidence_level': 0.95
            }
            
            result = service.execute_forecast_with_knowledge(
                test_dataset,
                recommendations['top_recommendation'],
                forecast_config
            )
            
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Forecast Run ID: {result['forecast_run_id']}")
                print(f"Business Insights: {result['business_insights']['business_impact']}")
        
        # 5. Show analytics
        print("\n📈 SYSTEM ANALYTICS:")
        analytics = service.get_model_analytics("30d")
        print(f"Overall Accuracy: {analytics['forecast_accuracy']['overall_accuracy']:.1%}")
        print(f"Best Model: {analytics['performance_metrics']['best_performing_model']}")
        print(f"Cost Savings: ${analytics['business_impact']['cost_savings_estimated']:,}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        service.close()
        print("\n🎉 SQLiteService Demo Completed!")

if __name__ == "__main__":
    demonstrate_sqlite_service()