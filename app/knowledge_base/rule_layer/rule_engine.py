# app/knowledge_base/rule_layer/rule_engine.py
"""
FIXED RuleEngine - No circular dependencies
Uses ModelRepository for data access
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional
import re
# In rule_engine.py
from app.services.shared_utils import (
    sanitize_dataset_info,
    extract_model_name_from_action,
    validate_dataset_basic
)

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Core Rule Engine - NO CIRCULAR DEPENDENCIES
    Uses repository pattern for data access
    """
    
    def __init__(self, model_repository=None, rules_file: str = None):
        """
        Initialize with optional repository injection
        
        Args:
            model_repository: Optional ModelRepository instance
            rules_file: Optional path to selection rules YAML file
        """
        self.model_repository = model_repository
        self.validation_rules = []
        self.selection_rules = {}
        self.rules_loaded = False
        
        # ✅ FIX: Provide default rules file path
        if rules_file is None:
            rules_file = "app/knowledge_base/rule_layer/model_selection_rules.yaml"
        
        # Load YAML rules with the provided file path
        self._load_validation_rules()
        self.selection_rules = self._load_selection_rules(rules_file)
        
        # Check if rules loaded successfully
        self.rules_loaded = len(self.validation_rules) > 0 and len(self.selection_rules) > 0
        
        if self.rules_loaded:
            logger.info(f"✅ RuleEngine initialized with {len(self.validation_rules)} validation + {len(self.selection_rules)} selection rules")
        else:
            logger.warning("⚠️ RuleEngine initialized but no YAML rules loaded")

    def _get_repository(self):
        """Lazy-load repository if needed"""
        if self.model_repository is None:
            from app.repositories.model_repository import get_model_repository
            self.model_repository = get_model_repository()
        return self.model_repository
    
    def _load_validation_rules(self):
        """Load data validation rules from YAML"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            possible_paths = [
                os.path.join(current_dir, 'data_validation_rules.yaml'),
                'app/knowledge_base/rule_layer/data_validation_rules.yaml',
            ]
            
            for rules_path in possible_paths:
                if os.path.exists(rules_path):
                    with open(rules_path, 'r') as f:
                        rules_data = yaml.safe_load(f)
                    self.validation_rules = rules_data.get('rules', [])
                    logger.info(f"✅ Loaded {len(self.validation_rules)} validation rules from: {rules_path}")
                    return
            
            logger.warning("❌ No validation rules YAML file found")
            self.validation_rules = []
            
        except Exception as e:
            logger.error(f"❌ Failed to load validation rules: {e}")
            self.validation_rules = []
    
    def _load_selection_rules(self, rules_file: str) -> Dict[str, Dict]:
        """Load model selection rules from YAML file - FIXED FOR LIST FORMAT"""
        try:
            with open(rules_file, 'r') as f:
                rules_data = yaml.safe_load(f)
            
            # ✅ FIX: Handle list format (your current YAML structure)
            if isinstance(rules_data.get('rules'), list):
                rules_dict = {}
                for rule_obj in rules_data['rules']:
                    rule_name = rule_obj['name']
                    rules_dict[rule_name] = {
                        'description': rule_obj.get('description', ''),
                        'condition': rule_obj.get('condition', ''),
                        'action': rule_obj.get('action', ''),
                        'rule_type': 'model_selection',
                        'priority': rule_obj.get('priority', 1),
                        'message': rule_obj.get('message', '')
                    }
                return rules_dict
            else:
                # Fallback to old dict format
                return rules_data.get('rules', {})
            
        except Exception as e:
            logger.error(f"❌ Failed to load selection rules from {rules_file}: {e}")
            return {}



    def validate_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dataset against validation rules - METADATA ONLY
        
        Args:
            dataset_info: Dictionary containing dataset METADATA (no DataFrame)
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        applied_rules = []
        
        # Sanitize to remove DataFrame objects
        sanitized_info = sanitize_dataset_info(dataset_info)
        
        # Basic validation if no rules loaded
        if not self.validation_rules:
            logger.debug("No validation rules loaded, using basic validation")
            return validate_dataset_basic(sanitized_info)
        
        # Process each validation rule
        for rule in self.validation_rules:
            try:
                rule_name = rule.get('name', 'unknown')
                condition = rule.get('condition', '')
                action = rule.get('action', '').upper()
                message = rule.get('message', 'Rule triggered')
                
                # Evaluate condition
                if self._evaluate_condition(condition, sanitized_info):
                    applied_rules.append(rule_name)
                    
                    if action == 'REJECT':
                        errors.append(message)
                        logger.debug(f"Rule '{rule_name}' REJECTED: {message}")
                    elif action == 'WARN':
                        warnings.append(message)
                        logger.debug(f"Rule '{rule_name}' WARNING: {message}")
                    
            except Exception as e:
                logger.error(f"Error evaluating validation rule '{rule.get('name', 'unknown')}': {e}")
        
        validation_result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'applied_rules': applied_rules
        }
        
        if validation_result['valid']:
            logger.info(f"✅ Dataset validation passed ({len(applied_rules)} rules applied)")
        else:
            logger.warning(f"❌ Dataset validation failed with {len(errors)} errors")
        
        return validation_result
    
    
    def _debug_rule_evaluation(self, dataset_info: Dict[str, Any]):
        """Debug why rules aren't matching"""
        logger.info("🔍 DEBUG: Rule Evaluation Analysis")
        logger.info(f"   Dataset: {dataset_info.get('name', 'unknown')}")
        logger.info(f"   Frequency: {dataset_info.get('frequency', 'unknown')}")
        logger.info(f"   Columns: {dataset_info.get('columns', [])}")
        logger.info(f"   Row count: {dataset_info.get('row_count', 0)}")
        logger.info(f"   Granularity: {dataset_info.get('granularity', 'unknown')}")
        
        for rule in self.selection_rules:
            rule_name = rule.get('name', 'unknown')
            condition = rule.get('condition', '')
            try:
                result = self._evaluate_condition(condition, dataset_info)
                logger.info(f"   Rule '{rule_name}': {result} - '{condition}'")
            except Exception as e:
                logger.info(f"   Rule '{rule_name}': ERROR - {e}")

    # Update select_model to call debug when no rules match
    def select_model(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate model based on dataset characteristics - FIXED VERSION"""
        try:
            sanitized_info = self._sanitize_dataset_info(dataset_info)
            
            # Apply selection rules
            for rule_name, rule in self.selection_rules.items():
                try:
                    condition = rule.get('condition', '')
                    action = rule.get('action', '')
                    message = rule.get('message', f'Rule {rule_name} matched')
                    
                    if self._evaluate_condition(condition, sanitized_info):
                        logger.info(f"✅ Rule matched: {rule_name}")
                        
                        # Extract model name from action
                        model_name = self._extract_model_from_action(action)
                        if model_name:
                            # ✅ FIX: Use repository's _verify_model_exists method
                            if self.model_repository and hasattr(self.model_repository, '_verify_model_exists'):
                                if self.model_repository._verify_model_exists(model_name):
                                    return {
                                        'selected_model': model_name,
                                        'reason': message,
                                        'confidence': 0.9,
                                        'rule_used': rule_name
                                    }
                                else:
                                    logger.warning(f"⚠️ Model verification failed for: {model_name}")
                            else:
                                # Repository not available, assume model exists
                                logger.warning(f"⚠️ Repository not available, assuming model exists: {model_name}")
                                return {
                                    'selected_model': model_name,
                                    'reason': message,
                                    'confidence': 0.8,  # Lower confidence without verification
                                    'rule_used': rule_name
                                }
                        else:
                            logger.warning(f"⚠️ Could not extract model name from action: {action}")
                            
                except Exception as e:
                    logger.error(f"❌ Error evaluating rule '{rule_name}': {e}")
            
            # No rules matched successfully, use fallback
            logger.warning("⚠️ No selection rules matched - using fallback")
            self._debug_rule_evaluation(sanitized_info)
            return self._fallback_model_selection(sanitized_info)
            
        except Exception as e:
            logger.error(f"❌ Model selection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Emergency fallback
            return {
                'selected_model': 'Supply_Chain_Prophet_Forecaster',
                'reason': f'Selection error: {str(e)}',
                'confidence': 0.1,
                'rule_used': 'error_fallback'
            }

    def _sanitize_dataset_info(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dataset info for rule evaluation"""
        sanitized = dataset_info.copy()
        
        # Ensure all required keys exist
        sanitized.setdefault('columns', [])
        sanitized.setdefault('frequency', 'unknown')
        sanitized.setdefault('granularity', 'unknown')
        sanitized.setdefault('row_count', 0)
        sanitized.setdefault('missing_percentage', 0.0)
        
        return sanitized

    def _extract_model_from_action(self, action: str) -> Optional[str]:
        """Extract model name from rule action"""
        try:
            # Handle different action formats
            if "model_name = " in action:
                # Format: "model_name = 'Model_Name'"
                match = re.search(r"model_name\s*=\s*['\"]([^'\"]+)['\"]", action)
                if match:
                    return match.group(1)
            elif "SELECT" in action.upper():
                # Format: "SELECT model_id FROM ML_Models WHERE model_name = 'Model_Name'"
                match = re.search(r"model_name\s*=\s*['\"]([^'\"]+)['\"]", action, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # If no pattern matched, try to extract any quoted string
            match = re.search(r"['\"]([^'\"]+)['\"]", action)
            return match.group(1) if match else None
            
        except Exception as e:
            logger.error(f"❌ Failed to extract model from action '{action}': {e}")
            return None

    def _debug_rule_evaluation(self, dataset_info: Dict[str, Any]):
        """Debug why rules aren't matching"""
        logger.info("🔍 DEBUG: Rule Evaluation Analysis")
        logger.info(f"   Dataset: {dataset_info.get('filename', 'unknown')}")
        logger.info(f"   Frequency: {dataset_info.get('frequency')}")
        logger.info(f"   Columns: {dataset_info.get('columns')}")
        logger.info(f"   Row count: {dataset_info.get('row_count')}")
        logger.info(f"   Granularity: {dataset_info.get('granularity')}")
        
        for rule_name, rule in self.selection_rules.items():
            condition = rule.get('condition', '')
            result = self._evaluate_condition(condition, dataset_info)
            logger.info(f"   Rule '{rule_name}': {result} - '{condition}'")    
    
    def _fallback_model_selection(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback model selection with PROPER MODEL RESOLUTION"""
        columns = dataset_info.get('columns', [])
        frequency = dataset_info.get('frequency', '')
        row_count = dataset_info.get('row_count', 0)
        
        # Get available models from repository
        available_models = []
        if self.model_repository:
            available_models = self.model_repository.get_all_active_models()
        
        if not available_models:
            return {
                'selected_model': None,
                'confidence': 0.0,
                'reason': 'No models available in repository',
                'rule_name': 'no_models',
                'model_type': None,
                'source': 'fallback_logic'
            }
        
        # Enhanced heuristic-based selection with actual model names
        if frequency == 'monthly' and any(col in columns for col in ['shop_id', 'store_id']) and row_count >= 12:
            model_name = 'Monthly_Shop_Sales_Forecaster'
            # ✅ FIX: Use repository verification
            if self.model_repository and self.model_repository._verify_model_exists(model_name):
                return {
                    'selected_model': model_name,
                    'confidence': 0.8,
                    'reason': 'Fallback: Monthly shop data detected',
                    'rule_name': 'fallback_monthly_shop',
                    'model_type': 'lightgbm_regression',
                    'source': 'fallback_logic'
                }
        
        elif frequency == 'daily' and any(col in columns for col in ['shop_id', 'store_id']) and row_count >= 30:
            model_name = 'Daily_Shop_Sales_Forecaster'
            if self.model_repository and self.model_repository._verify_model_exists(model_name):
                return {
                    'selected_model': model_name,
                    'confidence': 0.8,
                    'reason': 'Fallback: Daily shop data detected',
                    'rule_name': 'fallback_daily_shop',
                    'model_type': 'lightgbm_regression',
                    'source': 'fallback_logic'
                }
        
        elif frequency == 'daily' and row_count >= 50:
            model_name = 'Supply_Chain_Prophet_Forecaster'
            if self.model_repository and self.model_repository._verify_model_exists(model_name):
                return {
                    'selected_model': model_name,
                    'confidence': 0.7,
                    'reason': 'Fallback: Daily time series data detected',
                    'rule_name': 'fallback_daily_ts',
                    'model_type': 'prophet_time_series',
                    'source': 'fallback_logic'
                }
        
        # Last resort: return first available model that exists
        for model in available_models:
            model_name = model['model_name']
            if self.model_repository and self.model_repository._verify_model_exists(model_name):
                return {
                    'selected_model': model_name,
                    'confidence': 0.5,
                    'reason': 'Fallback: Using first available model',
                    'rule_name': 'fallback_first_available',
                    'model_type': model.get('model_type', 'unknown'),
                    'source': 'fallback_logic'
                }
        
        # Ultimate fallback
        return {
            'selected_model': None,
            'confidence': 0.0,
            'reason': 'No valid models found in fallback',
            'rule_name': 'fallback_failed',
            'model_type': None,
            'source': 'fallback_logic'
        }
    # use shared_utils.py
    # def _sanitize_dataset_info(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """Remove any non-serializable objects from dataset info"""
    #     sanitized = dataset_info.copy()
        
    #     keys_to_remove = []
    #     for key, value in sanitized.items():
    #         if hasattr(value, '__class__') and 'DataFrame' in str(value.__class__):
    #             keys_to_remove.append(key)
    #         elif hasattr(value, 'dtypes'):
    #             keys_to_remove.append(key)
        
    #     for key in keys_to_remove:
    #         if key in sanitized:
    #             logger.debug(f"Removing non-serializable key: {key}")
    #             del sanitized[key]
        
    #     return sanitized
    
    # use shared_utils.py
    # def _basic_validation(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """Basic validation when no rules are loaded"""
    #     errors = []
    #     warnings = []
        
    #     row_count = dataset_info.get('row_count', 0)
    #     missing_percentage = dataset_info.get('missing_percentage', 0)
    #     columns = dataset_info.get('columns', [])
        
    #     if row_count < 12:
    #         errors.append("Insufficient data points (minimum 12 required)")
    #     elif row_count < 50:
    #         warnings.append("Limited data points may affect model performance")
        
    #     if missing_percentage > 0.3:
    #         errors.append("Too many missing values (>30%)")
    #     elif missing_percentage > 0.1:
    #         warnings.append("Moderate missing values may affect accuracy")
        
    #     if not any(col in columns for col in ['sales', 'demand', 'quantity']):
    #         warnings.append("No obvious target variable found")
        
    #     if 'date' not in columns:
    #         warnings.append("No date column found")
        
    #     return {
    #         'valid': len(errors) == 0,
    #         'errors': errors,
    #         'warnings': warnings,
    #         'applied_rules': ['basic_validation']
    #     }
    
    def _evaluate_condition(self, condition: str, dataset_info: Dict[str, Any]) -> bool:
        """
        Safely evaluate a rule condition
        FIXED: Properly handles dataset.get() patterns and provides safe evaluation context
        """
        if not condition:
            return False
        
        try:
            # Clean up multi-line conditions
            condition = ' '.join(condition.split())
            
            # Replace dataset.get('key') with dataset_info.get('key')
            # This handles the YAML syntax properly
            safe_condition = condition.replace("dataset.get(", "dataset_info.get(")
            
            # Create safe evaluation context
            safe_builtins = {
                'any': any, 
                'all': all, 
                'len': len,
                'str': str, 
                'int': int, 
                'float': float,
                'bool': bool, 
                'list': list, 
                'dict': dict,
                'min': min, 
                'max': max, 
                'sum': sum,
                'True': True,
                'False': False,
                'None': None
            }
            
            # Safe namespace with only dataset_info
            safe_namespace = {
                "__builtins__": safe_builtins,
                "dataset_info": dataset_info
            }
            
            # Evaluate the condition
            result = eval(safe_condition, safe_namespace)
            return bool(result)
            
        except Exception as e:
            logger.debug(f"Condition evaluation failed: '{condition}' - Error: {str(e)}")
            return False

    
    
    # use shared_utils.py
    # def _extract_model_name(self, action: str) -> Optional[str]:
    #     """Extract model name from action string"""
    #     import re
        
    #     if not action:
    #         return None
        
    #     patterns = [
    #         r"model_name\s*=\s*['\"]([A-Za-z0-9_]+)['\"]",
    #         r"['\"]([A-Z][A-Za-z0-9_]+Forecaster)['\"]",
    #         r"\b([A-Z][A-Za-z0-9_]+Forecaster)\b",
    #     ]
        
    #     for pattern in patterns:
    #         match = re.search(pattern, action, re.IGNORECASE)
    #         if match:
    #             return match.group(1)
        
    #     return None
    

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded rules"""
        return {
            'validation_rules_count': len(self.validation_rules),
            'selection_rules_count': len(self.selection_rules),
            'rules_loaded': self.rules_loaded
        }
    
    # Add this method to your RuleEngine class in rule_engine.py

    def _resolve_model_name(self, rule_model_name: str) -> Optional[str]:
        """
        Resolve rule model names to actual database model names
        UPDATED: Minimal mapping since YAML now uses exact names
        """
        
        # Legacy aliases only (YAML should use exact names going forward)
        model_name_mapping = {
            # Legacy Prophet aliases (for backward compatibility)
            'Prophet': 'Supply_Chain_Prophet_Forecaster',
            'prophet_forecaster': 'Supply_Chain_Prophet_Forecaster',
            
            # Legacy LSTM aliases
            'LSTM': 'LSTM Daily TotalRevenue Forecaster',
            'LSTM_Daily_Forecaster': 'LSTM Daily TotalRevenue Forecaster',
            
            # Legacy XGBoost alias (if old rules exist)
            'XGBoost': 'XGBoost Multi-Location NumberOfPieces Forecaster',
        }
        
        # Exact match (most common case after YAML update)
        if rule_model_name in model_name_mapping:
            resolved = model_name_mapping[rule_model_name]
            logger.info(f"🔀 Legacy name resolved: '{rule_model_name}' → '{resolved}'")
            return resolved
        
        # Assume exact database name (NEW BEHAVIOR)
        logger.debug(f"✅ Using exact model name: '{rule_model_name}'")
        return rule_model_name
    
    