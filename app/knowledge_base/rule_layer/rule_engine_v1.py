# app/knowledge_base/rule_layer/rule_engine.py
# Base Rule Engine Class - Core rule processing logic
import os
import sys
import yaml
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class RuleEngine:
    """
    Core Rule Engine - Handles validation and model selection using YAML rules
    WORKS WITH METADATA ONLY - NO DATAFRAME OBJECTS
    """
    
    def __init__(self, db_connection=None):
        """Initialize the rule engine and load YAML rules"""
        self.db_connection = db_connection
        self.validation_rules = []
        self.selection_rules = []
        self.rules_loaded = False
        
        # Load YAML rules
        self._load_validation_rules()
        self._load_selection_rules()
        
        if self.rules_loaded:
            logger.info(f"✅ RuleEngine initialized with {len(self.validation_rules)} validation rules and {len(self.selection_rules)} selection rules")
        else:
            logger.warning("⚠️ RuleEngine initialized but no YAML rules loaded")
    

    def _load_validation_rules(self):
        """Load data validation rules from YAML with better path resolution"""
        try:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            possible_paths = [
                os.path.join(current_dir, 'data_validation_rules.yaml'),
                os.path.join(current_dir, '../rule_layer/data_validation_rules.yaml'),
                os.path.join(project_root, 'app/knowledge_base/rule_layer/data_validation_rules.yaml'),
                'data_validation_rules.yaml'  # Current directory
            ]
            
            for rules_path in possible_paths:
                if os.path.exists(rules_path):
                    with open(rules_path, 'r') as f:
                        rules_data = yaml.safe_load(f)
                    self.validation_rules = rules_data.get('rules', [])
                    logger.info(f"✅ Loaded {len(self.validation_rules)} validation rules from: {rules_path}")
                    return
            
            logger.warning("❌ No validation rules YAML file found in searched paths")
            self.validation_rules = []
            
        except Exception as e:
            logger.error(f"❌ Failed to load validation rules: {e}")
            self.validation_rules = []

    def _load_selection_rules(self):
        """Load model selection rules from YAML"""
        try:
            # Try multiple possible paths
            possible_paths = [
                'app/knowledge_base/rule_layer/model_selection_rules.yaml',
                './app/knowledge_base/rule_layer/model_selection_rules.yaml',
                'knowledge_base/rule_layer/model_selection_rules.yaml',
                '../rule_layer/model_selection_rules.yaml'
            ]
            
            for rules_path in possible_paths:
                if os.path.exists(rules_path):
                    with open(rules_path, 'r') as f:
                        rules_data = yaml.safe_load(f)
                    self.selection_rules = rules_data.get('rules', [])
                    self.rules_loaded = True
                    logger.info(f"✅ Loaded {len(self.selection_rules)} selection rules from: {rules_path}")
                    return
            
            logger.warning("❌ No selection rules YAML file found")
            self.selection_rules = []
            
        except Exception as e:
            logger.error(f"❌ Failed to load selection rules: {e}")
            self.selection_rules = []
    
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
        
        # Sanitize dataset_info to ensure no DataFrame objects
        sanitized_info = self._sanitize_dataset_info(dataset_info)
        
        # Basic validation if no rules loaded
        if not self.validation_rules:
            logger.debug("No validation rules loaded, using basic validation")
            return self._basic_validation(sanitized_info)
        
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
                    elif action == 'ACCEPT':
                        logger.debug(f"Rule '{rule_name}' ACCEPTED: {message}")
                    
            except Exception as e:
                logger.error(f"Error evaluating validation rule '{rule.get('name', 'unknown')}': {e}")
                # Add error but don't break the whole validation
                errors.append(f"Rule evaluation error: {str(e)}")
        
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
    


    def _sanitize_dataset_info(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Remove any non-serializable objects from dataset info"""
        sanitized = dataset_info.copy()
        
        # Remove DataFrame objects and other non-serializable items
        keys_to_remove = []
        for key, value in sanitized.items():
            if hasattr(value, '__class__') and 'DataFrame' in str(value.__class__):
                keys_to_remove.append(key)
            elif hasattr(value, 'dtypes'):  # Likely a DataFrame
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in sanitized:
                logger.debug(f"Removing non-serializable key from dataset_info: {key}")
                del sanitized[key]
        
        return sanitized
    
    def _basic_validation(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation when no rules are loaded"""
        errors = []
        warnings = []
        
        row_count = dataset_info.get('row_count', 0)
        missing_percentage = dataset_info.get('missing_percentage', 0)
        columns = dataset_info.get('columns', [])
        
        if row_count < 12:
            errors.append("Insufficient data points (minimum 12 required)")
        elif row_count < 50:
            warnings.append("Limited data points may affect model performance")
        
        if missing_percentage > 0.3:
            errors.append("Too many missing values (>30%)")
        elif missing_percentage > 0.1:
            warnings.append("Moderate missing values may affect accuracy")
        
        # Check for required columns
        if not any(col in columns for col in ['sales', 'demand', 'quantity']):
            warnings.append("No obvious target variable found (sales, demand, quantity)")
        
        if 'date' not in columns:
            warnings.append("No date column found - time series models may not work")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'applied_rules': ['basic_validation']
        }
    # !!!!!
    # def select_model(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Select appropriate model based on selection rules - METADATA ONLY
        
    #     Args:
    #         dataset_info: Dictionary containing dataset METADATA (no DataFrame)
            
    #     Returns:
    #         Dictionary with selected model information
    #     """
    #     # Sanitize dataset_info first
    #     sanitized_info = self._sanitize_dataset_info(dataset_info)
        
    #     # If no rules loaded, use fallback logic
    #     if not self.selection_rules:
    #         return self._fallback_model_selection(sanitized_info)
        
    #     # Process rules by priority (highest first)
    #     sorted_rules = sorted(
    #         self.selection_rules,
    #         key=lambda x: x.get('priority', 0),
    #         reverse=True
    #     )
        
    #     for rule in sorted_rules:
    #         try:
    #             rule_name = rule.get('name', 'unknown')
    #             condition = rule.get('condition', '')
    #             action = rule.get('action', '')
    #             message = rule.get('message', 'Rule matched')
    #             priority = rule.get('priority', 0)
                
    #             # Evaluate condition
    #             if self._evaluate_condition(condition, sanitized_info):
    #                 # Extract model name from action
    #                 model_name = self._extract_model_name(action)
                    
    #                 if model_name:
    #                     logger.info(f"✅ Rule '{rule_name}' matched - selecting {model_name}")
    #                     return {
    #                         'selected_model': model_name,
    #                         'confidence': min(priority / 10.0, 1.0),  # Normalize priority to 0-1
    #                         'reason': message,
    #                         'rule_name': rule_name,
    #                         'model_type': self._infer_model_type(model_name),
    #                         'source': 'rule_engine'
    #                     }
                    
    #         except Exception as e:
    #             logger.error(f"Error evaluating selection rule '{rule.get('name', 'unknown')}': {e}")
    #             # Continue to next rule instead of breaking
        
    #     # No rules matched
    #     logger.warning("⚠️ No selection rules matched - using fallback")
    #     return self._fallback_model_selection(sanitized_info)
    

    # ==================================================
    # !!!! Problem: RuleEngine imports KnowledgeBaseService, but KnowledgeBaseService likely uses RuleEngine. This creates a circular import.
    # ==================================================
    # def _get_available_models(self) -> List[Dict[str, Any]]:
    #     """Get available models using actual dataset with RuleEngine selection"""
    #     try:
    #         # Get all models from knowledge base
    #         from app.services.knowledge_base_services.core.knowledge_base_service import SupplyChainService as KnowledgeBaseService
    #         kb_service = KnowledgeBaseService("supply_chain.db")
    #         all_models = kb_service.get_all_models()
    #         kb_service.close()
            
    #         if not all_models:
    #             logger.warning("No models found in knowledge base")
    #             return []  # Return empty list - let calling function handle fallback
            
    #         # Filter to active models only
    #         active_models = [model for model in all_models if model.get('is_active', True)]
            
    #         if not active_models:
    #             logger.warning("No active models found")
    #             return []
            
    #         # We'll let the calling function provide the actual dataset context
    #         # This method just returns all active models for evaluation
    #         logger.info(f"📊 Found {len(active_models)} active models for evaluation")
    #         return active_models
            
    #     except Exception as e:
    #         logger.error(f"❌ Error getting available models: {e}")
    #         return []  # Empty list - let calling function handle fallback
    
    # !!!! need to be removed 
    

    def _evaluate_condition(self, condition: str, dataset_info: Dict[str, Any]) -> bool:
        """
        Safely evaluate a rule condition - METADATA ONLY
        
        Args:
            condition: Python expression as string
            dataset_info: Dataset METADATA dictionary (no DataFrame)
            
        Returns:
            Boolean result of condition evaluation
        """
        if not condition:
            return False
        
        try:
            # Create a safe evaluation context
            # Replace 'dataset.' with 'dataset_info.get('
            safe_condition = condition.replace('dataset.', "dataset_info.get('")
            
            # Handle common patterns for metadata access
            safe_condition = safe_condition.replace("'columns'", "', [])")  # dataset.columns -> dataset_info.get('columns', [])
            safe_condition = safe_condition.replace("'row_count'", "', 0)")  # dataset.row_count -> dataset_info.get('row_count', 0)
            safe_condition = safe_condition.replace("'frequency'", "', '')")  # dataset.frequency -> dataset_info.get('frequency', '')
            safe_condition = safe_condition.replace("'granularity'", "', '')")  # dataset.granularity -> dataset_info.get('granularity', '')
            safe_condition = safe_condition.replace("'missing_percentage'", "', 0)")  # dataset.missing_percentage -> dataset_info.get('missing_percentage', 0)
            safe_condition = safe_condition.replace("'name'", "', '')")  # dataset.name -> dataset_info.get('name', '')
            
            # For any remaining patterns like dataset.some_attribute
            import re
            safe_condition = re.sub(
                r"dataset_info\.get\('(\w+)'(?!\))",
                r"dataset_info.get('\1', None)",
                safe_condition
            )
            
            # Add closing parentheses for get calls
            safe_condition = re.sub(
                r"dataset_info\.get\('[^']+'(?![)])",
                lambda m: m.group(0) + ")",
                safe_condition
            )
            
            # Safe evaluation with restricted namespace
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
                'sum': sum
            }
            
            result = eval(safe_condition, {"__builtins__": safe_builtins}, {"dataset_info": dataset_info})
            return bool(result)
            
        except Exception as e:
            logger.debug(f"Condition evaluation failed: '{condition}' - Error: {e}")
            return False
    

    def _extract_model_name(self, action: str) -> Optional[str]:
        """Extract model name from action string - IMPROVED VERSION"""
        import re
        
        if not action:
            return None
            
        # Common patterns in order of specificity
        patterns = [
            r"model_name\s*=\s*['\"]([A-Za-z0-9_]+)['\"]",  # model_name = 'ModelName'
            r"model_name\s*=\s*([A-Za-z0-9_]+)\b",           # model_name = ModelName
            r"SELECT.*WHERE.*model_name\s*=\s*['\"]([A-Za-z0-9_]+)['\"]",  # SQL pattern
            r"['\"]([A-Z][A-Za-z0-9_]+Forecaster)['\"]",     # 'MonthlyShopForecaster'
            r"['\"]([A-Z][A-Za-z0-9_]+Model)['\"]",          # 'TestAutoModel'
            r"\b([A-Z][A-Za-z0-9_]+Forecaster)\b",           # MonthlyShopForecaster
            r"\b([A-Z][A-Za-z0-9_]+Model)\b",                # TestAutoModel
            r"\b(Prophet|XGBoost|LightGBM|TFT)\b"      # Common model names
        ]
        
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                model_name = match.group(1)
                logger.debug(f"Extracted model name '{model_name}' from action: {action}")
                return model_name
        
        logger.warning(f"Could not extract model name from action: {action}")
        return None


    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name"""
        if not model_name:
            return 'unknown'
            
        name_lower = model_name.lower()
        
        if 'prophet' in name_lower:
            return 'time_series'
        elif 'lightgbm' in name_lower or 'lgb' in name_lower:
            return 'ensemble'
        elif 'xgboost' in name_lower or 'xgb' in name_lower:
            return 'ensemble'
        elif 'tft' in name_lower or 'temporal' in name_lower:
            return 'time_series'
        elif 'lstm' in name_lower or 'rnn' in name_lower:
            return 'neural_network'
        elif 'auto' in name_lower:
            return 'auto_ml'
        elif 'forecaster' in name_lower:
            return 'ensemble'
        else:
            return 'unknown'
    
    # def _fallback_model_selection(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Fallback model selection logic when no rules match - METADATA ONLY
    #     """
    #     columns = dataset_info.get('columns', [])
    #     frequency = dataset_info.get('frequency', '')
    #     row_count = dataset_info.get('row_count', 0)
    #     granularity = dataset_info.get('granularity', '')
        
    #     # Simple heuristic-based selection
    #     if frequency == 'monthly' and any(col in columns for col in ['shop_id', 'store_id']) and row_count >= 12:
    #         return {
    #             'selected_model': 'Monthly_Shop_Sales_Forecaster',
    #             'confidence': 0.7,
    #             'reason': 'Fallback: Monthly shop data detected',
    #             'rule_name': 'fallback_monthly_shop',
    #             'model_type': 'ensemble',
    #             'source': 'fallback_logic'
    #         }
    #     elif frequency == 'daily' and any(col in columns for col in ['shop_id', 'store_id']) and row_count >= 30:
    #         return {
    #             'selected_model': 'Daily_Shop_Sales_Forecaster',
    #             'confidence': 0.7,
    #             'reason': 'Fallback: Daily shop data detected',
    #             'rule_name': 'fallback_daily_shop',
    #             'model_type': 'ensemble',
    #             'source': 'fallback_logic'
    #         }
    #     elif frequency == 'daily' and row_count >= 50:
    #         return {
    #             'selected_model': 'Prophet',
    #             'confidence': 0.6,
    #             'reason': 'Fallback: Daily time series data detected',
    #             'rule_name': 'fallback_daily_ts',
    #             'model_type': 'time_series',
    #             'source': 'fallback_logic'
    #         }
    #     elif 'date' in columns and any(col in columns for col in ['demand', 'sales', 'quantity']) and row_count >= 30:
    #         return {
    #             'selected_model': 'Prophet',
    #             'confidence': 0.6,
    #             'reason': 'Fallback: Time series data with target detected',
    #             'rule_name': 'fallback_time_series',
    #             'model_type': 'time_series',
    #             'source': 'fallback_logic'
    #         }
    #     else:
    #         return {
    #             'selected_model': None,
    #             'confidence': 0.0,
    #             'reason': 'No suitable model found with available metadata',
    #             'rule_name': 'no_match',
    #             'model_type': None,
    #             'source': 'fallback_logic'
    #         }
    

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded rules"""
        return {
            'validation_rules_count': len(self.validation_rules),
            'selection_rules_count': len(self.selection_rules),
            'rules_loaded': self.rules_loaded,
            'validation_rule_types': self._count_rule_types(self.validation_rules),
            'selection_rule_priorities': self._count_priorities(self.selection_rules)
        }
    
    # def test_rules_with_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Test all rules against a dataset and see which ones fire
    #     Excellent for debugging and rule development
    #     """
    #     sanitized_info = self._sanitize_dataset_info(dataset_info)
        
    #     results = {
    #         'validation_rules_fired': [],
    #         'selection_rules_fired': [],
    #         'validation_result': None,
    #         'selection_result': None
    #     }
        
    #     # Test validation rules
    #     for rule in self.validation_rules:
    #         try:
    #             condition = rule.get('condition', '')
    #             if self._evaluate_condition(condition, sanitized_info):
    #                 results['validation_rules_fired'].append({
    #                     'name': rule.get('name'),
    #                     'action': rule.get('action'),
    #                     'message': rule.get('message'),
    #                     'priority': rule.get('priority')
    #                 })
    #         except Exception as e:
    #             logger.error(f"Error testing validation rule {rule.get('name')}: {e}")
        
    #     # Test selection rules  
    #     for rule in self.selection_rules:
    #         try:
    #             condition = rule.get('condition', '')
    #             if self._evaluate_condition(condition, sanitized_info):
    #                 results['selection_rules_fired'].append({
    #                     'name': rule.get('name'),
    #                     'action': rule.get('action'),
    #                     'message': rule.get('message'),
    #                     'priority': rule.get('priority')
    #                 })
    #         except Exception as e:
    #             logger.error(f"Error testing selection rule {rule.get('name')}: {e}")
        
    #     # Get actual results
    #     results['validation_result'] = self.validate_dataset(dataset_info)
    #     results['selection_result'] = self.select_model(dataset_info)
        
    #     return results


    def _count_rule_types(self, rules: List[Dict]) -> Dict[str, int]:
        """Count rules by type"""
        types = {}
        for rule in rules:
            action = rule.get('action', 'unknown').upper()
            types[action] = types.get(action, 0) + 1
        return types
    
    def _count_priorities(self, rules: List[Dict]) -> Dict[int, int]:
        """Count rules by priority"""
        priorities = {}
        for rule in rules:
            priority = rule.get('priority', 0)
            priorities[priority] = priorities.get(priority, 0) + 1
        return priorities
    
    def test_condition_evaluation(self, condition: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a condition with dataset info for debugging
        """
        try:
            result = self._evaluate_condition(condition, dataset_info)
            return {
                'success': True,
                'result': result,
                'condition': condition,
                'dataset_keys': list(dataset_info.keys())
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'condition': condition,
                'dataset_keys': list(dataset_info.keys())
            }


def demo_rule_engine():
    """Demo the base RuleEngine"""
    print("🚀 BASE RULE ENGINE DEMO")
    print("=" * 60)
    
    # Initialize engine
    engine = RuleEngine()
    
    # Show statistics
    stats = engine.get_rule_statistics()
    print(f"\n📊 RULE STATISTICS:")
    print(f"   Validation Rules: {stats['validation_rules_count']}")
    print(f"   Selection Rules: {stats['selection_rules_count']}")
    print(f"   Rules Loaded: {stats['rules_loaded']}")
    
    # Test validation
    print(f"\n✅ VALIDATION TEST:")
    test_dataset = {
        'name': 'test_data',
        'frequency': 'monthly',
        'row_count': 48,
        'columns': ['shop_id', 'date', 'sales'],
        'missing_percentage': 0.02,
        'granularity': 'shop_level'
    }
    
    validation_result = engine.validate_dataset(test_dataset)
    print(f"   Valid: {validation_result['valid']}")
    print(f"   Errors: {len(validation_result['errors'])}")
    if validation_result['errors']:
        for error in validation_result['errors']:
            print(f"     - {error}")
    print(f"   Warnings: {len(validation_result['warnings'])}")
    if validation_result['warnings']:
        for warning in validation_result['warnings']:
            print(f"     - {warning}")
    print(f"   Rules Applied: {len(validation_result['applied_rules'])}")
    
    # Test model selection
    print(f"\n🎯 MODEL SELECTION TEST:")
    selection_result = engine.select_model(test_dataset)
    print(f"   Selected Model: {selection_result['selected_model']}")
    print(f"   Confidence: {selection_result['confidence']:.1%}")
    print(f"   Reason: {selection_result['reason']}")
    print(f"   Rule: {selection_result['rule_name']}")
    print(f"   Model Type: {selection_result['model_type']}")
    print(f"   Source: {selection_result['source']}")
    
    # Test condition evaluation
    print(f"\n🧪 CONDITION EVALUATION TEST:")
    test_conditions = [
        "dataset.row_count > 10",
        "'sales' in dataset.columns",
        "dataset.frequency == 'monthly'"
    ]
    
    for condition in test_conditions:
        test_result = engine.test_condition_evaluation(condition, test_dataset)
        print(f"   Condition: '{condition}'")
        print(f"     Success: {test_result['success']}")
        if test_result['success']:
            print(f"     Result: {test_result['result']}")
        else:
            print(f"     Error: {test_result['error']}")
    
    print("\n🎉 BASE RULE ENGINE DEMO COMPLETED!")


if __name__ == "__main__":
    demo_rule_engine()