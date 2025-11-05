# app/services/knowledge_base_services/core/rule_manager.py
import logging
from typing import Dict, Any
from .rule_generator_service import RuleGeneratorService

logger = logging.getLogger(__name__)

class RuleManager:
    """
    Hybrid rule management - combines manual YAML rules with auto-generation
    """
    
    def __init__(self):
        self.generator = RuleGeneratorService()
        self.manual_rules_file = 'app/knowledge_base/rule_layer/model_selection_rules.yaml'
    
    def add_model_with_rules(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new model and automatically generate appropriate rules
        """
        logger.info(f"🔄 Adding model with auto-generated rules: {model_info['model_name']}")
        
        # Step 1: Check if model already has rules
        existing_rules = self.generator.list_generated_rules(model_info['model_name'])
        if existing_rules:
            logger.warning(f"⚠️ Model {model_info['model_name']} already has {len(existing_rules)} rules")
            return {
                'success': False,
                'message': f"Model {model_info['model_name']} already has existing rules",
                'existing_rules': existing_rules
            }
        
        # Step 2: Generate rules automatically
        generation_result = self.generator.generate_rules_for_model(model_info)
        
        if generation_result['success']:
            logger.info(f"✅ Successfully added {model_info['model_name']} with {len(generation_result['generated_rules'])} rules")
            
            # Step 3: Verify rules work with current dataset
            verification = self.verify_rules_with_sample_data(model_info, generation_result['generated_rules'])
            
            return {
                'success': True,
                'model_name': model_info['model_name'],
                'generated_rules': generation_result['generated_rules'],
                'verification': verification,
                'backup_created': generation_result['backup_created'],
                'optimization_stats': generation_result['optimization_stats']
            }
        else:
            logger.error(f"❌ Failed to generate rules for {model_info['model_name']}")
            return generation_result
    
    def validate_model_compatibility(self, model_name: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if a model can work with given dataset"""
        model = self.load_model(model_name)
        if not model:
            return {'compatible': False, 'error': 'Model not found'}
        
        # Check required features
        required_features = self._get_required_features(model_name)
        available_features = set(dataset_info.get('columns', []))
        
        missing_features = [f for f in required_features if f not in available_features]
        
        return {
            'compatible': len(missing_features) == 0,
            'missing_features': missing_features,
            'model_type': getattr(model, 'model_type', 'unknown'),
            'available_features': list(available_features)
        }

    def verify_rules_with_sample_data(self, model_info: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify generated rules work with typical dataset patterns
        """
        # Common dataset patterns for testing
        test_datasets = [
            {
                'name': 'retail_daily',
                'frequency': 'daily',
                'columns': ['date', 'sales', 'shop_id', 'demand'],
                'row_count': 100,
                'granularity': 'shop_level'
            },
            {
                'name': 'retail_monthly', 
                'frequency': 'monthly',
                'columns': ['date', 'sales', 'shop_id', 'region'],
                'row_count': 36,
                'granularity': 'shop_level'
            },
            {
                'name': 'simple_ts',
                'frequency': 'daily',
                'columns': ['date', 'demand'],
                'row_count': 50,
                'granularity': 'aggregate'
            }
        ]
        
        verification_results = {
            'matching_datasets': [],
            'non_matching_datasets': [],
            'rule_coverage': 0
        }
        
        for dataset in test_datasets:
            matches_any_rule = False
            for rule in rules:
                try:
                    # Test if rule condition matches this dataset
                    condition = rule['condition']
                    # Create safe evaluation context
                    safe_globals = {'dataset': dataset, 'len': len, 'any': any, 'all': all}
                    result = eval(condition, {"__builtins__": {}}, safe_globals)
                    
                    if result:
                        matches_any_rule = True
                        break
                except Exception as e:
                    logger.debug(f"Rule evaluation failed for {rule['name']}: {e}")
            
            if matches_any_rule:
                verification_results['matching_datasets'].append(dataset['name'])
            else:
                verification_results['non_matching_datasets'].append(dataset['name'])
        
        # Calculate coverage percentage
        total_datasets = len(test_datasets)
        matching_datasets = len(verification_results['matching_datasets'])
        verification_results['rule_coverage'] = (matching_datasets / total_datasets) * 100 if total_datasets > 0 else 0
        
        return verification_results
    
    def list_all_models_with_rules(self) -> Dict[str, Any]:
        """
        List all models that have rules defined
        """
        try:
            rules = self.generator.list_generated_rules()
            models = {}
            
            for rule in rules:
                # Extract model name from rule name (first part before underscore)
                model_name = rule['name'].split('_')[0]
                if model_name not in models:
                    models[model_name] = {
                        'rule_count': 0,
                        'highest_priority': 0,
                        'rules': []
                    }
                
                models[model_name]['rule_count'] += 1
                models[model_name]['highest_priority'] = max(
                    models[model_name]['highest_priority'], 
                    rule['priority']
                )
                models[model_name]['rules'].append(rule['name'])
            
            return {
                'total_models': len(models),
                'total_rules': len(rules),
                'models': models
            }
        except Exception as e:
            logger.error(f"Error listing models with rules: {e}")
            return {'total_models': 0, 'total_rules': 0, 'models': {}}
    
    def get_rule_statistics(self) -> Dict[str, Any]:

        """
        Get comprehensive statistics about rules
        """
        rules = self.generator.list_generated_rules()
        
        stats = {
            'total_rules': len(rules),
            'priority_distribution': {},
            'model_coverage': {},
            'condition_complexity': {
                'simple': 0,    # < 50 chars
                'medium': 0,    # 50-150 chars  
                'complex': 0    # > 150 chars
            }
        }
        
        for rule in rules:
            # Priority distribution
            priority = rule['priority']
            stats['priority_distribution'][priority] = stats['priority_distribution'].get(priority, 0) + 1
            
            # Model coverage
            model_name = rule['name'].split('_')[0]
            stats['model_coverage'][model_name] = stats['model_coverage'].get(model_name, 0) + 1
            
            # Condition complexity
            cond_length = len(rule['condition'])
            if cond_length < 50:
                stats['condition_complexity']['simple'] += 1
            elif cond_length < 150:
                stats['condition_complexity']['medium'] += 1
            else:
                stats['condition_complexity']['complex'] += 1
        
        return stats
    
    def record_prediction_accuracy(self, model_name: str, actual: List, predicted: List):
        """Record prediction accuracy for model performance tracking"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Store in database
        self.db.conn.execute("""
            INSERT INTO Model_Performance_History 
            (model_id, metrics, sample_size, training_date)
            VALUES (?, ?, ?, DATE('now'))
        """, (
            self._get_model_id(model_name),
            json.dumps({'MAE': mae, 'RMSE': rmse, 'MAPE': mape}),
            len(actual)
        ))
        self.db.conn.commit()