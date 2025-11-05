# app/services/knowledge_base_services/rule_generator_service.py
import yaml
import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RuleTemplates:
    """Pre-defined rule templates for common patterns - UPDATED FOR SIMPLE ACTIONS"""
    
    TEMPLATES = {
        'feature_based': {
            'name': "{model_name}_feature_match",
            'description': "Select {model_name} when required features are available",
            'condition': " and ".join(["'{{feature}}' in dataset.get('columns', [])" for feature in "{features}"]),
            'action': "model_name = '{model_name}'",  # SIMPLE ACTION
            'priority': 9,
            'message': "{model_name} selected - all required features available"
        },
        'frequency_based': {
            'name': "{model_name}_{frequency}_data",
            'description': "Select {model_name} for {frequency} data",
            'condition': "dataset.get('frequency') == '{frequency}'",
            'action': "model_name = '{model_name}'",  # SIMPLE ACTION
            'priority': 8,
            'message': "{model_name} selected - optimized for {frequency} data"
        },
        'data_quality_based': {
            'name': "{model_name}_quality_data",
            'description': "Select {model_name} for quality data",
            'condition': "dataset.get('row_count', 0) >= {min_rows} and dataset.get('missing_percentage', 0) <= {max_missing}",
            'action': "model_name = '{model_name}'",  # SIMPLE ACTION
            'priority': 7,
            'message': "{model_name} selected - data quality requirements met"
        },
        'target_based': {
            'name': "{model_name}_{target_variable}_target",
            'description': "Select {model_name} for {target_variable} forecasting",
            'condition': "'{target_variable}' in dataset.get('columns', []) and dataset.get('row_count', 0) >= {min_rows}",
            'action': "model_name = '{model_name}'",  # SIMPLE ACTION
            'priority': 8,
            'message': "{model_name} selected - optimized for {target_variable} forecasting"
        },
        'fallback': {
            'name': "{model_name}_fallback",
            'description': "Select {model_name} as fallback option",
            'condition': "dataset.get('row_count', 0) >= 20 and len(dataset.get('columns', [])) >= 2",
            'action': "model_name = '{model_name}'",  # SIMPLE ACTION
            'priority': 3,
            'message': "{model_name} selected as fallback option"
        }
    }
    
    @classmethod
    def apply_template(cls, template_name: str, **kwargs) -> Dict[str, Any]:
        """Apply a template with provided parameters"""
        template = cls.TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
        
        rule = template.copy()
        for key, value in rule.items():
            if isinstance(value, str):
                # Handle list formatting for features
                if key == 'condition' and '{features}' in value:
                    features = kwargs.get('features', [])
                    feature_conditions = " and ".join([f"'{feature}' in dataset.get('columns', [])" for feature in features])
                    rule[key] = value.replace(" and ".join(["'{feature}' in dataset.get('columns', [])" for feature in "{features}"]), feature_conditions)
                else:
                    rule[key] = value.format(**kwargs)
        
        return rule
    
    @classmethod
    def get_available_templates(cls) -> List[str]:
        """Get list of available template names"""
        return list(cls.TEMPLATES.keys())


class RuleGeneratorService:
    """
    Enhanced rule generator with validation, optimization, and version control
    UPDATED FOR SIMPLE ACTION FORMAT
    """
    
    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        self.rules_file = 'app/knowledge_base/rule_layer/model_selection_rules.yaml'
        self.backup_dir = 'app/knowledge_base/rule_layer/backups/'
        self._ensure_rules_file_exists()
        self._ensure_backup_dir_exists()
    
    def _ensure_rules_file_exists(self):
        """Ensure the rules YAML file exists"""
        os.makedirs(os.path.dirname(self.rules_file), exist_ok=True)
        if not os.path.exists(self.rules_file):
            # Create initial structure
            initial_rules = {
                'version': '2.1',
                'description': 'Auto-generated model selection rules with validation',
                'last_updated': datetime.now().isoformat(),
                'rules': []
            }
            with open(self.rules_file, 'w') as f:
                yaml.dump(initial_rules, f, default_flow_style=False, indent=2)
            logger.info(f"📁 Created new rules file: {self.rules_file}")
    
    def _ensure_backup_dir_exists(self):
        """Ensure backup directory exists"""
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def generate_rules_for_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically generate rules for a newly uploaded model with validation
        """
        logger.info(f"🤖 Generating rules for model: {model_info['model_name']}")
        
        # Create backup before generation
        backup_path = self.backup_rules_before_generation()
        
        try:
            model_name = model_info['model_name']
            model_type = model_info.get('model_type', 'unknown')
            required_features = model_info.get('required_features', [])
            target_variable = model_info.get('target_variable', 'unknown')
            
            # Generate rules based on model characteristics
            rules = self._generate_model_selection_rules(
                model_name, model_type, required_features, target_variable
            )
            
            # Validate generated rules
            validation_results = self.validate_generated_rules(rules)
            
            if validation_results['invalid']:
                logger.warning(f"⚠️ {len(validation_results['invalid'])} invalid rules detected")
                # Rollback if there are invalid rules
                if backup_path:
                    self.rollback_to_backup(backup_path)
                return {
                    'success': False,
                    'model_name': model_name,
                    'validation_results': validation_results,
                    'error': 'Invalid rules generated, rollback performed'
                }
            
            # Optimize rules
            optimized_rules = self.optimize_generated_rules(rules)
            logger.info(f"🔧 Optimized {len(rules)} rules to {len(optimized_rules)} rules")
            
            # Add to YAML file
            added_count = self._add_rules_to_yaml(optimized_rules)
            
            # Add to database if connection available
            db_count = 0
            if self.db_connection:
                db_count = self._add_rules_to_database(optimized_rules)
            
            logger.info(f"✅ Generated {len(optimized_rules)} rules for {model_name} (YAML: {added_count}, DB: {db_count})")
            
            return {
                'success': True,
                'model_name': model_name,
                'generated_rules': optimized_rules,
                'validation_results': validation_results,
                'optimization_stats': {
                    'original_rules': len(rules),
                    'optimized_rules': len(optimized_rules),
                    'reduction_percentage': ((len(rules) - len(optimized_rules)) / len(rules)) * 100
                },
                'yaml_added': added_count,
                'database_added': db_count,
                'backup_created': backup_path
            }
            
        except Exception as e:
            logger.error(f"❌ Rule generation failed: {e}")
            # Rollback on error
            if backup_path:
                self.rollback_to_backup(backup_path)
            raise
    
    def validate_generated_rules(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate generated rules for syntax and logic errors"""
        validation_results = {
            'valid': [],
            'invalid': [],
            'warnings': [],
            'summary': {
                'total_rules': len(rules),
                'valid_count': 0,
                'invalid_count': 0,
                'warning_count': 0
            }
        }
        
        for rule in rules:
            try:
                # Test condition syntax
                self._test_condition_syntax(rule['condition'])
                
                # Test action validity
                self._test_action_validity(rule['action'])
                
                # Check for common issues
                warnings = self._check_rule_warnings(rule)
                
                if warnings:
                    validation_results['warnings'].append({
                        'rule': rule['name'], 
                        'warnings': warnings
                    })
                    validation_results['summary']['warning_count'] += len(warnings)
                
                validation_results['valid'].append(rule['name'])
                validation_results['summary']['valid_count'] += 1
                
            except Exception as e:
                validation_results['invalid'].append({
                    'rule': rule['name'], 
                    'error': str(e),
                    'rule_data': rule
                })
                validation_results['summary']['invalid_count'] += 1
        
        return validation_results
    
    def _test_condition_syntax(self, condition: str) -> bool:
        """Test if condition syntax is valid"""
        # Create a mock dataset for testing
        test_dataset = {
            'columns': ['date', 'sales', 'shop_id', 'demand', 'value', 'target', 'quantity', 'revenue', 'volume'],
            'row_count': 100,
            'frequency': 'monthly',
            'granularity': 'shop_level',
            'missing_percentage': 0.1,
            'has_irregular_intervals': False
        }
        
        try:
            # Safe evaluation test
            safe_globals = {
                'dataset': test_dataset, 
                'len': len,
                'any': any,
                'all': all
            }
            safe_locals = {}
            
            # Clean condition for safe evaluation
            clean_condition = condition.replace('dataset.', '')
            clean_condition = re.sub(r"dataset\.get\('([^']+)',\s*[^)]+\)", r"dataset.get('\1', None)", condition)
            
            result = eval(clean_condition, {"__builtins__": {}}, {**safe_globals, **safe_locals})
            return isinstance(result, bool)
        except Exception as e:
            raise ValueError(f"Invalid condition syntax: {e}")
    
    def _test_action_validity(self, action: str) -> bool:
        """Test if action syntax is valid - UPDATED FOR SIMPLE FORMAT"""
        # Check for simple model_name assignment format
        if not action.strip().startswith('model_name = '):
            raise ValueError("Action must use format: model_name = 'ModelName'")
        
        # Check for quoted model name
        if "'" not in action and '"' not in action:
            raise ValueError("Model name must be quoted in action")
        
        return True
    
    def _check_rule_warnings(self, rule: Dict[str, Any]) -> List[str]:
        """Check for common rule issues and return warnings"""
        warnings = []
        
        # Check for overly complex conditions
        condition_length = len(rule['condition'])
        if condition_length > 200:
            warnings.append(f"Condition is very long ({condition_length} chars), consider simplifying")
        
        # Check for redundant conditions
        if 'dataset.row_count >= 20' in rule['condition'] and rule['priority'] > 5:
            warnings.append("Low row count requirement for high-priority rule")
        
        # Check priority consistency
        if rule['priority'] > 10 or rule['priority'] < 1:
            warnings.append(f"Priority {rule['priority']} outside recommended range 1-10")
        
        # Check for proper .get() usage in conditions
        if 'dataset.columns' in rule['condition'] and 'dataset.get(' not in rule['condition']:
            warnings.append("Consider using dataset.get('columns', []) instead of dataset.columns")
        
        return warnings
    
    def optimize_generated_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple rule optimization - remove duplicates and sort by priority"""
        if not rules:
            return []
        
        # Step 1: Remove exact duplicates (same condition + action)
        unique_rules = []
        seen_conditions = set()
        
        for rule in rules:
            # Create a signature based on condition and action
            rule_signature = (rule['condition'], rule['action'])
            
            if rule_signature not in seen_conditions:
                seen_conditions.add(rule_signature)
                unique_rules.append(rule)
            else:
                logger.debug(f"🔧 Removed duplicate rule: {rule['name']}")
        
        # Step 2: Sort by priority (highest first)
        unique_rules.sort(key=lambda x: x['priority'], reverse=True)
        
        # Step 3: Simple conflict resolution - keep highest priority per model
        final_rules = []
        model_priority = {}  # Track highest priority rule for each model
        
        for rule in unique_rules:
            model_name = rule['name'].split('_')[0]  # Extract model name
            
            # If we haven't seen this model, or this rule has higher priority
            if model_name not in model_priority or rule['priority'] > model_priority[model_name]:
                model_priority[model_name] = rule['priority']
                final_rules.append(rule)
            else:
                logger.debug(f"🔧 Skipped lower priority rule: {rule['name']}")
        
        logger.info(f"🔧 Simplified optimization: {len(rules)} → {len(final_rules)} rules")
        return final_rules


    # def optimize_generated_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """Optimize rules by removing redundancies and improving performance"""
    #     if not rules:
    #         return []
        
    #     optimized_rules = []
        
    #     # Group by model and priority
    #     rule_groups = {}
    #     for rule in rules:
    #         model_prefix = rule['name'].split('_')[0]  # Extract model name prefix
    #         key = (model_prefix, rule['priority'])
    #         if key not in rule_groups:
    #             rule_groups[key] = []
    #         rule_groups[key].append(rule)
        
    #     # Optimize each group
    #     for group_rules in rule_groups.values():
    #         if len(group_rules) > 1:
    #             # Check if rules are similar enough to merge
    #             if self._are_rules_similar(group_rules):
    #                 merged_rule = self._merge_similar_rules(group_rules)
    #                 optimized_rules.append(merged_rule)
    #                 logger.debug(f"🔧 Merged {len(group_rules)} rules into: {merged_rule['name']}")
    #             else:
    #                 # Keep rules separate but remove duplicates
    #                 unique_rules = self._remove_duplicate_rules(group_rules)
    #                 optimized_rules.extend(unique_rules)
    #         else:
    #             optimized_rules.extend(group_rules)
        
    #     # Sort by priority (highest first)
    #     optimized_rules.sort(key=lambda x: x['priority'], reverse=True)
        
    #     # Remove any rules that are completely overshadowed by higher priority rules
    #     final_rules = self._remove_overshadowed_rules(optimized_rules)
        
    #     logger.info(f"🔧 Optimization: {len(rules)} → {len(final_rules)} rules")
    #     return final_rules
    


    def _are_rules_similar(self, rules: List[Dict[str, Any]]) -> bool:
        """Check if rules are similar enough to merge"""
        if len(rules) < 2:
            return False
        
        base_rule = rules[0]
        
        # Check if actions are identical
        base_action = base_rule['action']
        if not all(rule['action'] == base_action for rule in rules[1:]):
            return False
        
        # Check if priorities are identical
        base_priority = base_rule['priority']
        if not all(rule['priority'] == base_priority for rule in rules[1:]):
            return False
        
        return True
    
    # def _merge_similar_rules(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """Merge similar rules into a single optimized rule"""
    #     base_rule = rules[0].copy()
        
    #     # Combine conditions with OR logic
    #     conditions = [rule['condition'] for rule in rules]
    #     base_rule['condition'] = f"({' or '.join([f'({cond})' for cond in conditions])})"
        
    #     # Update metadata
    #     base_rule['name'] = f"{base_rule['name'].split('_')[0]}_merged_{len(rules)}_scenarios"
    #     base_rule['description'] = f"Merged rule covering {len(rules)} similar scenarios"
    #     base_rule['message'] = f"{base_rule['message'].split(' - ')[0]} - multiple scenarios matched"
        
    #     return base_rule
    


    def _remove_duplicate_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate rules based on condition and action"""
        seen = set()
        unique_rules = []
        
        for rule in rules:
            rule_signature = (rule['condition'], rule['action'])
            if rule_signature not in seen:
                seen.add(rule_signature)
                unique_rules.append(rule)
        
        return unique_rules
    
    # def _remove_overshadowed_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """Remove rules that are completely overshadowed by higher priority rules"""
    #     if not rules:
    #         return []
        
    #     # Group by model
    #     model_rules = {}
    #     for rule in rules:
    #         model_name = rule['name'].split('_')[0]
    #         if model_name not in model_rules:
    #             model_rules[model_name] = []
    #         model_rules[model_name].append(rule)
        
    #     final_rules = []
    #     for model_name, model_rule_list in model_rules.items():
    #         # Keep highest priority rule for each model if conditions are similar
    #         if len(model_rule_list) > 1:
    #             highest_priority = max(rule['priority'] for rule in model_rule_list)
    #             high_priority_rules = [r for r in model_rule_list if r['priority'] == highest_priority]
    #             final_rules.extend(high_priority_rules)
    #         else:
    #             final_rules.extend(model_rule_list)
        
    #     return final_rules
    


    def _generate_model_selection_rules(self, model_name: str, model_type: str, 
                                      required_features: List[str], target_variable: str) -> List[Dict[str, Any]]:
        """Generate intelligent rules using templates"""
        rules = []
        
        # Rule 1: Feature-based template
        if required_features:
            rules.append(RuleTemplates.apply_template(
                'feature_based',
                model_name=model_name,
                features=required_features
            ))
        
        # Rule 2: Target-based template
        if target_variable and target_variable != 'unknown':
            rules.append(RuleTemplates.apply_template(
                'target_based',
                model_name=model_name,
                target_variable=target_variable,
                min_rows=30
            ))
        
        # Rule 3: Model-type specific rules
        if model_type == 'lightgbm':
            rules.extend(self._generate_lightgbm_rules(model_name, required_features))
        elif model_type == 'prophet':
            rules.extend(self._generate_prophet_rules(model_name))
        elif model_type == 'time_series':
            rules.extend(self._generate_timeseries_rules(model_name))
        elif model_type == 'regression':
            rules.extend(self._generate_regression_rules(model_name, required_features))
        
        # Rule 4: Fallback template
        rules.append(RuleTemplates.apply_template(
            'fallback',
            model_name=model_name
        ))
        
        return rules
    
    def _generate_lightgbm_rules(self, model_name: str, required_features: List[str]) -> List[Dict[str, Any]]:
        """Generate LightGBM specific rules using templates"""
        rules = []
        
        # Check for retail-specific features
        retail_features = ['shop_id', 'store_id', 'product_id', 'sku']
        has_retail_features = any(feature in required_features for feature in retail_features)
        
        if has_retail_features:
            # Retail data rule
            rules.append({
                'name': f"{model_name.lower()}_retail_data",
                'description': f"Select {model_name} for retail data",
                'condition': "('shop_id' in dataset.get('columns', []) or 'store_id' in dataset.get('columns', [])) and dataset.get('granularity') in ['shop_level', 'store_level']",
                'action': f"model_name = '{model_name}'",  # SIMPLE ACTION
                'priority': 10,
                'message': f"{model_name} selected - optimized for retail forecasting"
            })
        
        return rules
    
    def _generate_prophet_rules(self, model_name: str) -> List[Dict[str, Any]]:
        """Generate Prophet specific rules"""
        return [
            {
                'name': f"{model_name.lower()}_seasonal_data",
                'description': f"Select {model_name} for seasonal time series",
                'condition': "dataset.get('frequency') in ['daily', 'weekly', 'monthly'] and dataset.get('row_count', 0) >= 100",
                'action': f"model_name = '{model_name}'",  # SIMPLE ACTION
                'priority': 9,
                'message': f"{model_name} selected - excellent for seasonal patterns"
            }
        ]
    
    def _generate_timeseries_rules(self, model_name: str) -> List[Dict[str, Any]]:
        """Generate time series model rules"""
        return [{
            'name': f"{model_name.lower()}_temporal_data",
            'description': f"Select {model_name} for temporal data",
            'condition': "dataset.get('frequency') != 'none' and 'date' in dataset.get('columns', [])",
            'action': f"model_name = '{model_name}'",  # SIMPLE ACTION
            'priority': 8,
            'message': f"{model_name} selected - designed for time series data"
        }]
    
    def _generate_regression_rules(self, model_name: str, required_features: List[str]) -> List[Dict[str, Any]]:
        """Generate regression model rules"""
        rules = []
        
        # Feature-rich data
        if len(required_features) >= 5:
            rules.append({
                'name': f"{model_name.lower()}_feature_rich",
                'description': f"Select {model_name} for feature-rich data",
                'condition': f"len(dataset.get('columns', [])) >= 8 and dataset.get('row_count', 0) >= 500",
                'action': f"model_name = '{model_name}'",  # SIMPLE ACTION
                'priority': 8,
                'message': f"{model_name} selected - excellent for feature-rich data"
            })
        
        return rules
    
    def backup_rules_before_generation(self) -> str:
        """Create backup of current rules before generating new ones"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f"rules_backup_{timestamp}.yaml")
        
        try:
            if os.path.exists(self.rules_file):
                with open(self.rules_file, 'r') as source:
                    with open(backup_path, 'w') as target:
                        target.write(source.read())
                
                logger.info(f"📦 Created rules backup: {backup_path}")
                return backup_path
            else:
                logger.warning("📁 Rules file doesn't exist, no backup created")
                return ""
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return ""
    
    def rollback_to_backup(self, backup_path: str) -> bool:
        """Rollback to previous rules version"""
        try:
            if os.path.exists(backup_path):
                with open(backup_path, 'r') as source:
                    with open(self.rules_file, 'w') as target:
                        target.write(source.read())
                
                logger.info(f"🔄 Rolled back to backup: {backup_path}")
                return True
            else:
                logger.error(f"❌ Backup file not found: {backup_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def list_available_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        try:
            for filename in os.listdir(self.backup_dir):
                if filename.startswith('rules_backup_') and filename.endswith('.yaml'):
                    filepath = os.path.join(self.backup_dir, filename)
                    stat = os.stat(filepath)
                    backups.append({
                        'filename': filename,
                        'path': filepath,
                        'size_bytes': stat.st_size,
                        'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created_time'], reverse=True)
            return backups
        except Exception as e:
            logger.error(f"❌ Error listing backups: {e}")
            return []
    
    def _add_rules_to_yaml(self, rules: List[Dict[str, Any]]) -> int:
        """Add generated rules to YAML file, return count added"""
        try:
            # Load existing rules
            with open(self.rules_file, 'r') as file:
                existing_data = yaml.safe_load(file) or {}
            
            existing_rules = existing_data.get('rules', [])
            existing_rule_names = {rule['name'] for rule in existing_rules}
            
            added_count = 0
            for rule in rules:
                if rule['name'] not in existing_rule_names:
                    existing_rules.append(rule)
                    added_count += 1
                    logger.info(f"➕ Added rule to YAML: {rule['name']}")
            
            # Update metadata
            existing_data['rules'] = existing_rules
            existing_data['last_updated'] = datetime.now().isoformat()
            existing_data['total_rules'] = len(existing_rules)
            
            # Save updated rules
            with open(self.rules_file, 'w') as file:
                yaml.dump(existing_data, file, default_flow_style=False, indent=2)
            
            logger.info(f"💾 Updated YAML rules file: {self.rules_file} (added {added_count} rules)")
            return added_count
            
        except Exception as e:
            logger.error(f"❌ Error updating YAML rules: {e}")
            return 0
    
    def _add_rules_to_database(self, rules: List[Dict[str, Any]]) -> int:
        """Add generated rules to database, return count added"""
        try:
            added_count = 0
            for rule in rules:
                # Check if rule already exists
                cursor = self.db_connection.execute(
                    "SELECT name FROM Rules WHERE name = ?", (rule['name'],)
                )
                if not cursor.fetchone():
                    self.db_connection.execute("""
                        INSERT INTO Rules 
                        (name, description, condition, action, rule_type, priority, message, is_active, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rule['name'],
                        rule['description'],
                        rule['condition'],
                        rule['action'],
                        'model_selection',
                        rule['priority'],
                        rule.get('message', ''),
                        True,
                        datetime.now().isoformat()
                    ))
                    added_count += 1
                    logger.info(f"💾 Added rule to DB: {rule['name']}")
            
            self.db_connection.commit()
            logger.info(f"📊 Added {added_count} rules to database")
            return added_count
            
        except Exception as e:
            logger.error(f"❌ Error adding rules to database: {e}")
            return 0

    def list_generated_rules(self, model_name: str = None) -> List[Dict[str, Any]]:
        """List all generated rules, optionally filtered by model"""
        try:
            with open(self.rules_file, 'r') as file:
                data = yaml.safe_load(file) or {}
            
            rules = data.get('rules', [])
            if model_name:
                rules = [r for r in rules if model_name.lower() in r['name']]
            
            return rules
        except Exception as e:
            logger.error(f"❌ Error listing rules: {e}")
            return []


def demo_enhanced_rule_generator():
    """Test the enhanced rule generator with all new features"""
    print("🚀 ENHANCED RULE GENERATOR WITH SIMPLE ACTION FORMAT")
    print("=" * 60)
    
    generator = RuleGeneratorService()
    
    # Test model data
    test_models = [
        {
            'model_name': 'Advanced_Retail_Forecaster',
            'model_type': 'lightgbm',
            'required_features': ['shop_id', 'date', 'sales', 'inventory', 'promotion'],
            'target_variable': 'sales'
        },
        {
            'model_name': 'Simple_Time_Series',
            'model_type': 'prophet', 
            'required_features': ['date', 'value'],
            'target_variable': 'value'
        }
    ]
    
    for model in test_models:
        print(f"\n{'='*50}")
        print(f"GENERATING RULES FOR: {model['model_name']}")
        print(f"{'='*50}")
        
        # Create backup first
        backup_path = generator.backup_rules_before_generation()
        print(f"📦 Backup created: {backup_path}")
        
        # Generate rules
        result = generator.generate_rules_for_model(model)
        
        if result['success']:
            print(f"✅ Successfully generated rules for {model['model_name']}")
            print(f"📊 Generated {len(result['generated_rules'])} rules")
            
            # Show validation results
            validation = result['validation_results']
            print(f"🔍 Validation: {validation['summary']['valid_count']} valid, "
                  f"{validation['summary']['invalid_count']} invalid, "
                  f"{validation['summary']['warning_count']} warnings")
            
            # Show optimization stats
            opt_stats = result['optimization_stats']
            print(f"🔧 Optimization: {opt_stats['reduction_percentage']:.1f}% reduction "
                  f"({opt_stats['original_rules']} → {opt_stats['optimized_rules']})")
            
            # List some generated rules
            print(f"\n📋 Sample generated rules:")
            for i, rule in enumerate(result['generated_rules'][:3], 1):
                print(f"  {i}. {rule['name']} (priority: {rule['priority']})")
                print(f"     Condition: {rule['condition'][:80]}...")
                print(f"     Action: {rule['action']}")
            
        else:
            print(f"❌ Failed to generate rules: {result.get('error', 'Unknown error')}")
        
        # List available templates
        print(f"\n🎯 Available templates: {', '.join(RuleTemplates.get_available_templates())}")
    
    # Show backup information
    backups = generator.list_available_backups()
    print(f"\n📦 Available backups: {len(backups)}")
    for backup in backups[:3]:  # Show latest 3 backups
        print(f"  • {backup['filename']} ({backup['size_bytes']} bytes)")
    
    print("\n🎉 ENHANCED DEMO COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    demo_enhanced_rule_generator()