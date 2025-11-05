# app/scripts/manage_rules.py
#!/usr/bin/env python3
"""
Rule Management CLI - Hybrid approach for managing forecasting rules
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.knowledge_base_services.core.rule_manager import RuleManager


def main():
    parser = argparse.ArgumentParser(description='Manage forecasting rules')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add model command
    add_parser = subparsers.add_parser('add-model', help='Add a new model with auto-generated rules')
    add_parser.add_argument('--name', required=True, help='Model name')
    add_parser.add_argument('--type', required=True, choices=['lightgbm', 'prophet', 'time_series', 'regression'], help='Model type')
    add_parser.add_argument('--features', required=True, help='Comma-separated required features')
    add_parser.add_argument('--target', required=True, help='Target variable')
    
    # List command
    list_parser = subparsers.add_parser('list-models', help='List all models with rules')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show rule statistics')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify rules work with sample data')
    
    args = parser.parse_args()
    
    manager = RuleManager()
    
    if args.command == 'add-model':
        # Add new model with auto-generated rules
        model_info = {
            'model_name': args.name,
            'model_type': args.type,
            'required_features': [f.strip() for f in args.features.split(',')],
            'target_variable': args.target
        }
        
        result = manager.add_model_with_rules(model_info)
        print(json.dumps(result, indent=2))
        
    elif args.command == 'list-models':
        # List all models with rules
        models = manager.list_all_models_with_rules()
        print("📋 Models with Rules:")
        print(json.dumps(models, indent=2))
        
    elif args.command == 'stats':
        # Show statistics
        stats = manager.get_rule_statistics()
        print("📊 Rule Statistics:")
        print(json.dumps(stats, indent=2))
        
    elif args.command == 'verify':
        # Verify rules (you can extend this)
        print("✅ Rules verification - use specific model verification")
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()