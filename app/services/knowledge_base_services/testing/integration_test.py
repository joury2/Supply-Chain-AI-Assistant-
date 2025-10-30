# app/services/knowledge_base_services/integration_test.py
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

def integration_test():
    """Test the complete integrated system"""
    print("🚀 COMPLETE SYSTEM INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Import all services
        from app.services.knowledge_base_services.core.rule_engine_service import RuleEngineService
        from app.services.knowledge_base_services.core.forecasting_report_service import ForecastingReportService
        
        print("✅ All services imported successfully!")
        
        # Initialize services
        rule_service = RuleEngineService()
        report_service = ForecastingReportService()
        
        print("✅ Services initialized successfully!")
        
        # Test datasets
        test_datasets = [
            {
                'name': 'good_time_series',
                'frequency': 'monthly',
                'granularity': 'product_level', 
                'row_count': 100,
                'columns': ['date', 'demand'],
                'missing_percentage': 0.02,
                'expected': 'Should find compatible models'
            },
            {
                'name': 'incomplete_data',
                'frequency': 'monthly',
                'granularity': 'shop_level',
                'row_count': 8,
                'columns': ['shop_id', 'price'],
                'missing_percentage': 0.40,
                'expected': 'Should be rejected with clear reasons'
            }
        ]
        
        for dataset in test_datasets:
            print(f"\n{'='*40}")
            print(f"TESTING: {dataset['name']}")
            print(f"Expected: {dataset['expected']}")
            print(f"Columns: {dataset['columns']}")
            print(f"{'='*40}")
            
            # Step 1: Analyze dataset
            analysis = rule_service.analyze_dataset(dataset)
            
            # Step 2: Generate report
            report = report_service.generate_forecasting_report(analysis)
            
            # Display results
            print(f"📊 ANALYSIS RESULT: {report['title']}")
            print(f"💬 {report['message']}")
            
            if report['status'] == 'REJECTED':
                print(f"🔍 Rejection Reason: {report['details']['rejection_reason']}")
                
                # Show top recommendations
                high_priority_recs = [r for r in report['recommendations'] if r['priority'] == 'high']
                if high_priority_recs:
                    print(f"\n🔴 TOP {min(3, len(high_priority_recs))} CRITICAL ISSUES:")
                    for rec in high_priority_recs[:3]:
                        print(f"   • {rec['description']}")
            
            print(f"\n🚀 QUICK ACTIONS:")
            for action in report.get('quick_actions', [])[:2]:
                print(f"   • {action}")
        
        print(f"\n🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This might be due to path issues. Let's check the imports:")
        
        # Check what's available
        services_dir = os.path.join(project_root, 'app', 'services', 'knowledge_base_services')
        if os.path.exists(services_dir):
            print(f"📁 Files in services directory:")
            for file in os.listdir(services_dir):
                if file.endswith('.py'):
                    print(f"   - {file}")
        else:
            print(f"❌ Services directory not found: {services_dir}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    integration_test()