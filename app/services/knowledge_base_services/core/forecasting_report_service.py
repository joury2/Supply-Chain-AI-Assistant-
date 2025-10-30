# app/services/knowledge_base_services/core/forecasting_report_service.py
# Comprehensive Forecasting Report Service with Detailed Analysis and Actionable Insights 
import sys
import os
from typing import Dict, List, Any
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingReportService:
    """
    Generates user-friendly reports about forecasting feasibility
    """
    
    def generate_forecasting_report(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive, user-friendly forecasting report
        """
        summary = analysis_result['summary']
        recommendations = analysis_result['recommendations']
        selection_analysis = analysis_result.get('selection_analysis', {})
        
        if summary['can_proceed']:
            return self._generate_success_report(analysis_result)
        else:
            return self._generate_rejection_report(analysis_result)
    
    def _generate_success_report(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report when forecasting can proceed"""
        summary = analysis_result['summary']
        selection = analysis_result['model_selection']
        
        return {
            'status': 'SUCCESS',
            'title': '✅ Forecasting Ready',
            'message': f"Your dataset is ready for forecasting with {selection.get('selected_model')}",
            'details': {
                'selected_model': selection.get('selected_model'),
                'confidence': f"{selection.get('confidence', 0):.1%}",
                'reason': selection.get('reason', 'Dataset matches model requirements'),
                'next_steps': summary['next_steps']
            },
            'recommendations': self._format_recommendations(analysis_result['recommendations']),
            'quick_actions': [
                f"Generate forecast with {selection.get('selected_model')}",
                "Download forecast report",
                "Schedule automated forecasts"
            ]
        }
    
    def _generate_rejection_report(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed rejection report with actionable insights"""
        summary = analysis_result['summary']
        recommendations = analysis_result['recommendations']
        selection_analysis = analysis_result.get('selection_analysis', {})
        
        rejection_reason = summary.get('rejection_reason', 'UNKNOWN')
        
        if rejection_reason == 'DATA_VALIDATION_FAILED':
            title = "❌ Data Quality Issues"
            message = "Your dataset has validation errors that prevent forecasting"
        elif rejection_reason == 'NO_COMPATIBLE_MODELS':
            title = "❌ No Compatible Forecasting Models"
            message = "Your dataset doesn't match any available forecasting models"
        else:
            title = "❌ Cannot Generate Forecast"
            message = "There are issues with your dataset that prevent forecasting"
        
        # Analyze what models were considered and why they failed
        model_analysis = self._analyze_model_compatibility(selection_analysis)
        
        return {
            'status': 'REJECTED',
            'title': title,
            'message': message,
            'details': {
                'rejection_reason': rejection_reason,
                'compatible_models': summary.get('compatible_models_count', 0),
                'incompatible_models': summary.get('incompatible_models_count', 0),
                'model_analysis': model_analysis,
                'critical_issues': [r for r in recommendations if r['priority'] == 'high']
            },
            'recommendations': self._format_recommendations(recommendations),
            'action_plan': self._generate_action_plan(recommendations),
            'quick_actions': [
                "Fix data quality issues",
                "Add required features",
                "Contact support for help"
            ]
        }
    
    def _analyze_model_compatibility(self, selection_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model compatibility for the report"""
        analysis = selection_analysis.get('analysis', {})
        missing_reqs = analysis.get('missing_requirements', [])
        
        if not missing_reqs:
            return {'message': 'No model compatibility analysis available'}
        
        # Group by common issues
        common_issues = {}
        for req in missing_reqs:
            if isinstance(req, dict):  # Ensure it's a dictionary
                for issue in req.get('issues', []):
                    if issue not in common_issues:
                        common_issues[issue] = []
                    common_issues[issue].append(req.get('model_name', 'Unknown model'))
        
        if not common_issues:
            return {'message': 'No common issues identified'}
        
        return {
            'total_models_analyzed': len(missing_reqs),
            'common_issues': [
                {
                    'issue': issue,
                    'affected_models': models,
                    'model_count': len(models)
                }
                for issue, models in common_issues.items()
            ],
            'most_common_issue': max(common_issues.items(), key=lambda x: len(x[1])) if common_issues else None
        }
    
    def _format_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format recommendations for user display"""
        return [
            {
                'priority': rec['priority'],
                'category': rec['category'],
                'description': rec['message'],
                'action': rec['action'],
                'impact': rec.get('impact', '')
            }
            for rec in recommendations
        ]
    
    def _generate_action_plan(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a step-by-step action plan from recommendations"""
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        medium_priority = [r for r in recommendations if r['priority'] == 'medium']
        
        action_plan = []
        
        # Add high priority actions first
        for rec in high_priority:
            action_plan.append({
                'step': f"CRITICAL: {rec['action']}",
                'reason': rec['message'],
                'priority': 'high',
                'estimated_effort': 'Immediate attention required'
            })
        
        # Add medium priority actions
        for rec in medium_priority:
            action_plan.append({
                'step': rec['action'],
                'reason': rec['message'],
                'priority': 'medium',
                'estimated_effort': 'Recommended for better results'
            })
        
        return action_plan

def create_mock_analysis_result():
    """Create a mock analysis result for testing"""
    return {
        'summary': {
            'can_proceed': False,
            'status_message': '❌ Cannot proceed: No compatible models found',
            'confidence': 0.0,
            'rejection_reason': 'NO_COMPATIBLE_MODELS',
            'compatible_models_count': 0,
            'incompatible_models_count': 5,
            'next_steps': [
                'Add required features based on recommendations',
                'Consider different dataset structure',
                'Contact support for model customization'
            ]
        },
        'recommendations': [
            {
                'type': 'requirement',
                'priority': 'high',
                'category': 'model_compatibility',
                'message': 'TFT: Missing features: demand',
                'action': 'Add missing features or choose different model',
                'impact': 'Cannot use TFT for forecasting'
            },
            {
                'type': 'requirement', 
                'priority': 'high',
                'category': 'model_compatibility',
                'message': 'Prophet: Missing features: demand',
                'action': 'Add missing features or choose different model',
                'impact': 'Cannot use Prophet for forecasting'
            },
            {
                'type': 'suggestion',
                'priority': 'medium',
                'category': 'data_volume', 
                'message': 'Limited data (48 points) may affect forecast accuracy',
                'action': 'Collect more historical data',
                'impact': 'Improved model performance and reliability'
            }
        ],
        'selection_analysis': {
            'analysis': {
                'missing_requirements': [
                    {
                        'model_name': 'TFT',
                        'model_type': 'time_series',
                        'issues': ['Missing features: demand'],
                        'required_features': ['date', 'demand'],
                        'target_variable': 'demand'
                    },
                    {
                        'model_name': 'Prophet',
                        'model_type': 'time_series', 
                        'issues': ['Missing features: demand'],
                        'required_features': ['date', 'demand'],
                        'target_variable': 'demand'
                    }
                ]
            }
        }
    }

def test_report_service():
    """Test the reporting service with mock data"""
    print("📊 TESTING FORECASTING REPORT SERVICE")
    print("=" * 50)
    
    report_service = ForecastingReportService()
    
    # Create mock analysis result
    mock_analysis = create_mock_analysis_result()
    
    print("Testing with incompatible dataset...")
    print("Dataset columns: ['shop_id', 'date', 'price']")
    
    report = report_service.generate_forecasting_report(mock_analysis)
    
    print(f"\n📋 REPORT: {report['title']}")
    print(f"📝 Status: {report['status']}")
    print(f"💬 Message: {report['message']}")
    
    if report['status'] == 'REJECTED':
        print(f"\n🔍 DETAILS:")
        print(f"   Rejection Reason: {report['details']['rejection_reason']}")
        print(f"   Compatible Models: {report['details']['compatible_models']}")
        print(f"   Incompatible Models: {report['details']['incompatible_models']}")
        
        # Show model analysis
        model_analysis = report['details']['model_analysis']
        if 'common_issues' in model_analysis:
            print(f"\n📊 MODEL COMPATIBILITY ANALYSIS:")
            for issue in model_analysis['common_issues']:
                print(f"   • {issue['issue']}")
                print(f"     Affects {issue['model_count']} models")
        
        print(f"\n📋 ACTION PLAN:")
        for action in report.get('action_plan', []):
            priority_icon = "🔴" if action['priority'] == 'high' else "🟡"
            print(f"   {priority_icon} {action['step']}")
            print(f"      Why: {action['reason']}")
    
    print(f"\n🚀 QUICK ACTIONS:")
    for action in report.get('quick_actions', []):
        print(f"   • {action}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report.get('recommendations', []):
        priority_icon = "🔴" if rec['priority'] == 'high' else "🟡"
        print(f"   {priority_icon} [{rec['category']}] {rec['description']}")
        print(f"      → Action: {rec['action']}")

if __name__ == "__main__":
    test_report_service()