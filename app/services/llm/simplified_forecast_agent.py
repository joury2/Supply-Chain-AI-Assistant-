# app/services/llm/simplified_forecast_agent.py
"""
Simplified Forecast Agent - DEBUGGED VERSION
Fixed: Initialization errors, error handling, response generation
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import traceback
import logging
from pathlib import Path 

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Import your ACTUAL supply chain service
from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService

# Import interpretation service if needed
from app.services.llm.interpretation_service import LLMInterpretationService


# Configure logging properly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedForecastAgent:
    """
    Simplified forecast agent - FIXED VERSION
    """
    
     
    def __init__(
        self, 
        nvidia_api_key: str,
        vector_store_path: str = "vector_store",
        db_path: str = "supply_chain.db"
    ):
        logger.info("🚀 Starting agent initialization...")
        self.nvidia_api_key = nvidia_api_key
        self.vector_store_path = vector_store_path
        self.db_path = db_path
        
        # Initialize YOUR actual service FIRST
        logger.info("🔧 Initializing Supply Chain Forecasting Service...")
        self.forecasting_service = SupplyChainForecastingService(db_path=db_path)
        
        # Session state with persistence
        self._session_data = {}  # Store multiple sessions
        self._current_session_id = None
        self.conversation_history = []
        
        # Current dataset for backward compatibility
        self.current_dataset_df = None
        self.current_result = None
        
        # Initialize RAG
        logger.info("📚 Loading RAG vector store...")
        try:
            self.vectorstore = self._load_vectorstore()
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            self.has_rag = True
        except Exception as e:
            logger.warning(f"⚠️ RAG not available: {e}")
            self.vectorstore = None
            self.retriever = None
            self.has_rag = False
        
        # Initialize LLM
        logger.info("🤖 Connecting to NVIDIA Llama 3.1...")
        try:
            self.llm = ChatNVIDIA(
                model="meta/llama-3.1-70b-instruct",
                api_key=nvidia_api_key,
                temperature=0.7,
                max_completion_tokens=1024,
                stop=["Observation:", "\nObservation"]
            )
        except Exception as e:
            logger.warning(f"⚠️ LLM connection failed: {e}")
            self.llm = None
        
        # Initialize interpretation service
        logger.info("📊 Initializing Interpretation Service...")
        try:
            self.interpretation_service = LLMInterpretationService(
                nvidia_api_key=self.nvidia_api_key,
                vector_store_path=self.vector_store_path
            )
            self.has_interpretation = True
        except Exception as e:
            logger.warning(f"⚠️ Interpretation service unavailable: {e}")
            self.interpretation_service = None
            self.has_interpretation = False
        
        logger.info("✅ Forecast Agent initialized successfully!")

    def _create_tools(self) -> list:
        """Create LangChain tools from your backend functions"""
        
        def analyze_dataset_tool(analysis_type: str = "comprehensive") -> str:
            """Analyzes the uploaded dataset for forecasting suitability"""
            if self.current_dataset_df is None:
                return json.dumps({"error": "No dataset uploaded yet."})
            
            try:
                # Create dataset info WITHOUT DataFrame for analysis
                dataset_info = {
                    'name': 'user_uploaded_dataset',
                    'columns': list(self.current_dataset_df.columns),
                    'row_count': len(self.current_dataset_df),
                    'frequency': self._detect_frequency(),
                    'missing_percentage': float(self.current_dataset_df.isnull().sum().sum() / 
                                            (len(self.current_dataset_df) * len(self.current_dataset_df.columns)))
                }
                
                # Call service WITHOUT DataFrame in dict
                analysis = self.forecasting_service.analyze_dataset_with_knowledge_base(dataset_info)
                
                # Format for LLM (JSON-serializable only)
                result = {
                    "status": "success",
                    "validation": {
                        "valid": analysis['rule_analysis']['validation']['valid'],
                        "errors": analysis['rule_analysis']['validation'].get('errors', []),
                        "warnings": analysis['rule_analysis']['validation'].get('warnings', [])
                    },
                    "compatible_models": [
                        m['model_name'] for m in analysis['rule_analysis']['selection_analysis']['analysis'].get('compatible_models', [])
                    ],
                    "selected_model": analysis['rule_analysis']['model_selection'].get('selected_model'),
                    "confidence": analysis['combined_summary'].get('confidence', 0),
                    "can_proceed": analysis['combined_summary']['can_proceed'],
                    "recommendations": [
                        r if isinstance(r, str) else r.get('message', str(r)) 
                        for r in (analysis.get('knowledge_base_recommendations', []) or [])[:3]
                    ],
                    "summary": str(analysis['combined_summary'])
                }
                
                # Store for later use
                self.current_result = analysis
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                return json.dumps({
                    "error": f"Analysis failed: {str(e)}",
                    "detail": error_detail[:500]
                })

        def run_forecast_tool(horizon_days: str) -> str:
            """Runs complete forecasting pipeline using the selected model"""
            if self.current_dataset_df is None:
                return json.dumps({"error": "No dataset available. Upload dataset first."})
            
            try:
                # Validate horizon input
                try:
                    horizon = int(horizon_days)
                except (ValueError, TypeError):
                    return json.dumps({
                        "error": f"Invalid horizon: '{horizon_days}'. Must be a number.",
                        "example": "Use '30' for 30 days"
                    })
                
                if horizon <= 0:
                    return json.dumps({
                        "error": f"Invalid horizon: {horizon}. Must be a positive number."
                    })
                
                # Create dataset info WITH DataFrame for processing
                dataset_info = self._create_dataset_info(include_dataframe=True)
                
                # Call YOUR actual complete forecasting service
                result = self.forecasting_service.process_forecasting_request(
                    dataset_info=dataset_info,
                    forecast_horizon=horizon,
                    business_context={'source': 'chatbot_interface'}
                )
                
                self.current_result = result
                
                # 🎯 ADDED: Generate interpretation after successful forecast
                interpretation = None
                if result['status'] == 'success' and hasattr(self, 'interpretation_service'):
                    try:
                        interpretation = self.interpretation_service.interpret_forecast(
                            forecast_result=result['forecast'],
                            analysis_result=result,
                            business_context={'type': 'general', 'source': 'forecast_agent'}
                        )
                        logger.info("✅ Forecast interpretation generated successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Interpretation service failed: {str(e)}")
                        interpretation = {
                            'summary': 'Forecast completed successfully. Interpretation service temporarily unavailable.',
                            'key_insights': ['Forecast generated with confidence intervals', 'Consider monitoring actual vs predicted values'],
                            'business_recommendations': ['Monitor forecast accuracy over time', 'Adjust plans based on actual performance']
                        }
                
                # Format response based on status (JSON-serializable)
                if result['status'] == 'success':
                    response = {
                        "status": "success",
                        "model_used": result['selected_model']['model_name'],
                        "confidence": f"{result['selected_model']['confidence']*100:.1f}%",
                        "forecast_horizon": horizon,
                        "forecast_summary": {
                            "values_count": len(result['forecast'].get('values', [])),
                            "has_confidence_intervals": 'confidence_intervals' in result['forecast'],
                            "first_value": result['forecast'].get('values', [None])[0],
                            "last_value": result['forecast'].get('values', [None])[-1] if result['forecast'].get('values') else None
                        },
                        "interpretation": interpretation,  # 🎯 ADDED: Include interpretation in response
                        "visualizations_available": list(result.get('visualizations', {}).keys())
                    }
                else:
                    response = {
                        "status": result['status'],
                        "error": result.get('error', 'Unknown error'),
                        "recommendations": result.get('recommendations', [])
                    }
                
                return json.dumps(response, indent=2)
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                return json.dumps({
                    "error": f"Forecasting failed: {str(e)}",
                    "detail": error_detail[:500]
                })
            
        def get_model_details_tool(model_name: str) -> str:
            """Retrieves detailed information about a specific forecasting model"""
            try:
                # Use RAG to retrieve model information
                query = f"Tell me about {model_name} model requirements and capabilities"
                docs = self.retriever.get_relevant_documents(query)
                
                # Combine relevant information
                details = "\n\n".join([doc.page_content for doc in docs[:2]])
                return details
                
            except Exception as e:
                return f"Error retrieving model details: {str(e)}"
        
        def explain_results_tool(aspect: str) -> str:
            """Explains forecast results or recommendations using RAG knowledge and actual results"""
            if self.current_result is None:
                return "No forecast results available yet. Please run analysis or forecast first."
            
            try:
                # Use RAG to provide context
                query = f"Explain {aspect} in forecasting context"
                docs = self.retriever.get_relevant_documents(query)
                
                context = docs[0].page_content if docs else ""
                
                # Add actual results
                result_info = ""
                if 'analysis' in self.current_result:
                    result_info = f"\nYour analysis results:\n{json.dumps(self.current_result.get('combined_summary', {}), indent=2)}"
                elif 'selected_model' in self.current_result:
                    result_info = f"\nYour forecast used:\n" + \
                                f"Model: {self.current_result['selected_model']['model_name']}\n" + \
                                f"Confidence: {self.current_result['selected_model']['confidence']:.1%}"
                
                return f"Based on knowledge base:\n{context}\n{result_info}"
                
            except Exception as e:
                return f"Error explaining: {str(e)}"

        def get_forecast_data_tool(data_format: str = "summary") -> str:
            """Retrieves actual forecast data values"""
            if self.current_result is None or 'forecast' not in self.current_result:
                return json.dumps({"error": "No forecast data available. Run forecast first."})
            
            try:
                forecast = self.current_result['forecast']
                
                if data_format == "summary":
                    return json.dumps({
                        "forecast_points": len(forecast.get('values', [])),
                        "first_value": forecast.get('values', [None])[0],
                        "last_value": forecast.get('values', [None])[-1] if forecast.get('values') else None,
                        "has_confidence_intervals": 'confidence_intervals' in forecast
                    }, indent=2)
                elif data_format == "values":
                    return json.dumps({
                        "values": forecast.get('values', [])[:10],
                        "note": "Showing first 10 values. Use 'full' format for complete data."
                    }, indent=2)
                else:  # full
                    return json.dumps(forecast, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"Error retrieving forecast data: {str(e)}"})

        def interpret_forecast_patterns_tool(aspect: str) -> str:
            """Interprets specific patterns in the forecast (trend, seasonality, anomalies)"""
            if self.current_result is None or 'forecast' not in self.current_result:
                return json.dumps({"error": "No forecast available to interpret. Run forecast first."})
            
            try:
                forecast = self.current_result['forecast']
                forecast_values = forecast.get('values', [])
                
                if not forecast_values:
                    return json.dumps({"error": "No forecast values found"})
                
                interpretation = {}
                
                # Analyze trend
                if aspect in ["trend", "all"]:
                    first_half_avg = sum(forecast_values[:len(forecast_values)//2]) / (len(forecast_values)//2)
                    second_half_avg = sum(forecast_values[len(forecast_values)//2:]) / (len(forecast_values) - len(forecast_values)//2)
                    
                    trend_change = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                    
                    if trend_change > 5:
                        trend_desc = f"Upward trend: Values are increasing by approximately {trend_change:.1f}% over the forecast period"
                    elif trend_change < -5:
                        trend_desc = f"Downward trend: Values are decreasing by approximately {abs(trend_change):.1f}% over the forecast period"
                    else:
                        trend_desc = "Stable trend: Values remain relatively constant throughout the forecast period"
                    
                    interpretation['trend'] = {
                        'description': trend_desc,
                        'change_percentage': round(trend_change, 2),
                        'direction': 'up' if trend_change > 0 else 'down' if trend_change < 0 else 'stable'
                    }
                
                # Analyze volatility/uncertainty
                if aspect in ["uncertainty", "all"]:
                    values_std = np.std(forecast_values)
                    values_mean = np.mean(forecast_values)
                    coefficient_of_variation = (values_std / values_mean) * 100
                    
                    if 'confidence_intervals' in forecast:
                        ci = forecast['confidence_intervals']
                        avg_interval_width = np.mean([u - l for u, l in zip(ci['upper'], ci['lower'])])
                        uncertainty_desc = f"Moderate uncertainty with confidence intervals averaging ±{avg_interval_width:.1f} units"
                    else:
                        if coefficient_of_variation < 10:
                            uncertainty_desc = "Low volatility: Forecast values are quite stable"
                        elif coefficient_of_variation < 25:
                            uncertainty_desc = "Moderate volatility: Some variation expected"
                        else:
                            uncertainty_desc = "High volatility: Significant variation in forecast values"
                    
                    interpretation['uncertainty'] = {
                        'description': uncertainty_desc,
                        'coefficient_of_variation': round(coefficient_of_variation, 2),
                        'volatility_level': 'low' if coefficient_of_variation < 10 else 'moderate' if coefficient_of_variation < 25 else 'high'
                    }
                
                # Detect potential anomalies or unusual patterns
                if aspect in ["anomalies", "all"]:
                    mean_val = np.mean(forecast_values)
                    std_val = np.std(forecast_values)
                    
                    anomalies = []
                    for i, val in enumerate(forecast_values):
                        z_score = (val - mean_val) / std_val if std_val > 0 else 0
                        if abs(z_score) > 2:
                            anomalies.append({
                                'period': i + 1,
                                'value': round(val, 2),
                                'z_score': round(z_score, 2)
                            })
                    
                    if anomalies:
                        anomaly_desc = f"Found {len(anomalies)} unusual value(s) that deviate significantly from the forecast pattern"
                    else:
                        anomaly_desc = "No unusual patterns detected - forecast values follow expected distribution"
                    
                    interpretation['anomalies'] = {
                        'description': anomaly_desc,
                        'count': len(anomalies),
                        'details': anomalies[:3]
                    }
                
                # Add context from RAG
                rag_query = f"Explain {aspect} in forecasting context and what it means for decision making"
                rag_docs = self.retriever.get_relevant_documents(rag_query)
                rag_context = rag_docs[0].page_content[:500] if rag_docs else ""
                
                interpretation['knowledge_context'] = rag_context
                interpretation['model_used'] = self.current_result.get('selected_model', {}).get('model_name', 'Unknown')
                
                return json.dumps(interpretation, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"Interpretation failed: {str(e)}"})

        def compare_scenarios_tool(scenario_description: str) -> str:
            """Helps users understand 'what-if' scenarios and compare different forecasts"""
            if self.current_result is None or 'forecast' not in self.current_result:
                return json.dumps({"error": "No baseline forecast available for comparison"})
            
            try:
                baseline = self.current_result['forecast']['values']
                
                # Parse common scenario types
                scenario_lower = scenario_description.lower()
                
                comparison = {
                    'baseline_forecast': {
                        'mean': round(np.mean(baseline), 2),
                        'total': round(sum(baseline), 2),
                        'min': round(min(baseline), 2),
                        'max': round(max(baseline), 2)
                    }
                }
                
                # Detect percentage change scenarios
                if 'increase' in scenario_lower or 'decrease' in scenario_lower:
                    import re
                    percentages = re.findall(r'(\d+)%', scenario_description)
                    
                    if percentages:
                        pct = float(percentages[0]) / 100
                        if 'decrease' in scenario_lower:
                            pct = -pct
                        
                        adjusted_values = [v * (1 + pct) for v in baseline]
                        
                        comparison['scenario'] = {
                            'description': scenario_description,
                            'adjustment': f"{pct*100:+.1f}%",
                            'mean': round(np.mean(adjusted_values), 2),
                            'total': round(sum(adjusted_values), 2),
                            'difference_from_baseline': round(sum(adjusted_values) - sum(baseline), 2)
                        }
                        
                        # Business impact
                        diff_pct = ((sum(adjusted_values) - sum(baseline)) / sum(baseline)) * 100
                        comparison['impact'] = f"This scenario would result in {diff_pct:+.1f}% change in total forecasted values"
                
                else:
                    # Generic scenario - provide guidance
                    comparison['guidance'] = """To compare scenarios, please specify:
                    - Percentage changes (e.g., 'increase by 20%')
                    - Absolute changes (e.g., 'add 100 units per day')
                    - External factors (e.g., 'impact of new product launch')
                    
                    I can help you understand the implications of different assumptions."""
                
                return json.dumps(comparison, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"Scenario comparison failed: {str(e)}"})

        def explain_model_choice_tool(question: str) -> str:
            """Explains why a specific model was chosen and its characteristics"""
            if self.current_result is None:
                return json.dumps({"error": "No analysis or forecast result available"})
            
            try:
                explanation = {}
                
                # Get selected model info
                if 'selected_model' in self.current_result:
                    model_info = self.current_result['selected_model']
                    explanation['selected_model'] = {
                        'name': model_info.get('model_name', 'Unknown'),
                        'confidence': f"{model_info.get('confidence', 0)*100:.1f}%",
                        'selection_reason': model_info.get('reason', 'Not specified'),
                        'source': model_info.get('source', 'Unknown')
                    }
                
                # Get alternative models if available
                if 'analysis' in self.current_result:
                    analysis = self.current_result['analysis']
                    if 'rule_analysis' in analysis:
                        compatible = analysis['rule_analysis']['selection_analysis']['analysis'].get('compatible_models', [])
                        
                        if compatible:
                            explanation['compatible_models'] = [
                                {
                                    'name': m['model_name'],
                                    'status': m.get('status', 'unknown'),
                                    'score': f"{m.get('compatibility_score', 0):.1f}%"
                                }
                                for m in compatible[:5]
                            ]
                
                # Use RAG to explain model characteristics
                if 'selected_model' in explanation:
                    model_name = explanation['selected_model']['name']
                    rag_query = f"Explain {model_name} model strengths weaknesses and when to use it"
                    rag_docs = self.retriever.get_relevant_documents(rag_query)
                    
                    if rag_docs:
                        explanation['model_characteristics'] = rag_docs[0].page_content[:800]
                
                # Add dataset context
                if self.current_dataset_info:
                    explanation['dataset_context'] = {
                        'rows': self.current_dataset_info.get('row_count', 0),
                        'columns': len(self.current_dataset_info.get('columns', [])),
                        'frequency': self.current_dataset_info.get('frequency', 'unknown')
                    }
                    
                    # Explain why this model fits
                    rows = explanation['dataset_context']['rows']
                    if rows < 50:
                        explanation['fit_reason'] = "Selected model works well with limited data"
                    elif rows < 500:
                        explanation['fit_reason'] = "Selected model is suitable for medium-sized datasets"
                    else:
                        explanation['fit_reason'] = "Selected model can leverage large dataset effectively"
                
                return json.dumps(explanation, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"Model explanation failed: {str(e)}"})

        def suggest_actions_tool(context: str) -> str:
            """Suggests actionable next steps based on forecast results"""
            if self.current_result is None or 'forecast' not in self.current_result:
                return json.dumps({"error": "No forecast available for action suggestions"})
            
            try:
                forecast = self.current_result['forecast']
                forecast_values = forecast.get('values', [])
                
                if not forecast_values:
                    return json.dumps({"error": "No forecast values available"})
                
                suggestions = {
                    'context': context,
                    'forecast_summary': {
                        'horizon': len(forecast_values),
                        'average_value': round(np.mean(forecast_values), 2),
                        'total_value': round(sum(forecast_values), 2),
                        'trend': 'increasing' if forecast_values[-1] > forecast_values[0] else 'decreasing'
                    },
                    'recommendations': []
                }
                
                # Context-specific recommendations
                context_lower = context.lower()
                
                if 'inventory' in context_lower or 'stock' in context_lower:
                    suggestions['recommendations'].extend([
                        {
                            'priority': 'high',
                            'action': 'Review inventory levels',
                            'detail': f"Ensure stock can meet forecasted demand of {sum(forecast_values):.0f} units over the forecast period"
                        },
                        {
                            'priority': 'medium',
                            'action': 'Adjust reorder points',
                            'detail': f"Based on average daily forecast of {np.mean(forecast_values):.1f}, recalculate safety stock"
                        }
                    ])
                
                elif 'budget' in context_lower or 'financial' in context_lower:
                    suggestions['recommendations'].extend([
                        {
                            'priority': 'high',
                            'action': 'Allocate budget resources',
                            'detail': f"Plan for total forecasted value of {sum(forecast_values):.0f}"
                        },
                        {
                            'priority': 'medium',
                            'action': 'Monitor variance',
                            'detail': "Set up alerts for actual vs forecast deviation > 15%"
                        }
                    ])
                
                elif 'capacity' in context_lower or 'production' in context_lower:
                    peak_demand = max(forecast_values)
                    avg_demand = np.mean(forecast_values)
                    
                    suggestions['recommendations'].extend([
                        {
                            'priority': 'high',
                            'action': 'Ensure production capacity',
                            'detail': f"Peak forecasted demand is {peak_demand:.0f}, ensure capacity covers {peak_demand * 1.1:.0f} (110% buffer)"
                        },
                        {
                            'priority': 'medium',
                            'action': 'Optimize scheduling',
                            'detail': f"Average demand of {avg_demand:.0f} allows for efficiency improvements"
                        }
                    ])
                
                else:
                    # Generic recommendations
                    suggestions['recommendations'].extend([
                        {
                            'priority': 'high',
                            'action': 'Monitor forecast accuracy',
                            'detail': "Track actual vs forecasted values to refine future predictions"
                        },
                        {
                            'priority': 'medium',
                            'action': 'Plan for uncertainty',
                            'detail': "Consider confidence intervals when making critical decisions"
                        },
                        {
                            'priority': 'low',
                            'action': 'Document assumptions',
                            'detail': "Record any assumptions made for future reference"
                        }
                    ])
                
                # Add data-driven insights
                if len(forecast_values) > 7:
                    # Check for weekly patterns
                    weekly_avg = [np.mean(forecast_values[i:i+7]) for i in range(0, len(forecast_values)-6, 7)]
                    if len(weekly_avg) > 1:
                        weekly_variance = np.std(weekly_avg) / np.mean(weekly_avg)
                        if weekly_variance > 0.15:
                            suggestions['recommendations'].append({
                                'priority': 'medium',
                                'action': 'Address weekly variations',
                                'detail': f"Significant weekly variation detected ({weekly_variance*100:.1f}% coefficient of variation)"
                            })
                
                return json.dumps(suggestions, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"Action suggestion failed: {str(e)}"})

        # Create LangChain tools
        tools = [
            Tool(
                name="analyze_dataset",
                func=analyze_dataset_tool,
                description="Analyzes uploaded dataset for forecasting. Returns validation status, compatible models, and recommendations. Use this when user uploads data or asks to analyze their dataset."
            ),
            Tool(
                name="run_forecast",
                func=run_forecast_tool,
                description="Runs COMPLETE forecasting pipeline for specified number of days. Input should be number of days as string (e.g., 30 for one month, 90 for quarter). Use this when user asks to forecast or predict future values."
            ),
            Tool(
                name="get_model_details",
                func=get_model_details_tool,
                description="Retrieves detailed information about a specific model from knowledge base. Use when user asks about model capabilities, requirements, or how a model works."
            ),
            Tool(
                name="explain_results",
                func=explain_results_tool,
                description="Explains forecast results or recommendations using domain knowledge from RAG and actual results. Use when user wants to understand results, model selection, or needs clarification."
            ),
            Tool(
                name="get_forecast_data",
                func=get_forecast_data_tool,
                description="Retrieves actual forecast data values. Use when user asks for specific numbers, wants to see predictions, or needs data for download."
            ),
            Tool(
                name="interpret_forecast_patterns",
                func=interpret_forecast_patterns_tool,
                description="Interprets patterns in the forecast like trend, seasonality, anomalies, or uncertainty. Use when user asks to 'explain the forecast', 'what patterns do you see', 'analyze the trend', or 'how confident are you'"
            ),
            Tool(
                name="compare_scenarios",
                func=compare_scenarios_tool,
                description="Compares what-if scenarios with baseline forecast. Use when user asks 'what if', 'compare scenarios', 'impact of changes', or wants to understand different assumptions"
            ),
            Tool(
                name="explain_model_choice",
                func=explain_model_choice_tool,
                description="Explains why specific model was chosen and its characteristics. Use when user asks 'why this model', 'tell me about the model', 'what other models could work', or 'model capabilities'"
            ),
            Tool(
                name="suggest_actions",
                func=suggest_actions_tool,
                description="Suggests actionable next steps based on forecast. Use when user asks 'what should I do', 'next steps', 'recommendations', 'how to use this forecast', or mentions specific contexts like inventory, budget, capacity"
            )
        ]
        
        return tools

    def _create_dataset_info(self, include_dataframe: bool = False) -> Dict[str, Any]:
        """
        Create dataset info dict from uploaded DataFrame
        
        Args:
            include_dataframe: If True, includes actual DataFrame (for processing)
                            If False, excludes it (for JSON serialization)
        """
        if self.current_dataset_df is None:
            return None
        
        df = self.current_dataset_df
        
        dataset_info = {
            'name': 'user_uploaded_dataset',
            'columns': list(df.columns),
            'row_count': len(df),
            'frequency': self._detect_frequency(),
            'missing_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'description': f'User uploaded dataset with {len(df)} rows and {len(df.columns)} columns'
        }
        
        # Only include DataFrame when explicitly needed (for processing, not for JSON)
        if include_dataframe:
            dataset_info['data'] = df
        
        return dataset_info



    def _load_vectorstore(self):
        """Load the pre-built vector store"""
        if not Path(self.vector_store_path).exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Run rag_setup.py first!"
            )
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        return FAISS.load_local(
            self.vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )


    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with RAG context"""
        
        template = """You are an AI forecasting assistant helping users analyze data and create forecasts.

IMPORTANT: The user has ALREADY uploaded a dataset. You can directly use the tools without asking them to upload again.

Available tools:
{tools}

Use this EXACT format for EVERY response:

Thought: [Think about what to do - be concise]
Action: [Choose ONE tool name from the list above]
Action Input: [The input for that tool]
Observation: [The tool's response will appear here]

... (repeat Thought/Action/Action Input/Observation as needed)

Thought: I now have the complete information to answer the user
Final Answer: [Your complete, helpful response to the user]

CRITICAL RULES:
1. ALWAYS use a tool from the list - NEVER use "None" or skip Action
2. After getting the information you need, provide Final Answer immediately
3. DO NOT repeat the same action multiple times
4. For dataset analysis, use "analyze_dataset" once
5. For forecasting, use "run_forecast" with number of days
6. Keep responses conversational and helpful

Previous conversation:
{chat_history}

User question: {input}

Begin!

Thought: {agent_scratchpad}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([
                    f"- {tool.name}: {tool.description}" 
                    for tool in self.tools
                ]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input"
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=60,
            return_intermediate_steps=False
        )
        
        return agent_executor


    def create_session(self, session_id: str = None):
        """Create a new session"""
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        self._session_data[session_id] = {
            'dataset_df': None,
            'result': None,
            'created_at': pd.Timestamp.now()
        }
        self._current_session_id = session_id
        logger.info(f"📝 Created session: {session_id}")
        return session_id

    def set_current_session(self, session_id: str):
        """Set current session"""
        if session_id in self._session_data:
            self._current_session_id = session_id
            logger.info(f"📝 Switched to session: {session_id}")
        else:
            logger.warning(f"Session {session_id} not found")

    def upload_dataset(self, df: pd.DataFrame, session_id: str = None):
        """Upload dataset to specific session"""
        if session_id is None:
            session_id = self._current_session_id
            if session_id is None:
                session_id = self.create_session()
        
        if session_id not in self._session_data:
            self.create_session(session_id)
        
        self._session_data[session_id]['dataset_df'] = df
        self._session_data[session_id]['result'] = None
        self._current_session_id = session_id
        
        logger.info(f"✅ Dataset uploaded to session {session_id}: {len(df)} rows × {len(df.columns)} columns")

    def _get_current_dataset(self):
        """Safely get current dataset"""
        if (self._current_session_id and 
            self._current_session_id in self._session_data):
            return self._session_data[self._current_session_id]['dataset_df']
        return None

    def _get_current_result(self):
        """Safely get current result"""
        if (self._current_session_id and 
            self._current_session_id in self._session_data):
            return self._session_data[self._current_session_id]['result']
        return None

    def _set_current_result(self, result):
        """Safely set current result"""
        if (self._current_session_id and 
            self._current_session_id in self._session_data):
            self._session_data[self._current_session_id]['result'] = result


    def upload_dataset(self, df: pd.DataFrame):
        """Upload dataset"""
        try:
            self.current_dataset_df = df
            self.current_result = None
            logger.info(f"✅ Dataset uploaded: {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"❌ Dataset upload failed: {e}")
            raise
    
    def _detect_intent(self, message: str) -> Dict[str, Any]:
        """Detect user intent - ENHANCED with query/analysis detection"""
        message_lower = message.lower()
        
        # ✅ NEW: Check if user is querying/analyzing existing forecast results
        # This should be checked BEFORE "forecast" intent to avoid generating new forecasts
        # Check both current_result and session-based storage
        current_result = self.current_result or self._get_current_result()
        has_forecast_result = (
            current_result is not None and 
            'forecast' in current_result and 
            current_result.get('forecast', {}).get('values')
        )
        
        # Query/analysis keywords that indicate user wants to analyze existing forecast
        query_keywords = [
            'which', 'what', 'when', 'where', 'how many', 'how much',
            'highest', 'lowest', 'maximum', 'minimum', 'max', 'min',
            'peak', 'best', 'worst', 'top', 'bottom',
            'from the forecast', 'in the forecast', 'of the forecast',
            'from forecast', 'in forecast', 'of forecast',
            'show me', 'tell me', 'find', 'identify'
        ]
        
        # If we have a forecast result and message contains query keywords, it's a query
        if has_forecast_result and any(kw in message_lower for kw in query_keywords):
            return {'intent': 'query_forecast', 'params': {'query': message}}
        
        # Priority order matters!
        if any(kw in message_lower for kw in ['analyze', 'check', 'validate', 'quality']):
            return {'intent': 'analyze', 'params': {}}
        
        # Only trigger new forecast if explicitly requested
        # Check if user wants a NEW forecast (not querying existing one)
        wants_new_forecast = (
            any(kw in message_lower for kw in ['forecast', 'predict', 'generate', 'create']) and
            not any(kw in message_lower for kw in ['from the', 'in the', 'of the', 'from forecast', 'in forecast'])
        )
        
        if wants_new_forecast:
            import re
            numbers = re.findall(r'\d+', message)
            
            # Determine horizon
            if 'month' in message_lower:
                horizon = int(numbers[0]) * 30 if numbers else 30
            elif 'quarter' in message_lower:
                horizon = int(numbers[0]) * 90 if numbers else 90
            elif 'week' in message_lower:
                horizon = int(numbers[0]) * 7 if numbers else 7
            elif 'year' in message_lower:
                horizon = int(numbers[0]) * 365 if numbers else 365
            elif numbers:
                horizon = int(numbers[0])
            else:
                horizon = 30
            
            return {'intent': 'forecast', 'params': {'horizon': horizon}}
        
        if any(kw in message_lower for kw in ['interpret', 'explain', 'insight', 'meaning']):
            return {'intent': 'interpret', 'params': {}}
        
        if any(kw in message_lower for kw in ['data', 'values', 'numbers', 'results']):
            return {'intent': 'get_data', 'params': {}}
        
        if any(kw in message_lower for kw in ['model', 'algorithm', 'why']):
            return {'intent': 'model_info', 'params': {}}
        
        # Default to question
        return {'intent': 'question', 'params': {}}
    
    def _analyze_dataset(self) -> str:
        """Analyze dataset - WITH ERROR HANDLING"""
        if self.current_dataset_df is None:
            return "❌ **No dataset uploaded.** Please upload data first."
        
        try:
            logger.info("🔍 Starting dataset analysis...")
            
            # Create dataset info
            dataset_info = {
                'name': 'user_uploaded_dataset',
                'columns': list(self.current_dataset_df.columns),
                'row_count': len(self.current_dataset_df),
                'frequency': self._detect_frequency(),
                'missing_percentage': float(
                    self.current_dataset_df.isnull().sum().sum() / 
                    (len(self.current_dataset_df) * len(self.current_dataset_df.columns))
                )
            }
            
            logger.info(f"📊 Dataset info created: {dataset_info['row_count']} rows")
            
            # Run analysis
            analysis = self.forecasting_service.analyze_dataset_with_knowledge_base(dataset_info)
            self.current_result = analysis
            
            logger.info("✅ Analysis complete")
            
            # Format response
            validation = analysis.get('rule_analysis', {}).get('validation', {})
            is_valid = validation.get('valid', False)
            
            if is_valid:
                compatible_models = analysis.get('rule_analysis', {}).get('selection_analysis', {}).get('analysis', {}).get('compatible_models', [])
                selected_model = analysis.get('rule_analysis', {}).get('model_selection', {}).get('selected_model', 'Unknown')
                
                response = f"""✅ **Dataset Analysis Complete**

📊 **Dataset Summary:**
- **Rows:** {dataset_info['row_count']:,}
- **Columns:** {len(dataset_info['columns'])}
- **Frequency:** {dataset_info['frequency']}
- **Missing Data:** {dataset_info['missing_percentage']:.1%}

✅ **Validation:** Passed

🤖 **Compatible Models:** {len(compatible_models)} found
- **Recommended:** {selected_model}

💡 **Next Steps:**
Ready to forecast! Ask me to "forecast next 30 days" or specify your horizon.
"""
            else:
                errors = validation.get('errors', ['Unknown validation error'])
                warnings = validation.get('warnings', [])
                
                response = f"""⚠️ **Dataset Analysis - Issues Found**

❌ **Validation Errors:**
{chr(10).join(f'  • {e}' for e in errors)}

⚠️ **Warnings:**
{chr(10).join(f'  • {w}' for w in warnings) if warnings else '  • None'}

💡 **Recommendations:**
Please fix the errors above before proceeding with forecasting.
"""
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            logger.error(traceback.format_exc())
            return f"❌ **Analysis failed:** {str(e)}\n\nPlease check your dataset format and try again."
    
    def _run_forecast(self, horizon: int) -> str:
        """Run forecast - WITH ERROR HANDLING"""
        if self.current_dataset_df is None:
            return "❌ **No dataset uploaded.** Please upload data first."
        
        try:
            # 🆕 Use the actual session ID from the agent
            session_id = getattr(self, 'session_id', 'unknown_session')
            dataset_name = f"temp_session_{session_id}"
            
            dataset_info = {
                'name': dataset_name,  # This will match the registered schema!
                'columns': list(self.current_dataset_df.columns),
                'row_count': len(self.current_dataset_df),
                'frequency': self._detect_frequency(),
                'data': self.current_dataset_df
            }
            
                
            logger.info("📊 Dataset info created with DataFrame")
            
            # Run forecast
            result = self.forecasting_service.process_forecasting_request(
                dataset_info=dataset_info,
                forecast_horizon=horizon,
                business_context={'source': 'chatbot_interface'}
            )
            
            self.current_result = result
            logger.info(f"✅ Forecast complete: {result.get('status')}")
            
            if result.get('status') == 'success':
                forecast = result.get('forecast', {})
                model = result.get('selected_model', {})
                
                # Generate interpretation (with fallback)
                if self.has_interpretation:
                    try:
                        logger.info("📊 Generating interpretation...")
                        interpretation = self.interpretation_service.interpret_forecast(
                            forecast_result=forecast,
                            analysis_result=result,
                            business_context={'type': 'general'}
                        )
                        result['interpretation'] = interpretation
                        logger.info("✅ Interpretation generated")
                    except Exception as e:
                        logger.warning(f"⚠️ Interpretation failed: {e}")
                        result['interpretation'] = {
                            'key_insights': ['Forecast generated successfully'],
                            'summary': 'Forecast completed. Detailed interpretation unavailable.'
                        }
                else:
                    result['interpretation'] = {
                        'key_insights': ['Forecast generated successfully'],
                        'summary': 'Forecast completed. Interpretation service not available.'
                    }
                
                # Format response
                values = forecast.get('values', [])
                if values:
                    avg_value = np.mean(values)
                    total = sum(values)
                    
                    interpretation = result.get('interpretation', {})
                    insights = interpretation.get('key_insights', ['Forecast generated'])
                    
                    response = f"""✅ **Forecast Complete**

📈 **Forecast Summary:**
- **Horizon:** {horizon} days
- **Model Used:** {model.get('model_name', 'Unknown')}
- **Confidence:** {model.get('confidence', 0)*100:.1f}%

📊 **Predictions:**
- **Average:** {avg_value:.2f} units/day
- **Total:** {total:.2f} units
- **Range:** {min(values):.2f} - {max(values):.2f}

💡 **Key Insights:**
{chr(10).join(f'  • {insight}' for insight in insights[:3])}

📊 Visualization is displayed below.
"""
                else:
                    response = "⚠️ **Forecast completed but no values generated.** Please check your data."
                
                return response
            else:
                error = result.get('error', 'Unknown error')
                return f"❌ **Forecast failed:** {error}\n\nPlease check your dataset and try again."
                
        except Exception as e:
            logger.error(f"❌ Forecast failed: {e}")
            logger.error(traceback.format_exc())
            return f"❌ **Forecast failed:** {str(e)}\n\nPlease check your dataset format and try again."
    
    
    def _interpret_results(self) -> str:
        """Interpret results"""
        if self.current_result is None or 'forecast' not in self.current_result:
            return "❌ **No forecast results available.** Please run a forecast first."
        
        try:
            if 'interpretation' in self.current_result:
                interp = self.current_result['interpretation']
                
                response = f"""📊 **Forecast Interpretation**

{interp.get('summary', 'Forecast completed successfully.')}

💡 **Key Insights:**
{chr(10).join(f'  {i+1}. {insight}' for i, insight in enumerate(interp.get('key_insights', ['No insights available'])))}

🎯 **Business Recommendations:**
{chr(10).join(f'  • {rec}' for rec in interp.get('business_implications', ['Monitor forecast accuracy'])[:3])}

⚠️ **Risk Factors:**
{chr(10).join(f'  • {risk}' for risk in interp.get('risk_factors', ['Market volatility'])[:3])}
"""
                return response
            else:
                return "ℹ️ **Forecast completed successfully.** Detailed interpretation not available."
        except Exception as e:
            logger.error(f"Interpretation error: {e}")
            return "ℹ️ **Forecast completed successfully.** Interpretation temporarily unavailable."
    
    def _get_forecast_data(self) -> str:
        """Get forecast data"""
        if self.current_result is None or 'forecast' not in self.current_result:
            return "❌ **No forecast data available.** Please run a forecast first."
        
        try:
            forecast = self.current_result['forecast']
            values = forecast.get('values', [])
            
            if not values:
                return "❌ **No forecast values found.**"
            
            response = f"""📊 **Forecast Values**

**First 10 Values:**
{chr(10).join(f'  Day {i+1}: {v:.2f}' for i, v in enumerate(values[:10]))}

**Summary Statistics:**
- **Total Points:** {len(values)}
- **Mean:** {np.mean(values):.2f}
- **Min:** {min(values):.2f}
- **Max:** {max(values):.2f}
- **Std Dev:** {np.std(values):.2f}

💡 Full data available in visualization.
"""
            return response
        except Exception as e:
            logger.error(f"Get data error: {e}")
            return "❌ **Error retrieving forecast data.**"
    
    def _query_forecast_result(self, query: str) -> str:
        """Query and analyze the last forecast result instead of generating a new forecast"""
        # Check both current_result and session-based storage
        current_result = self.current_result or self._get_current_result()
        
        if current_result is None or 'forecast' not in current_result:
            return "❌ **No forecast results available.** Please run a forecast first."
        
        try:
            forecast = current_result['forecast']
            values = forecast.get('values', [])
            
            if not values:
                return "❌ **No forecast values found.**"
            
            query_lower = query.lower()
            response_parts = []
            
            # Analyze the query and provide relevant insights
            # Find highest/lowest values
            if any(kw in query_lower for kw in ['highest', 'maximum', 'max', 'peak', 'best', 'top']):
                max_idx = np.argmax(values)
                max_value = values[max_idx]
                day_number = max_idx + 1
                
                response_parts.append(f"📈 **Highest Forecast Value:**")
                response_parts.append(f"- **Day {day_number}:** {max_value:.2f} units")
                
                # Add context if available
                if 'confidence_intervals' in forecast and forecast['confidence_intervals']:
                    ci = forecast['confidence_intervals']
                    if ci.get('upper') and len(ci['upper']) > max_idx:
                        response_parts.append(f"- **Upper bound:** {ci['upper'][max_idx]:.2f} units")
                
            elif any(kw in query_lower for kw in ['lowest', 'minimum', 'min', 'worst', 'bottom']):
                min_idx = np.argmin(values)
                min_value = values[min_idx]
                day_number = min_idx + 1
                
                response_parts.append(f"📉 **Lowest Forecast Value:**")
                response_parts.append(f"- **Day {day_number}:** {min_value:.2f} units")
                
                # Add context if available
                if 'confidence_intervals' in forecast and forecast['confidence_intervals']:
                    ci = forecast['confidence_intervals']
                    if ci.get('lower') and len(ci['lower']) > min_idx:
                        response_parts.append(f"- **Lower bound:** {ci['lower'][min_idx]:.2f} units")
            
            # Find specific day or range
            elif 'day' in query_lower:
                import re
                day_numbers = re.findall(r'\d+', query)
                if day_numbers:
                    day_num = int(day_numbers[0])
                    if 1 <= day_num <= len(values):
                        day_value = values[day_num - 1]
                        response_parts.append(f"📅 **Day {day_num} Forecast:**")
                        response_parts.append(f"- **Value:** {day_value:.2f} units")
                        
                        if 'confidence_intervals' in forecast and forecast['confidence_intervals']:
                            ci = forecast['confidence_intervals']
                            if ci.get('lower') and ci.get('upper') and len(ci['lower']) > day_num - 1:
                                response_parts.append(f"- **Range:** {ci['lower'][day_num-1]:.2f} - {ci['upper'][day_num-1]:.2f} units")
                    else:
                        response_parts.append(f"⚠️ Day {day_num} is out of range. Forecast has {len(values)} days.")
            
            # General statistics
            else:
                response_parts.append(f"📊 **Forecast Analysis:**")
                response_parts.append(f"- **Total Days:** {len(values)}")
                response_parts.append(f"- **Average:** {np.mean(values):.2f} units/day")
                response_parts.append(f"- **Total:** {sum(values):.2f} units")
                response_parts.append(f"- **Range:** {min(values):.2f} - {max(values):.2f} units")
                
                # Find peak and trough
                max_idx = np.argmax(values)
                min_idx = np.argmin(values)
                response_parts.append(f"- **Peak (Day {max_idx + 1}):** {values[max_idx]:.2f} units")
                response_parts.append(f"- **Lowest (Day {min_idx + 1}):** {values[min_idx]:.2f} units")
            
            # Add trend information
            if len(values) > 1:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                trend_change = ((np.mean(second_half) - np.mean(first_half)) / np.mean(first_half)) * 100
                
                if abs(trend_change) > 1:
                    trend_desc = "increasing" if trend_change > 0 else "decreasing"
                    response_parts.append(f"\n📈 **Trend:** {trend_desc} by {abs(trend_change):.1f}% over the forecast period")
            
            # Add model information
            if 'selected_model' in current_result:
                model = current_result['selected_model']
                response_parts.append(f"\n🤖 **Model:** {model.get('model_name', 'Unknown')}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Query forecast error: {e}")
            logger.error(traceback.format_exc())
            return f"❌ **Error analyzing forecast:** {str(e)}"
    
    def _get_model_info(self) -> str:
        """Get model information"""
        if self.current_result is None:
            return "❌ **No analysis or forecast results available.**"
        
        try:
            if 'selected_model' in self.current_result:
                model = self.current_result['selected_model']
                
                response = f"""🤖 **Model Information**

**Selected Model:** {model.get('model_name', 'Unknown')}
**Confidence:** {model.get('confidence', 0)*100:.1f}%
**Selection Reason:** {model.get('reason', 'Best match for your data')}
"""
                
                # Add RAG context if available
                if self.has_rag:
                    try:
                        docs = self.vectorstore.similarity_search(
                            f"Explain {model['model_name']} model",
                            k=1
                        )
                        if docs:
                            response += f"\n**About {model['model_name']}:**\n{docs[0].page_content[:500]}..."
                    except:
                        pass
                
                return response
            else:
                return "ℹ️ **No model information available yet.** Run analysis or forecast first."
        except Exception as e:
            logger.error(f"Model info error: {e}")
            return "ℹ️ **Model information temporarily unavailable.**"
    
    def _answer_question(self, question: str) -> str:
        """Answer general questions"""
        if not self.has_rag or not self.llm:
            return """ℹ️ I can help with:
• Dataset analysis
• Forecasting
• Result interpretation

Please ask about those topics."""
        
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            # Search knowledge base
            docs = self.vectorstore.similarity_search(question, k=2)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Use LLM to answer
            messages = [
                SystemMessage(content="You are a helpful forecasting assistant. Answer based on context."),
                HumanMessage(content=f"Context:\n{context[:1000]}\n\nQuestion: {question}\n\nAnswer:")
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Question answering error: {e}")
            return "I'm having trouble answering that. Please ask about dataset analysis or forecasting."
    
    def _detect_frequency(self) -> str:
        """Detect data frequency"""
        if self.current_dataset_df is None:
            return "unknown"
        
        df = self.current_dataset_df
        date_col = None
        
        for col in df.columns:
            if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
                date_col = col
                break
        
        if date_col is None:
            return "unknown"
        
        try:
            dates = pd.to_datetime(df[date_col]).sort_values()
            if len(dates) < 2:
                return "unknown"
            
            time_diffs = dates.diff().dropna()
            most_common_diff = time_diffs.mode().iloc[0]
            days = most_common_diff.total_seconds() / (24 * 3600)
            
            if 0.9 <= days <= 1.1:
                return "daily"
            elif 6.5 <= days <= 7.5:
                return "weekly"
            elif 28 <= days <= 31:
                return "monthly"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def ask_question(self, message: str) -> str:
        """
        Main entry point - handles all user messages
        """
        try:
            logger.info(f"💬 Processing message: {message[:50]}...")
            
            # Detect intent
            intent_result = self._detect_intent(message)
            intent = intent_result['intent']
            params = intent_result['params']
            
            logger.info(f"🎯 Detected intent: {intent}")
            
            # Route to handler
            if intent == 'query_forecast':
                # ✅ NEW: Query existing forecast results instead of generating new forecast
                response = self._query_forecast_result(params.get('query', message))
            elif intent == 'analyze':
                response = self._analyze_dataset()
            elif intent == 'forecast':
                response = self._run_forecast(params['horizon'])
            elif intent == 'interpret':
                response = self._interpret_results()
            elif intent == 'get_data':
                response = self._get_forecast_data()
            elif intent == 'model_info':
                response = self._get_model_info()
            elif intent == 'question':
                response = self._answer_question(message)
            else:
                response = """I can help you with:
• **Dataset analysis** - "analyze my dataset"
• **Forecasting** - "forecast next 30 days"
• **Result interpretation** - "interpret the results"
• **Query forecast** - "which day has the highest sales?"

What would you like to do?"""
            
            # Store in history
            self.conversation_history.append({
                'user': message,
                'assistant': response,
                'intent': intent
            })
            
            logger.info("✅ Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            logger.error(traceback.format_exc())
            return f"❌ **Sorry, I encountered an error:** {str(e)}\n\nPlease try again or rephrase your question."
    
    def get_current_forecast_data(self) -> Optional[Dict]:
        """Get forecast data for visualization"""
        try:
            if self.current_result and 'forecast' in self.current_result:
                return self.current_result['forecast']
        except:
            pass
        return None
    
    def close(self):
        """Cleanup"""
        try:
            if hasattr(self, 'forecasting_service'):
                self.forecasting_service.close()
        except:
            pass


# Test function
if __name__ == "__main__":
    print("🧪 Testing Simplified Forecast Agent...\n")
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not set!")
        exit(1)
    
    try:
        agent = SimplifiedForecastAgent(nvidia_api_key=api_key)
        print("✅ Agent initialized successfully\n")
        
        # Test with sample data
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'demand': [100 + i + (i % 7) * 10 for i in range(100)]
        })
        
        agent.upload_dataset(df)
        print("✅ Dataset uploaded\n")
        
        print("="*60)
        print("TEST 1: Analyze Dataset")
        print("="*60)
        print(agent.ask_question("Analyze my dataset"))
        
        print("\n" + "="*60)
        print("TEST 2: Forecast")
        print("="*60)
        print(agent.ask_question("Forecast next 30 days"))
        
        print("\n✅ All tests passed!")
        
        agent.close()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(traceback.format_exc())