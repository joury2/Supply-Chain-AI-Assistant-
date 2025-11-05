# app/services/llm/forecast_agent.py
# Forecast Agent with RAG and Function Calling

"""
Forecast Agent with RAG and Function Calling
Uses LangChain + NVIDIA Llama 3.1 + SupplyChainForecastingService
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

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

# Set up logging
logger = logging.getLogger(__name__)

class ForecastAgent:
    """
    RAG-powered forecasting agent with function calling
    Integrates with your SupplyChainForecastingService
    """

    def __init__(
        self, 
        nvidia_api_key: str,
        vector_store_path: str = "vector_store",
        db_path: str = "supply_chain.db"
    ):
        self.nvidia_api_key = nvidia_api_key
        self.vector_store_path = vector_store_path
        self.db_path = db_path
        
        # Initialize YOUR actual service
        print("🔧 Initializing Supply Chain Forecasting Service...")
        self.forecasting_service = SupplyChainForecastingService(db_path=db_path)
        
        # Current session data
        self.current_dataset_df = None  # The pandas DataFrame
        self.current_dataset_info = None  # The dataset info dict
        self.current_result = None  # Last result from service
        
        # Initialize RAG
        print("📚 Loading RAG vector store...")
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Initialize LLM
        print("🤖 Connecting to NVIDIA Llama 3.1...")
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            api_key=nvidia_api_key,
            temperature=0.7,
            max_completion_tokens=1024,
            stop=["Observation:", "\nObservation"]
        )
        
        # Initialize interpretation service
        print("📊 Initializing Interpretation Service...")
        self.interpretation_service = LLMInterpretationService(
            nvidia_api_key=self.nvidia_api_key,
            vector_store_path=self.vector_store_path
        )
        
        # Setup agent
        print("🎯 Setting up agent with tools...")
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

        print("✅ Forecast Agent initialized!\n")
    
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

    def _detect_frequency(self) -> str:
        """Detect the frequency of the time series data"""
        if self.current_dataset_df is None:
            return "unknown"
        
        df = self.current_dataset_df
        
        # Look for date column more flexibly
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
            
            # Calculate most common difference
            time_diffs = dates.diff().dropna()
            most_common_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.iloc[0]
            
            days = most_common_diff.total_seconds() / (24 * 3600)
            
            if 0.9 <= days <= 1.1:
                return "daily"
            elif 6.5 <= days <= 7.5:
                return "weekly"
            elif 28 <= days <= 31:
                return "monthly"
            elif 89 <= days <= 92:
                return "quarterly"
            elif 365 <= days <= 366:
                return "yearly"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Frequency detection failed: {e}")
            return "unknown"

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

    # ... [Keep all the other methods the same as before] ...

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

    def upload_dataset(self, df: pd.DataFrame):
        """Upload a dataset for analysis"""
        self.current_dataset_df = df
        self.current_dataset_info = self._create_dataset_info()
        self.current_result = None
        print(f"✅ Dataset uploaded: {len(df)} rows, {len(df.columns)} columns")
    
    def ask_question(self, user_message: str) -> str:
        """Process user message with RAG context"""
        try:
            # Retrieve relevant context from RAG
            rag_docs = self.retriever.get_relevant_documents(user_message)
            rag_context = "\n\n".join([doc.page_content for doc in rag_docs[:3]])
            
            # Enhance user message with RAG context
            enhanced_message = f"""User question: {user_message}

Relevant knowledge base context:
{rag_context[:1000]}  

Please use this context along with available tools to help the user."""
            
            # Run agent
            response = self.agent_executor.invoke({"input": enhanced_message})
            return response['output']
            
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return f"Sorry, I encountered an error: {str(e)}\n\nPlease try again or rephrase your question."
    
    def get_current_forecast_data(self) -> Optional[Dict]:
        """Get current forecast data for visualization"""
        if self.current_result and 'forecast' in self.current_result:
            return self.current_result['forecast']
        return None
    
    def get_interpretation_data(self) -> Optional[Dict]:
        """Get current interpretation data"""
        if self.current_result and 'interpretation' in self.current_result:
            return self.current_result['interpretation']
        return None

    def close(self):
        """Cleanup resources"""
        self.forecasting_service.close()



# Example usage
if __name__ == "__main__":
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        print("⚠️  Set NVIDIA_API_KEY environment variable!")
        print("Get your key from: https://build.nvidia.com/")
        exit(1)
    
    # Initialize agent
    agent = ForecastAgent(nvidia_api_key=nvidia_api_key)
    
    # Example: Upload dataset
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'demand': [100 + i + (i % 7) * 10 for i in range(100)],
        'region': ['North'] * 100
    })
    agent.upload_dataset(sample_data)
    
    # Example conversations
    print("\n" + "="*60)
    print("💬 Example Conversation")
    print("="*60 + "\n")
    
    queries = [
        "Analyze my dataset",
        "What models work with my data?",
        "Forecast the next 30 days"
    ]
    
    for query in queries:
        print(f"\n👤 User: {query}")
        response = agent.ask_question(query)
        print(f"\n🤖 Agent: {response}")
        print("\n" + "-"*60)
    
    agent.close()