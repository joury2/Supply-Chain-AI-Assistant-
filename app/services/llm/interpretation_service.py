# app/services/llm/interpretation_service.py
"""
Enhanced LLM Interpretation Service
Provides rich, contextual interpretations of forecast results using NVIDIA Llama 3.1 + RAG
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)


class LLMInterpretationService:
    """
    Advanced LLM-based forecast interpretation service
    Uses RAG to provide domain-aware insights and recommendations
    """
    
    def __init__(
        self,
        nvidia_api_key: Optional[str] = None,
        vector_store_path: str = "vector_store",
        use_rag: bool = True
    ):
        """
        Initialize the interpretation service
        
        Args:
            nvidia_api_key: NVIDIA API key (defaults to env var)
            vector_store_path: Path to RAG vector store
            use_rag: Whether to use RAG for enhanced interpretations
        """
        self.nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        self.vector_store_path = vector_store_path
        self.use_rag = use_rag
        
        if not self.nvidia_api_key:
            logger.warning("⚠️ No NVIDIA API key provided - using simplified interpretations")
            self.llm = None
            self.retriever = None
        else:
            # Initialize LLM
            logger.info("🤖 Initializing NVIDIA Llama 3.1...")
            self.llm = ChatNVIDIA(
                model="meta/llama-3.1-70b-instruct",
                api_key=self.nvidia_api_key,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Initialize RAG retriever if enabled
            if self.use_rag and Path(vector_store_path).exists():
                logger.info("📚 Loading RAG knowledge base...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                vectorstore = FAISS.load_local(
                    vector_store_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                logger.info("✅ RAG retriever ready")
            else:
                self.retriever = None
                logger.info("ℹ️ RAG disabled or vector store not found")
        
        logger.info("✅ LLM Interpretation Service initialized")
    
    def _get_rag_context(self, query: str) -> str:
        """Retrieve relevant context from RAG knowledge base"""
        if not self.retriever:
            return ""
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs[:2]])
            return context[:1500]  # Limit context size
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""
    
    def _analyze_forecast_patterns(self, forecast_values: List[float]) -> Dict[str, Any]:
        """Analyze statistical patterns in forecast"""
        if not forecast_values:
            return {}
        
        values = np.array(forecast_values)
        
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Trend analysis
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        trend_change = ((np.mean(second_half) - np.mean(first_half)) / np.mean(first_half)) * 100
        
        if trend_change > 5:
            trend = "increasing"
            trend_desc = f"upward trend of {trend_change:.1f}%"
        elif trend_change < -5:
            trend = "decreasing"
            trend_desc = f"downward trend of {abs(trend_change):.1f}%"
        else:
            trend = "stable"
            trend_desc = "relatively stable pattern"
        
        # Volatility
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        if cv < 10:
            volatility = "low"
        elif cv < 25:
            volatility = "moderate"
        else:
            volatility = "high"
        
        # Detect anomalies
        z_scores = np.abs((values - mean_val) / std_val) if std_val > 0 else np.zeros_like(values)
        anomalies = np.where(z_scores > 2)[0].tolist()
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val),
            'range': float(max_val - min_val),
            'trend': trend,
            'trend_change_pct': float(trend_change),
            'trend_description': trend_desc,
            'volatility': volatility,
            'coefficient_of_variation': float(cv),
            'anomaly_count': len(anomalies),
            'anomaly_periods': anomalies[:5]  # First 5 anomalies
        }
    
    def _get_confidence_interpretation(self, confidence: float) -> Dict[str, str]:
        """Interpret confidence level"""
        if confidence >= 0.8:
            return {
                'level': 'high',
                'description': 'very reliable',
                'recommendation': 'Can be used confidently for critical planning decisions'
            }
        elif confidence >= 0.6:
            return {
                'level': 'medium',
                'description': 'reasonably reliable',
                'recommendation': 'Suitable for operational planning with regular monitoring'
            }
        elif confidence >= 0.4:
            return {
                'level': 'moderate',
                'description': 'use with caution',
                'recommendation': 'Consider as one input among multiple factors'
            }
        else:
            return {
                'level': 'low',
                'description': 'limited reliability',
                'recommendation': 'Use only for preliminary insights; seek additional data'
            }
    
    def _generate_llm_interpretation(
        self,
        forecast_data: Dict[str, Any],
        analysis_data: Dict[str, Any],
        patterns: Dict[str, Any],
        business_context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate rich interpretation using LLM"""
        if not self.llm:
            return self._generate_fallback_interpretation(patterns)
        
        # Get RAG context
        rag_context = ""
        if self.retriever:
            model_name = analysis_data.get('selected_model', {}).get('model_name', 'Unknown')
            query = f"Explain {model_name} forecast results interpretation and business implications"
            rag_context = self._get_rag_context(query)
        
        # Build prompt
        prompt_template = PromptTemplate(
            input_variables=["forecast_summary", "patterns", "confidence", "business_context", "rag_context"],
            template="""You are an expert supply chain forecasting analyst. Interpret these forecast results for business stakeholders.

Forecast Summary:
{forecast_summary}

Statistical Patterns:
{patterns}

Model Confidence: {confidence}

Business Context:
{business_context}

Domain Knowledge:
{rag_context}

Provide a clear, actionable interpretation covering:
1. What the forecast shows (2-3 sentences)
2. Key patterns and trends (2-3 sentences)
3. Business implications (2-3 bullet points)
4. Risk factors to monitor (2-3 items)
5. Recommended actions (2-3 concrete steps)

Write in a professional but accessible tone. Focus on actionable insights.

Interpretation:"""
        )
        
        # Prepare inputs
        forecast_values = forecast_data.get('values', [])
        horizon = len(forecast_values)
        
        forecast_summary = f"""
Forecast Horizon: {horizon} periods
Average Forecasted Value: {patterns['mean']:.2f}
Total Forecasted Demand: {sum(forecast_values):.2f}
Range: {patterns['min']:.2f} to {patterns['max']:.2f}
Trend: {patterns['trend_description']}
"""
        
        patterns_text = f"""
Volatility: {patterns['volatility']} (CV: {patterns['coefficient_of_variation']:.1f}%)
Anomalies Detected: {patterns['anomaly_count']}
Trend: {patterns['trend']} ({patterns['trend_change_pct']:+.1f}% change)
"""
        
        confidence = analysis_data.get('combined_summary', {}).get('confidence', 0.5)
        
        context_text = json.dumps(business_context or {'source': 'general_forecast'}, indent=2)
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = chain.run(
                forecast_summary=forecast_summary,
                patterns=patterns_text,
                confidence=f"{confidence:.1%}",
                business_context=context_text,
                rag_context=rag_context[:800]
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return self._generate_fallback_interpretation(patterns)
    
    def _generate_fallback_interpretation(self, patterns: Dict[str, Any]) -> str:
        """Generate interpretation without LLM"""
        return f"""
Forecast Analysis Summary:

The forecast shows a {patterns['trend_description']} over the prediction period, 
with average values around {patterns['mean']:.1f} units. 

Volatility is {patterns['volatility']}, indicating {'stable and predictable' if patterns['volatility'] == 'low' else 'moderate variation' if patterns['volatility'] == 'moderate' else 'significant fluctuation'} in the forecasted values.

Key Recommendations:
• Monitor actual performance against forecasts regularly
• Maintain appropriate safety stock based on volatility
• Review forecast accuracy and adjust planning as needed
"""
    
    def interpret_forecast(
        self,
        forecast_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        business_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation of forecast results
        
        Args:
            forecast_result: Forecast output from forecasting service
            analysis_result: Analysis output including model selection
            business_context: Optional business context (inventory, budget, etc.)
        
        Returns:
            Detailed interpretation with insights and recommendations
        """
        logger.info("🔍 Generating forecast interpretation...")
        
        # Extract data
        forecast_values = forecast_result.get('values', [])
        confidence = analysis_result.get('combined_summary', {}).get('confidence', 0.5)
        selected_model = analysis_result.get('selected_model', {})
        
        if not forecast_values:
            logger.warning("No forecast values to interpret")
            return {
                'error': 'No forecast values available',
                'summary': 'Unable to generate interpretation'
            }
        
        # Analyze patterns
        patterns = self._analyze_forecast_patterns(forecast_values)
        
        # Get confidence interpretation
        confidence_interp = self._get_confidence_interpretation(confidence)
        
        # Generate LLM interpretation
        detailed_interpretation = self._generate_llm_interpretation(
            forecast_result,
            analysis_result,
            patterns,
            business_context
        )
        
        # Build structured output
        interpretation = {
            'summary': detailed_interpretation,
            'confidence_assessment': {
                'level': confidence_interp['level'],
                'score': float(confidence),
                'description': confidence_interp['description'],
                'recommendation': confidence_interp['recommendation']
            },
            'forecast_statistics': {
                'horizon_periods': len(forecast_values),
                'average_value': patterns['mean'],
                'total_forecasted': sum(forecast_values),
                'min_value': patterns['min'],
                'max_value': patterns['max'],
                'volatility': patterns['volatility'],
                'trend': patterns['trend']
            },
            'key_insights': self._generate_key_insights(patterns, confidence_interp),
            'business_implications': self._generate_business_implications(
                patterns,
                confidence_interp,
                business_context
            ),
            'risk_factors': self._identify_risk_factors(patterns, confidence),
            'recommended_actions': self._generate_recommendations(
                patterns,
                confidence_interp,
                business_context
            ),
            'model_information': {
                'model_name': selected_model.get('model_name', 'Unknown'),
                'model_confidence': selected_model.get('confidence', 0),
                'selection_reason': selected_model.get('reason', 'Not specified')
            }
        }
        
        logger.info("✅ Interpretation generated successfully")
        return interpretation
    
    def _generate_key_insights(
        self,
        patterns: Dict[str, Any],
        confidence: Dict[str, str]
    ) -> List[str]:
        """Generate key insights from patterns"""
        insights = []
        
        # Trend insight
        if patterns['trend'] == 'increasing':
            insights.append(
                f"Forecast shows {patterns['trend_description']}, "
                f"suggesting growing demand over the forecast period"
            )
        elif patterns['trend'] == 'decreasing':
            insights.append(
                f"Forecast indicates {patterns['trend_description']}, "
                f"signaling potential declining demand"
            )
        else:
            insights.append(
                "Forecast exhibits stable pattern with minimal trend, "
                "indicating consistent demand levels"
            )
        
        # Volatility insight
        if patterns['volatility'] == 'low':
            insights.append(
                "Low volatility suggests predictable patterns, "
                "enabling confident planning decisions"
            )
        elif patterns['volatility'] == 'high':
            insights.append(
                f"High volatility (CV: {patterns['coefficient_of_variation']:.1f}%) "
                "indicates significant variation; maintain higher safety buffers"
            )
        
        # Confidence insight
        insights.append(
            f"Model confidence is {confidence['level']}, "
            f"indicating {confidence['description']} predictions"
        )
        
        # Anomaly insight
        if patterns['anomaly_count'] > 0:
            insights.append(
                f"Detected {patterns['anomaly_count']} unusual period(s) "
                "that deviate from expected patterns - review carefully"
            )
        
        return insights[:4]  # Limit to 4 key insights
    
    def _generate_business_implications(
        self,
        patterns: Dict[str, Any],
        confidence: Dict[str, str],
        business_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate business implications"""
        implications = []
        
        context_type = (business_context or {}).get('type', 'general')
        
        if context_type == 'inventory' or 'inventory' in str(business_context).lower():
            if patterns['trend'] == 'increasing':
                implications.append("Increase inventory levels to meet growing demand")
            implications.append(
                f"Maintain {patterns['volatility']} volatility buffer in safety stock"
            )
            implications.append("Review reorder points based on updated forecast")
        
        elif context_type == 'budget' or 'budget' in str(business_context).lower():
            total = sum([patterns['mean']] * 30)  # Approximate
            implications.append(f"Budget for approximately {total:.0f} total units")
            implications.append("Allocate contingency based on forecast uncertainty")
        
        elif context_type == 'production' or 'capacity' in str(business_context).lower():
            implications.append(
                f"Plan capacity for peak demand of {patterns['max']:.0f} units"
            )
            implications.append(
                f"Average production target: {patterns['mean']:.0f} units per period"
            )
        
        else:
            # General implications
            implications.extend([
                "Use forecast to align supply chain operations",
                f"Plan for average demand of {patterns['mean']:.0f} units per period",
                "Monitor actual vs forecast performance regularly"
            ])
        
        return implications[:3]
    
    def _identify_risk_factors(
        self,
        patterns: Dict[str, Any],
        confidence: float
    ) -> List[str]:
        """Identify risk factors"""
        risks = []
        
        if confidence < 0.6:
            risks.append(
                "Lower model confidence - results should be validated with domain experts"
            )
        
        if patterns['volatility'] == 'high':
            risks.append(
                "High forecast volatility increases risk of stock-outs or overstocking"
            )
        
        if patterns['anomaly_count'] > 0:
            risks.append(
                f"{patterns['anomaly_count']} anomalous period(s) detected - "
                "investigate potential causes"
            )
        
        if patterns['trend'] in ['increasing', 'decreasing']:
            risks.append(
                "Significant trend present - external factors may amplify or reverse trajectory"
            )
        
        # Generic risks
        risks.extend([
            "Market volatility and unexpected external events",
            "Changes in customer behavior or competitive landscape"
        ])
        
        return risks[:4]
    
    def _generate_recommendations(
        self,
        patterns: Dict[str, Any],
        confidence: Dict[str, str],
        business_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Monitoring recommendation
        if confidence['level'] in ['medium', 'moderate', 'low']:
            recommendations.append({
                'priority': 'high',
                'action': 'Implement weekly forecast monitoring',
                'detail': 'Track actual vs predicted values to validate accuracy'
            })
        
        # Volatility-based recommendation
        if patterns['volatility'] == 'high':
            recommendations.append({
                'priority': 'high',
                'action': 'Increase safety stock buffers',
                'detail': f"Maintain {patterns['coefficient_of_variation']:.0f}% additional buffer"
            })
        
        # Trend-based recommendation
        if patterns['trend'] == 'increasing':
            recommendations.append({
                'priority': 'medium',
                'action': 'Scale up capacity planning',
                'detail': f"Prepare for {patterns['trend_change_pct']:.1f}% demand growth"
            })
        elif patterns['trend'] == 'decreasing':
            recommendations.append({
                'priority': 'medium',
                'action': 'Optimize inventory levels',
                'detail': 'Reduce excess stock in anticipation of declining demand'
            })
        
        # Generic recommendations
        recommendations.extend([
            {
                'priority': 'medium',
                'action': 'Document forecast assumptions',
                'detail': 'Record business context and external factors for future reference'
            },
            {
                'priority': 'low',
                'action': 'Schedule forecast review',
                'detail': 'Establish regular cadence for forecast updates and refinements'
            }
        ])
        
        return recommendations[:4]


# Convenience function for quick interpretations
def interpret_forecast_quick(
    forecast_result: Dict[str, Any],
    analysis_result: Dict[str, Any],
    nvidia_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick interpretation without initializing service
    
    Args:
        forecast_result: Forecast output
        analysis_result: Analysis output
        nvidia_api_key: Optional API key
    
    Returns:
        Interpretation results
    """
    service = LLMInterpretationService(nvidia_api_key=nvidia_api_key)
    return service.interpret_forecast(forecast_result, analysis_result)


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    mock_forecast = {
        'values': [100 + i + (i % 7) * 5 for i in range(30)],
        'confidence_intervals': {
            'upper': [110 + i + (i % 7) * 5 for i in range(30)],
            'lower': [90 + i + (i % 7) * 5 for i in range(30)]
        }
    }
    
    mock_analysis = {
        'combined_summary': {'confidence': 0.75},
        'selected_model': {
            'model_name': 'Prophet',
            'confidence': 0.75,
            'reason': 'Best for daily data with seasonality'
        }
    }
    
    # Initialize service
    service = LLMInterpretationService()
    
    # Generate interpretation
    interpretation = service.interpret_forecast(
        mock_forecast,
        mock_analysis,
        business_context={'type': 'inventory', 'product': 'Widget A'}
    )
    
    # Display results
    print("\n" + "="*60)
    print("FORECAST INTERPRETATION")
    print("="*60 + "\n")
    print(interpretation['summary'])
    print("\n📊 Key Statistics:")
    print(json.dumps(interpretation['forecast_statistics'], indent=2))
    print("\n💡 Key Insights:")
    for insight in interpretation['key_insights']:
        print(f"  • {insight}")
    print("\n⚠️ Risk Factors:")
    for risk in interpretation['risk_factors']:
        print(f"  • {risk}")
    print("\n✅ Recommended Actions:")
    for rec in interpretation['recommended_actions']:
        print(f"  [{rec['priority'].upper()}] {rec['action']}: {rec['detail']}")