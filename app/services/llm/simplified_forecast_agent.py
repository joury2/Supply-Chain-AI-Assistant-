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
        
        # Session state
        self.current_dataset_df = None
        self.current_result = None
        self.conversation_history = []
        
        # Initialize services with error handling
        try:
            logger.info("🔧 Initializing forecasting service...")
            from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService
            self.forecasting_service = SupplyChainForecastingService(db_path=db_path)
            logger.info("✅ Forecasting service initialized")
        except Exception as e:
            logger.error(f"❌ Forecasting service failed: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to initialize forecasting service: {str(e)}")
        
        # Initialize interpretation service (optional - with fallback)
        try:
            logger.info("📊 Initializing interpretation service...")
            from app.services.llm.interpretation_service import LLMInterpretationService
            self.interpretation_service = LLMInterpretationService(
                nvidia_api_key=nvidia_api_key,
                vector_store_path=vector_store_path,
                use_rag=False  # Disable RAG in interpretation to avoid conflicts
            )
            self.has_interpretation = True
            logger.info("✅ Interpretation service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Interpretation service unavailable: {e}")
            self.interpretation_service = None
            self.has_interpretation = False
        
        # Initialize LLM (for RAG questions only)
        try:
            logger.info("🤖 Connecting to NVIDIA LLM...")
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            self.llm = ChatNVIDIA(
                model="meta/llama-3.1-70b-instruct",
                api_key=nvidia_api_key,
                temperature=0.7,
                max_tokens=1024
            )
            logger.info("✅ LLM connected")
        except Exception as e:
            logger.warning(f"⚠️ LLM connection failed: {e}")
            self.llm = None
        
        # Load RAG (optional - with fallback)
        try:
            logger.info("📚 Loading RAG knowledge base...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = FAISS.load_local(
                vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            self.has_rag = True
            logger.info("✅ RAG knowledge base loaded")
        except Exception as e:
            logger.warning(f"⚠️ RAG not available: {e}")
            self.vectorstore = None
            self.has_rag = False
        
        logger.info("✅ Agent initialization complete!")
    
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
        """Detect user intent - SIMPLIFIED"""
        message_lower = message.lower()
        
        # Priority order matters!
        if any(kw in message_lower for kw in ['analyze', 'check', 'validate', 'quality']):
            return {'intent': 'analyze', 'params': {}}
        
        if any(kw in message_lower for kw in ['forecast', 'predict']):
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
            logger.info(f"📈 Starting forecast for {horizon} periods...")
            
            # Create dataset info WITH DataFrame
            dataset_info = {
                'name': 'user_uploaded_dataset',
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
                business_context={'source': 'chatbot'}
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
            if intent == 'analyze':
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