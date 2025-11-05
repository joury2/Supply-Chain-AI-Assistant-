# streamlit_app/pages/ForecastChat.py
# Streamlit Chat Interface for AI Forecasting Assistant

"""
Streamlit Chat Interface for AI Forecasting Assistant
Complete UI with file upload + conversational AI + forecast visualization + interpretation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
import json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.services.llm.forecast_agent import ForecastAgent

# Page config
st.set_page_config(
    page_title="AI Forecasting Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ddd;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left-color: #667eea;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #28a745;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        display: inline-block;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        display: inline-block;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .info-badge {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        display: inline-block;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .stButton > button {
        width: 100%;
    }
    .interpretation-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .insight-item {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid #28a745;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'forecast_done' not in st.session_state:
    st.session_state.forecast_done = False
if 'show_data_preview' not in st.session_state:
    st.session_state.show_data_preview = False
if 'current_forecast_data' not in st.session_state:
    st.session_state.current_forecast_data = None
if 'interpretation_data' not in st.session_state:
    st.session_state.interpretation_data = None

# Initialize agent
@st.cache_resource
def init_agent():
    """Initialize the forecast agent"""
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    
    if not nvidia_api_key:
        st.error("⚠️ NVIDIA_API_KEY not found in environment variables!")
        st.info("Get your API key from: https://build.nvidia.com/")
        st.code("export NVIDIA_API_KEY='your-key-here'")
        st.stop()
    
    try:
        with st.spinner("🤖 Initializing AI Forecasting Assistant..."):
            agent = ForecastAgent(nvidia_api_key=nvidia_api_key)
        return agent
    except FileNotFoundError as e:
        st.error("❌ Vector store not found! Please run `python rag_setup.py` first.")
        st.code("python rag_setup.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        st.stop()

# Helper function to create forecast visualization
def create_forecast_plot(forecast_data: dict, dataset_df: pd.DataFrame = None):
    """Create interactive forecast visualization"""
    fig = go.Figure()
    
    # Historical data if available
    if dataset_df is not None and len(dataset_df) > 0:
        # Try to find date and target columns
        date_cols = dataset_df.select_dtypes(include=['datetime64']).columns
        numeric_cols = dataset_df.select_dtypes(include=['number']).columns
        
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            target_col = numeric_cols[0]
            
            fig.add_trace(go.Scatter(
                x=dataset_df[date_col],
                y=dataset_df[target_col],
                mode='lines',
                name='Historical Data',
                line=dict(color='#1f77b4', width=2)
            ))
    
    # Forecast data
    if 'values' in forecast_data:
        forecast_values = forecast_data['values']
        horizon = len(forecast_values)
        
        # Create future dates (simplified - you might want to use actual date logic)
        if dataset_df is not None and len(date_cols) > 0:
            last_date = dataset_df[date_col].max()
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
        else:
            future_dates = list(range(1, horizon + 1))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6)
        ))
        
        # Confidence intervals if available
        if 'confidence_intervals' in forecast_data:
            ci = forecast_data['confidence_intervals']
            if 'upper' in ci and 'lower' in ci:
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=ci['upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=ci['lower'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    fill='tonexty',
                    showlegend=True
                ))
    
    fig.update_layout(
        title='Forecast Visualization',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Helper function to display interpretation
def display_interpretation(interpretation_data: dict):
    """Display forecast interpretation with insights"""
    if not interpretation_data:
        return
    
    st.markdown("---")
    st.markdown("### 📊 Forecast Interpretation")
    
    # Main interpretation section
    with st.container():
        st.markdown(f"""
        <div class="interpretation-section">
            <h4>📈 Executive Summary</h4>
            <p>{interpretation_data.get('summary', 'No summary available')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key insights
    if 'key_insights' in interpretation_data and interpretation_data['key_insights']:
        st.markdown("#### 💡 Key Insights")
        for insight in interpretation_data['key_insights']:
            st.markdown(f"""
            <div class="insight-item">
                <strong>•</strong> {insight}
            </div>
            """, unsafe_allow_html=True)
    
    # Business recommendations
    if 'business_recommendations' in interpretation_data and interpretation_data['business_recommendations']:
        with st.expander("🎯 Business Recommendations", expanded=True):
            for i, recommendation in enumerate(interpretation_data['business_recommendations'], 1):
                st.markdown(f"**{i}.** {recommendation}")
    
    # Risk factors
    if 'risk_factors' in interpretation_data and interpretation_data['risk_factors']:
        with st.expander("⚠️ Risk Factors", expanded=False):
            for risk in interpretation_data['risk_factors']:
                st.markdown(f"• {risk}")
    
    # Technical details
    if 'technical_details' in interpretation_data and interpretation_data['technical_details']:
        with st.expander("🔧 Technical Details", expanded=False):
            for detail in interpretation_data['technical_details']:
                st.markdown(f"• {detail}")

# Sidebar
with st.sidebar:
    st.title("🤖 AI Forecasting")
    st.markdown("---")
    
    # Upload section
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your time series data (CSV or Excel)"
    )
    
    if uploaded_file and not st.session_state.dataset_uploaded:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Try to parse date columns
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            st.session_state.current_df = df
            
            # Initialize agent if needed
            if st.session_state.agent is None:
                st.session_state.agent = init_agent()
            
            # Upload to agent
            st.session_state.agent.upload_dataset(df)
            st.session_state.dataset_uploaded = True
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.info(f"{len(df):,} rows × {len(df.columns)} columns")
            
            # Show data preview toggle
            st.session_state.show_data_preview = True
            
            # Add welcome message
            welcome_msg = f"""Great! I've received your dataset with **{len(df):,} rows** and **{len(df.columns)} columns**. 

I can help you:
- 📊 Analyze data quality and patterns
- 🤖 Recommend the best forecasting models
- 📈 Generate accurate forecasts with interpretation
- 💡 Explain results and provide actionable insights

What would you like to do first?"""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_msg
            })
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Data preview
    if st.session_state.show_data_preview and st.session_state.current_df is not None:
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(st.session_state.current_df.head(5), use_container_width=True)
    
    st.markdown("---")
    
    # Dataset status
    st.markdown("### 📊 Status")
    if st.session_state.dataset_uploaded:
        st.markdown('<span class="success-badge">✓ Dataset Loaded</span>', unsafe_allow_html=True)
        
        if st.session_state.current_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(st.session_state.current_df):,}")
            with col2:
                st.metric("Columns", len(st.session_state.current_df.columns))
        
        if st.session_state.analysis_done:
            st.markdown('<span class="success-badge">✓ Analysis Complete</span>', unsafe_allow_html=True)
        
        if st.session_state.forecast_done:
            st.markdown('<span class="success-badge">✓ Forecast Generated</span>', unsafe_allow_html=True)
            
        if st.session_state.interpretation_data:
            st.markdown('<span class="info-badge">📊 Interpretation Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warning-badge">⚠ No Dataset</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🎯 Quick Actions")
    
    if st.button("🔍 Analyze Dataset", disabled=not st.session_state.dataset_uploaded):
        st.session_state.messages.append({"role": "user", "content": "Please analyze my dataset"})
        st.rerun()
    
    if st.button("📈 Forecast Next Month", disabled=not st.session_state.dataset_uploaded):
        st.session_state.messages.append({"role": "user", "content": "Forecast the next 30 days"})
        st.rerun()
    
    if st.button("📊 Forecast Next Quarter", disabled=not st.session_state.dataset_uploaded):
        st.session_state.messages.append({"role": "user", "content": "Forecast the next 90 days"})
        st.rerun()
    
    # Interpretation actions
    st.markdown("### 📊 Interpretation")
    
    if st.button("💡 Get Forecast Interpretation", disabled=not st.session_state.forecast_done):
        st.session_state.messages.append({"role": "user", "content": "Interpret the forecast results"})
        st.rerun()
    
    if st.button("🎯 Business Insights", disabled=not st.session_state.forecast_done):
        st.session_state.messages.append({"role": "user", "content": "Provide business insights from the forecast"})
        st.rerun()
    
    st.markdown("---")
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.session_state.analysis_done = False
        st.session_state.forecast_done = False
        st.session_state.interpretation_data = None
        st.rerun()
    
    if st.button("🗑️ Remove Dataset"):
        st.session_state.dataset_uploaded = False
        st.session_state.current_df = None
        st.session_state.messages = []
        st.session_state.analysis_done = False
        st.session_state.forecast_done = False
        st.session_state.show_data_preview = False
        st.session_state.interpretation_data = None
        st.rerun()
    
    st.markdown("---")
    
    # Help section
    with st.expander("💡 How to use"):
        st.markdown("""
        **Getting Started:**
        1. Upload your time series data (CSV/Excel)
        2. Ask questions in natural language
        3. Get analysis, forecasts, and insights
        
        **Example Questions:**
        - "Analyze my dataset"
        - "What models work with my data?"
        - "Forecast next 3 months"
        - "Why was this model selected?"
        - "Interpret the forecast results"
        - "Provide business insights"
        - "What's the confidence level?"
        - "Show me the forecast values"
        
        **Tips:**
        - Dataset should have date/time column
        - At least one numeric column to forecast
        - Minimum 24 data points recommended
        - Use interpretation for business insights
        """)

# Main chat area
st.title("💬 AI Forecasting Assistant")
st.markdown("Ask me anything about your data and forecasts!")

# Initialize agent if not done
if st.session_state.agent is None and st.session_state.dataset_uploaded:
    st.session_state.agent = init_agent()

# Display chat history
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Welcome message
        st.markdown("""
        <div class="assistant-message chat-message">
        <strong>👋 Welcome to AI Forecasting Assistant!</strong><br><br>
        I'm powered by <strong>NVIDIA Llama 3.1</strong> and specialized in time series forecasting with advanced interpretation.<br><br>
        
        <strong>I can help you:</strong>
        <ul>
        <li>📊 Analyze data quality and detect patterns</li>
        <li>🤖 Recommend the best forecasting models for your data</li>
        <li>📈 Generate accurate forecasts with confidence intervals</li>
        <li>📊 Provide detailed forecast interpretation and insights</li>
        <li>🎯 Offer business recommendations based on forecasts</li>
        <li>💡 Explain results and provide actionable insights</li>
        </ul>
        
        <strong>Get started by uploading your dataset in the sidebar!</strong> 📁
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display messages
        for i, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                <strong>👤 You:</strong><br>{content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>{content}
                </div>
                """, unsafe_allow_html=True)
                
                # Check if this is a forecast result and show visualization
                if i == len(st.session_state.messages) - 1:  # Last message
                    if st.session_state.agent and st.session_state.forecast_done:
                        forecast_data = st.session_state.agent.get_current_forecast_data()
                        if forecast_data:
                            st.plotly_chart(
                                create_forecast_plot(forecast_data, st.session_state.current_df),
                                use_container_width=True
                            )
                    
                    # Show interpretation if available
                    if st.session_state.interpretation_data:
                        display_interpretation(st.session_state.interpretation_data)

# Chat input
prompt = st.chat_input(
    "Ask me anything about forecasting..." if st.session_state.dataset_uploaded else "Please upload a dataset first...",
    disabled=not st.session_state.dataset_uploaded
)

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get agent response
    with st.spinner("🤔 Thinking..."):
        try:
            response = st.session_state.agent.ask_question(prompt)
            
            # Check if response indicates analysis or forecast completion
            if "analyz" in prompt.lower() or "validation" in response.lower():
                st.session_state.analysis_done = True
            
            if "forecast" in prompt.lower() and "success" in response.lower():
                st.session_state.forecast_done = True
                # Store forecast data for visualization
                st.session_state.current_forecast_data = st.session_state.agent.get_current_forecast_data()
            
            # Check if interpretation was requested
            if any(keyword in prompt.lower() for keyword in ['interpret', 'insight', 'business', 'recommendation', 'explain']):
                try:
                    # Get interpretation using the interpretation service
                    forecast_data = st.session_state.agent.get_current_forecast_data()
                    if forecast_data and hasattr(st.session_state.agent, 'interpretation_service'):
                        interpretation = st.session_state.agent.interpretation_service.interpret_forecast(
                            forecast_result=forecast_data,
                            analysis_result={'status': 'success'},  # You might want to pass actual analysis result
                            business_context={'type': 'general', 'source': 'streamlit_ui'}
                        )
                        st.session_state.interpretation_data = interpretation
                except Exception as e:
                    st.warning(f"Note: Interpretation service unavailable: {str(e)}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

# Show example questions if no dataset
if not st.session_state.dataset_uploaded:
    st.info("👆 Upload a dataset in the sidebar to get started!")
    
    st.markdown("### 📝 Example Questions You Can Ask:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Dataset Analysis:**
        - "Analyze my dataset"
        - "Check data quality"
        - "What's wrong with my data?"
        - "How much data do I need?"
        - "Detect outliers and anomalies"
        """)
    
    with col2:
        st.markdown("""
        **📈 Forecasting & Interpretation:**
        - "Forecast next month"
        - "Predict next quarter"
        - "Interpret the forecast results"
        - "Provide business insights"
        - "What model should I use?"
        - "Explain the forecast confidence"
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>
🤖 Powered by <strong>NVIDIA Llama 3.1 70B</strong> + RAG Knowledge Base + Advanced Interpretation<br>
Built with LangChain, Streamlit, and your Supply Chain Forecasting Service<br>
<em>All forecasts include detailed interpretation and business insights</em>
</small>
</div>
""", unsafe_allow_html=True)