# streamlit_app/pages/ImprovedForecastChat.py
"""
Improved Streamlit UI with better error handling and UX
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Use the SIMPLIFIED agent (recommended)
from app.services.llm.simplified_forecast_agent import SimplifiedForecastAgent

st.set_page_config(
    page_title="AI Forecasting Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS (keep your existing CSS)
st.markdown("""
<style>
    /* Your existing CSS */
    .stApp { max-width: 1400px; margin: 0 auto; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .user-message { background: #e3f2fd; }
    .assistant-message { background: #f5f5f5; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Initialize agent (with error handling)
@st.cache_resource
def init_agent():
    """Initialize agent with proper error handling"""
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        st.error("⚠️ NVIDIA_API_KEY not set!")
        st.code("export NVIDIA_API_KEY='your-key'")
        st.stop()
    
    try:
        agent = SimplifiedForecastAgent(nvidia_api_key=api_key)
        return agent
    except FileNotFoundError:
        st.error("❌ Vector store not found! Run: python rag_setup.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        st.stop()

# Sidebar
with st.sidebar:
    st.title("🤖 AI Forecasting")
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Time series data with date column"
    )
    
    if uploaded_file and not st.session_state.dataset_uploaded:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Parse dates
            for col in df.columns:
                if 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            # Initialize agent if needed
            if st.session_state.agent is None:
                with st.spinner("Initializing..."):
                    st.session_state.agent = init_agent()
            
            # Upload dataset
            st.session_state.agent.upload_dataset(df)
            st.session_state.dataset_uploaded = True
            
            # Welcome message
            welcome = f"""✅ Dataset uploaded successfully!

**{len(df):,} rows** × **{len(df.columns)} columns**

I can help you:
- 📊 Analyze data quality
- 🤖 Recommend best models
- 📈 Generate forecasts
- 💡 Interpret results

What would you like to do?"""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")
    
    # Status
    st.markdown("---")
    st.markdown("### 📊 Status")
    
    if st.session_state.dataset_uploaded:
        st.success("✓ Dataset Loaded")
    else:
        st.warning("⚠ No Dataset")
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze", disabled=not st.session_state.dataset_uploaded, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Analyze my dataset"})
            st.rerun()
    
    with col2:
        if st.button("📈 Forecast", disabled=not st.session_state.dataset_uploaded, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Forecast next 30 days"})
            st.rerun()
    
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🗑️ Remove Dataset", use_container_width=True):
        st.session_state.dataset_uploaded = False
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()

# Main area
st.title("💬 AI Forecasting Assistant")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show visualization if this is the last message and forecast exists
        if msg["role"] == "assistant" and msg == st.session_state.messages[-1]:
            if st.session_state.agent and st.session_state.dataset_uploaded:
                forecast_data = st.session_state.agent.get_current_forecast_data()
                
                if forecast_data and 'values' in forecast_data:
                    values = forecast_data['values']
                    
                    # Create plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Add confidence intervals if available
                    if 'confidence_intervals' in forecast_data:
                        ci = forecast_data['confidence_intervals']
                        fig.add_trace(go.Scatter(
                            y=ci['upper'],
                            mode='lines',
                            name='Upper Bound',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            y=ci['lower'],
                            mode='lines',
                            name='Lower Bound',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(31, 119, 180, 0.2)',
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title='Forecast Visualization',
                        xaxis_title='Period',
                        yaxis_title='Value',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# Chat input (with loading state)
if prompt := st.chat_input(
    "Ask me anything..." if st.session_state.dataset_uploaded else "Upload dataset first...",
    disabled=not st.session_state.dataset_uploaded or st.session_state.is_processing
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                st.session_state.is_processing = True
                
                response = st.session_state.agent.ask_question(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.is_processing = False
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.is_processing = False

# Welcome screen
if not st.session_state.messages:
    st.info("👆 Upload a dataset in the sidebar to get started!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Dataset Analysis:**
        - "Analyze my dataset"
        - "Check data quality"
        - "What's wrong with my data?"
        """)
    
    with col2:
        st.markdown("""
        **📈 Forecasting:**
        - "Forecast next 30 days"
        - "Predict next quarter"
        - "Show me the results"
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>🤖 Powered by NVIDIA Llama 3.1 + Supply Chain Forecasting Service</small>
</div>
""", unsafe_allow_html=True)