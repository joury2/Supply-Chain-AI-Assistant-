# streamlit_app/pages/chatbot.py
# Chatbot page for Forecasting Assistant with Guided Workflow
# streamlit run streamlit_app/pages/chatbot.py 
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import your services
from app.knowledge_base.rule_layer.rule_engine import RuleEngineService
from app.services.model_serving.model_registry_service import ModelRegistryService

# Page config
st.set_page_config(
    page_title="Forecasting Assistant",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'forecast_result' not in st.session_state:
    st.session_state.forecast_result = None

# Initialize services
@st.cache_resource
def init_services():
    """Initialize backend services"""
    try:
        rule_engine = RuleEngineService()
        model_registry = ModelRegistryService()
        return rule_engine, model_registry
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        return None, None

rule_engine, model_registry = init_services()

# Header
st.title("📈 Forecasting Assistant")
st.markdown("### Upload your data and let me help you forecast!")

# Sidebar for navigation
with st.sidebar:
    st.markdown("### 🔍 Current Step")
    steps = {
        'upload': '1️⃣ Upload Data',
        'analyze': '2️⃣ Analyze Dataset',
        'forecast_type': '3️⃣ Choose Forecast',
        'results': '4️⃣ View Results'
    }
    for key, label in steps.items():
        if st.session_state.step == key:
            st.markdown(f"**{label}** ← You are here")
        else:
            st.markdown(f"{label}")
    
    st.divider()
    if st.button("🔄 Start Over"):
        for key in ['step', 'uploaded_data', 'analysis_result', 'forecast_result']:
            st.session_state[key] = None if key != 'step' else 'upload'
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

# STEP 1: Upload Data
if st.session_state.step == 'upload':
    with col1:
        st.markdown("## 📁 Upload Your Dataset")
        st.markdown("Upload a CSV or Excel file containing your time series data.")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Your file should contain a date/time column and numeric values to forecast"
        )
        
        if uploaded_file:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.uploaded_data = df
                
                # Show preview
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
                
                with st.expander("📊 Data Preview (first 10 rows)", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Quick stats
                st.markdown("#### Quick Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with stats_col2:
                    st.metric("Total Columns", len(df.columns))
                with stats_col3:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                
                if st.button("🔍 Analyze Dataset →", type="primary", use_container_width=True):
                    st.session_state.step = 'analyze'
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **Your dataset should include:**
        - A date/time column
        - At least one numeric column to forecast
        - Preferably 24+ data points
        
        **Supported formats:**
        - CSV files
        - Excel files (.xlsx, .xls)
        """)

# STEP 2: Analyze Dataset
elif st.session_state.step == 'analyze':
    with col1:
        st.markdown("## 🔍 Analyzing Your Dataset...")
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # Show data info
            st.markdown("#### Dataset Overview")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown("**Columns:**")
                for col in df.columns:
                    dtype = df[col].dtype
                    st.text(f"• {col} ({dtype})")
            with info_col2:
                st.markdown("**Data Range:**")
                st.text(f"Rows: {len(df)}")
                st.text(f"Columns: {len(df.columns)}")
            
            # Column selection
            st.markdown("#### 🎯 Column Mapping")
            st.markdown("Help me understand your data by identifying key columns:")
            
            col_select1, col_select2 = st.columns(2)
            with col_select1:
                date_column = st.selectbox(
                    "📅 Date/Time Column",
                    options=df.columns.tolist(),
                    help="Select the column containing dates or timestamps"
                )
            with col_select2:
                target_column = st.selectbox(
                    "🎯 Target Column (what to forecast)",
                    options=[col for col in df.columns if col != date_column],
                    help="Select the numeric column you want to forecast"
                )
            
            st.divider()
            
            # Validation button
            if st.button("✓ Validate & Continue →", type="primary", use_container_width=True):
                with st.spinner("🔄 Running validation checks..."):
                    try:
                        # Call your rule engine for validation
                        # Mock response for now - replace with actual call
                        analysis_result = {
                            'status': 'success',
                            'data_quality': {
                                'total_rows': len(df),
                                'missing_values': df.isnull().sum().sum(),
                                'date_range': f"{df[date_column].min()} to {df[date_column].max()}"
                            },
                            'compatible_models': ['ARIMA', 'Prophet', 'LightGBM'],
                            'recommendations': ['Dataset looks good for forecasting'],
                            'issues': []
                        }
                        
                        st.session_state.analysis_result = analysis_result
                        st.session_state.step = 'forecast_type'
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Validation failed: {str(e)}")
        else:
            st.warning("No data found. Please upload a file first.")
            if st.button("← Back to Upload"):
                st.session_state.step = 'upload'
                st.rerun()
    
    with col2:
        st.markdown("### ℹ️ What's happening?")
        st.info("""
        The system is checking:
        - ✓ Data types and formats
        - ✓ Missing values
        - ✓ Time series integrity
        - ✓ Model compatibility
        """)

# STEP 3: Choose Forecast Type
elif st.session_state.step == 'forecast_type':
    with col1:
        st.markdown("## 🎯 What would you like to forecast?")
        
        if st.session_state.analysis_result:
            # Show validation summary
            with st.expander("✅ Validation Summary", expanded=False):
                result = st.session_state.analysis_result
                st.success("Dataset passed validation checks!")
                
                if 'compatible_models' in result:
                    st.markdown("**Compatible Models:**")
                    for model in result['compatible_models']:
                        st.markdown(f"- {model}")
            
            st.divider()
            
            # Forecast options
            st.markdown("### Choose your forecast horizon:")
            
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("📅 Next Month", use_container_width=True, type="primary"):
                    st.session_state.forecast_horizon = 30
                    st.session_state.forecast_type = "Next Month"
                    st.session_state.step = 'results'
                    st.rerun()
                
                if st.button("📊 Next Quarter", use_container_width=True):
                    st.session_state.forecast_horizon = 90
                    st.session_state.forecast_type = "Next Quarter"
                    st.session_state.step = 'results'
                    st.rerun()
            
            with button_col2:
                if st.button("📈 Next 6 Months", use_container_width=True):
                    st.session_state.forecast_horizon = 180
                    st.session_state.forecast_type = "Next 6 Months"
                    st.session_state.step = 'results'
                    st.rerun()
                
                if st.button("🎨 Custom Period", use_container_width=True):
                    custom_days = st.number_input("Enter number of days:", min_value=1, max_value=365, value=30)
                    if st.button("Run Custom Forecast"):
                        st.session_state.forecast_horizon = custom_days
                        st.session_state.forecast_type = f"Next {custom_days} days"
                        st.session_state.step = 'results'
                        st.rerun()
            
            st.divider()
            
            if st.button("← Back to Analysis"):
                st.session_state.step = 'analyze'
                st.rerun()
    
    with col2:
        st.markdown("### 📊 Recommendation")
        st.info("""
        Based on your dataset:
        
        **Suggested forecast:**
        Next Month (30 days)
        
        **Why?**
        Your dataset has 48 months of data, 
        which is ideal for short to medium-term 
        forecasting.
        """)

# STEP 4: Results
elif st.session_state.step == 'results':
    with col1:
        st.markdown(f"## 📈 Forecast Results: {st.session_state.get('forecast_type', 'Forecast')}")
        
        # Run forecasting
        with st.spinner("🔄 Generating forecast..."):
            try:
                # Mock forecast - Replace with actual forecast service call
                # forecast_result = your_forecast_service.predict(...)
                
                # Create sample forecast visualization
                import numpy as np
                dates = pd.date_range(start='2024-01-01', periods=st.session_state.forecast_horizon)
                values = np.random.randn(st.session_state.forecast_horizon).cumsum() + 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, 
                    y=values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title='Forecast Results',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Results summary
                st.markdown("### 📊 Forecast Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Forecast Period", st.session_state.forecast_type)
                with summary_col2:
                    st.metric("Model Used", "LightGBM")
                with summary_col3:
                    st.metric("Confidence", "85%")
                
                st.success("✅ Forecast completed successfully!")
                
                # Download button
                st.download_button(
                    label="📥 Download Forecast Data",
                    data=pd.DataFrame({'date': dates, 'forecast': values}).to_csv(index=False),
                    file_name='forecast_results.csv',
                    mime='text/csv'
                )
                
                st.divider()
                
                # Next steps
                st.markdown("### 🤔 What's next?")
                st.markdown("""
                Soon you'll be able to:
                - Ask questions about the forecast
                - Request interpretations and insights
                - Compare different forecast scenarios
                
                **Coming in Phase 2: Conversational AI Analysis**
                """)
                
                if st.button("📊 Forecast Another Dataset", type="primary"):
                    st.session_state.step = 'upload'
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Forecasting failed: {str(e)}")
                if st.button("← Back to Forecast Selection"):
                    st.session_state.step = 'forecast_type'
                    st.rerun()
    
    with col2:
        st.markdown("### 💡 Understanding Results")
        st.info("""
        **Your forecast shows:**
        - Predicted values for the selected period
        - Confidence intervals (coming soon)
        - Trend analysis
        
        **Model Selected:**
        LightGBM was chosen based on your 
        dataset characteristics.
        """)

# Footer
st.markdown("---")
st.markdown("*Forecasting Assistant v1.0 - Guided Workflow Mode*")