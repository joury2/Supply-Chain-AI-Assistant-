# streamlit_app/pages/ForecastWithAPI.py
"""
Streamlit Frontend with Session Management in Sidebar
FIXED: Infinite upload loop and session duplication
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import os
import hashlib
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
API_TOKEN = os.getenv("FASTAPI_API_TOKEN", "forecasting-api-token-v1")

st.set_page_config(
    page_title="AI Forecasting (FastAPI Backend)",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .session-card {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .session-card:hover {
        background: #e0e2e6;
        cursor: pointer;
    }
    .active-session {
        background: #d4edda;
        border-left-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['session_id', 'messages', 'dataset_info', 'forecast_data', 'all_sessions', 'file_hashes', 'upload_complete']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'all_sessions' else {} if key == 'file_hashes' else False if key == 'upload_complete' else None if key != 'messages' else []

# ============================================================================
# API Helper Functions
# ============================================================================

def api_request(method: str, endpoint: str, data=None, files=None):
    """Make API request"""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    url = f"{FASTAPI_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, files=files, timeout=60)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=60)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. Backend may be slow.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                st.error(f"Detail: {error_detail.get('detail', 'Unknown')}")
            except:
                pass
        return None

def check_health():
    """Check backend health"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def fetch_all_sessions():
    """Fetch all sessions from backend"""
    result = api_request("GET", "/api/v1/sessions")
    return result if result else []

def switch_session(session_id: str):
    """Switch to different session"""
    st.session_state.session_id = session_id
    st.session_state.messages = []
    st.session_state.forecast_data = None
    st.session_state.upload_complete = False
    
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"📁 Switched to session: `{session_id[:8]}...`\n\nWhat would you like to do?"
    })

def get_file_hash(file_content):
    """Generate hash for file content to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()

def find_existing_session(file_content):
    """Check if file already has an active session"""
    file_hash = get_file_hash(file_content)
    return st.session_state.file_hashes.get(file_hash)

# ============================================================================
# UI Components
# ============================================================================

def show_forecast_chart(forecast_data):
    """Display forecast visualization - SAFE VERSION"""
    if not forecast_data or not isinstance(forecast_data, dict):
        st.info("No forecast data to display")
        return
    
    values = forecast_data.get('values', [])
    if not values:
        st.info("No forecast values available")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # SAFE access to confidence intervals
    if forecast_data.get('confidence_intervals'):
        ci = forecast_data['confidence_intervals']
        x_vals = list(range(len(values)))
        
        if ci.get('upper') and ci.get('lower'):
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=ci['upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=ci['lower'],
                mode='lines',
                name='Confidence Interval',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
    
    fig.update_layout(
        title='Forecast Results',
        xaxis_title='Period',
        yaxis_title='Value',
        height=450,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Sidebar with Session Management
# ============================================================================

with st.sidebar:
    st.title("🤖 AI Forecasting")
    st.caption("FastAPI Backend")
    
    # Health check
    is_healthy = check_health()
    if is_healthy:
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Offline")
        st.info("Start: `uvicorn app.api.main:app --reload`")
        st.stop()
    
    st.markdown("---")
    
    # Session Management Section
    st.markdown("### 📂 Sessions")
    
    # Fetch sessions button
    if st.button("🔄 Refresh Sessions", use_container_width=True):
        st.session_state.all_sessions = fetch_all_sessions()
        st.rerun()
    
    # Display sessions
    if st.session_state.all_sessions:
        st.markdown(f"**{len(st.session_state.all_sessions)} active session(s)**")
        
        for session in st.session_state.all_sessions:
            session_id = session['session_id']
            is_active = session_id == st.session_state.session_id
            
            # Session card
            card_class = "session-card active-session" if is_active else "session-card"
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="{card_class}">
                        <small>📁 {session_id[:12]}...</small><br>
                        <small>📊 {session['dataset_rows']:,} rows × {session['dataset_columns']} cols</small><br>
                        <small>🕐 {session['created_at'][:16]}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if not is_active:
                        if st.button("↪", key=f"switch_{session_id}", help="Switch to this session"):
                            switch_session(session_id)
                            st.rerun()
                    else:
                        st.markdown("✓")
    else:
        st.info("No active sessions")
    
    st.markdown("---")
    
    # File upload - FIXED: Prevent infinite loop
    st.markdown("### 📁 Upload Dataset")
    
    # Only process upload if not already completed
    if not st.session_state.upload_complete:
        uploaded_file = st.file_uploader(
            "CSV or Excel",
            type=['csv', 'xlsx'],
            help="Time series data",
            key="file_uploader"  # Important: Add key to prevent re-upload
        )
        
        if uploaded_file and not st.session_state.upload_complete:
            file_content = uploaded_file.getvalue()
            
            # Check for existing session first
            existing_session = find_existing_session(file_content)
            if existing_session:
                st.info("🔄 Using existing session for this dataset")
                switch_session(existing_session)
                st.session_state.upload_complete = True
                st.rerun()
            else:
                # New upload
                with st.spinner("⬆️ Uploading..."):
                    files = {"file": uploaded_file}
                    result = api_request("POST", "/api/v1/upload", files=files)
                    
                    if result:
                        # Store file hash to prevent duplicates
                        file_hash = get_file_hash(file_content)
                        st.session_state.file_hashes[file_hash] = result['session_id']
                        
                        st.session_state.session_id = result['session_id']
                        st.session_state.dataset_info = result
                        st.session_state.all_sessions = fetch_all_sessions()
                        st.session_state.upload_complete = True  # Mark as complete
                        
                        st.success(f"✅ Uploaded!")
                        st.info(f"{result['rows']:,} rows × {len(result['columns'])} cols")
                        
                        welcome = f"""✅ **Dataset uploaded!**

**{result['rows']:,} rows** × **{len(result['columns'])} columns**
**Frequency:** {result.get('frequency', 'Unknown')}

Ready! Try:
- "Analyze my dataset"
- "Forecast next 30 days"
"""
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": welcome
                        })
                        st.rerun()
    
    # Reset upload state button
    if st.session_state.upload_complete:
        if st.button("📁 Upload New Dataset", use_container_width=True):
            st.session_state.upload_complete = False
            st.rerun()
    
    st.markdown("---")
    
    # Current session status
    st.markdown("### 📊 Current Session")
    
    if st.session_state.session_id:
        st.success(f"✓ Active")
        st.code(st.session_state.session_id[:16] + "...", language=None)
        
        if st.session_state.dataset_info:
            st.metric("Rows", f"{st.session_state.dataset_info['rows']:,}")
    else:
        st.warning("⚠ No Active Session")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze", disabled=not st.session_state.session_id, use_container_width=True):
            with st.spinner("Analyzing..."):
                result = api_request("POST", "/api/v1/analyze", {"session_id": st.session_state.session_id})
                if result:
                    msg = f"""✅ **Analysis Complete**

**Status:** {result['status']}
**Models:** {len(result.get('compatible_models', []))}
**Selected:** {result.get('selected_model', {}).get('model_name', 'Unknown')}

{chr(10).join('• ' + r for r in result.get('recommendations', [])[:3])}
"""
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.rerun()
    
    with col2:
        if st.button("📈 Forecast", disabled=not st.session_state.session_id, use_container_width=True):
            with st.spinner("Forecasting..."):
                result = api_request("POST", "/api/v1/forecast", {
                    "session_id": st.session_state.session_id,
                    "horizon": 30
                })
                
                if not result:
                    st.error("❌ Failed to start forecast job")
                    st.rerun()
                
                job_id = result.get('job_id')
                if not job_id:
                    st.error("❌ No job ID returned from forecast request")
                    st.rerun()
                
                # Use placeholder for progress
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Poll for job completion
                max_attempts = 50  # 10 seconds total
                for attempt in range(max_attempts):
                    time.sleep(0.2)
                    
                    status = api_request("GET", f"/api/v1/forecast/status/{job_id}")
                    if not status:
                        status_placeholder.error("❌ Failed to get forecast status")
                        continue  # Try again
                    
                    job_status = status.get('status', 'unknown')
                    
                    if job_status == 'completed':
                        progress_placeholder.success("✅ Forecast completed!")
                        
                        # SAFE ACCESS with multiple checks
                        result_data = status.get('result', {})
                        forecast_data = result_data.get('forecast_data', {})
                        
                        if forecast_data and forecast_data.get('values'):
                            st.session_state.forecast_data = forecast_data
                            values_count = len(forecast_data['values'])
                            msg = f"""✅ **Forecast Complete!**

    {values_count} predictions generated.

    See visualization below.
    """
                        else:
                            msg = "✅ **Forecast Complete!**\n\nForecast data is being processed."
                        
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        st.rerun()
                        break
                        
                    elif job_status == 'failed':
                        error_msg = status.get('error', 'Unknown error occurred')
                        progress_placeholder.error(f"❌ Forecast failed")
                        status_placeholder.error(f"Error: {error_msg}")
                        break
                        
                    elif job_status in ['pending', 'processing']:
                        progress = status.get('progress', 0)
                        progress_placeholder.progress(progress / 100)
                        status_placeholder.info(f"🔄 Forecast in progress... {progress}%")
                    
                    elif job_status == 'unknown':
                        status_placeholder.warning("⚠️ Forecast status unknown")
                    
                    # Last attempt - show timeout message
                    if attempt == max_attempts - 1:
                        progress_placeholder.warning("⏰ Forecast taking longer than expected...")
                        status_placeholder.info("""
                        The forecast is still running in the background. 
                        You can:
                        - Continue using the chat
                        - Check back later
                        - The results will appear when ready
                        """)
    
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🗑️ Delete Session", disabled=not st.session_state.session_id, use_container_width=True):
        if st.session_state.session_id:
            # Remove from file hashes
            file_hashes_to_remove = []
            for file_hash, session_id in st.session_state.file_hashes.items():
                if session_id == st.session_state.session_id:
                    file_hashes_to_remove.append(file_hash)
            
            for file_hash in file_hashes_to_remove:
                del st.session_state.file_hashes[file_hash]
            
            api_request("DELETE", f"/api/v1/session/{st.session_state.session_id}")
            st.session_state.session_id = None
            st.session_state.dataset_info = None
            st.session_state.messages = []
            st.session_state.forecast_data = None
            st.session_state.upload_complete = False
            st.session_state.all_sessions = fetch_all_sessions()
            st.rerun()

# ============================================================================
# Main Area
# ============================================================================

st.title("💬 AI Forecasting Assistant")
st.caption("Powered by FastAPI Backend")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show forecast chart
if st.session_state.messages and st.session_state.forecast_data:
    show_forecast_chart(st.session_state.forecast_data)

# Chat input
if prompt := st.chat_input(
    "Ask me anything..." if st.session_state.session_id else "Upload dataset or switch session...",
    disabled=not st.session_state.session_id
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = api_request("POST", "/api/v1/chat", {
                "message": prompt,
                "session_id": st.session_state.session_id
            })
            
            if result:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['response']
                })
                
                if result.get('has_forecast_data'):
                    st.session_state.forecast_data = result['forecast_data']
                
                st.rerun()

# Welcome screen
if not st.session_state.messages:
    st.info("👆 Upload a dataset or switch to an existing session!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Analysis:**
        - Data validation
        - Model recommendation
        - Quality checks
        """)
    
    with col2:
        st.markdown("""
        **📈 Forecasting:**
        - Multi-horizon forecasts
        - Confidence intervals
        - Business insights
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>Frontend: Streamlit | Backend: FastAPI | AI: NVIDIA Llama 3.1</small>
</div>
""", unsafe_allow_html=True)