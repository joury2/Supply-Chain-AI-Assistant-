# streamlit_app/pages/SelectionValidation.py

"""
Dedicated UI for Model Selection & Validation Process
Shows the hybrid approach: Knowledge Base + Rules
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.services.knowledge_base_services.core.supply_chain_service import SupplyChainForecastingService

st.set_page_config(
    page_title="Model Selection & Validation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Model Selection & Validation Dashboard")
st.markdown("**Hybrid Approach:** Knowledge Base (Relational) + Rule Engine")

# Initialize service
@st.cache_resource
def init_service():
    return SupplyChainForecastingService()

service = init_service()

# File upload
uploaded_file = st.file_uploader("Upload Dataset for Analysis", type=['csv', 'xlsx'])

if uploaded_file:
    # Read data
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
    
    # Show dataset preview
    with st.expander("📊 Dataset Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Detect frequency
    def detect_frequency(df):
        for col in df.columns:
            if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
                try:
                    dates = pd.to_datetime(df[col]).sort_values()
                    diff = dates.diff().mode().iloc[0]
                    days = diff.total_seconds() / (24 * 3600)
                    
                    if 0.9 <= days <= 1.1:
                        return "daily"
                    elif 6.5 <= days <= 7.5:
                        return "weekly"
                    elif 28 <= days <= 31:
                        return "monthly"
                except:
                    pass
        return "unknown"
    
    # Create dataset info
    dataset_info = {
        'name': uploaded_file.name,
        'columns': list(df.columns),
        'row_count': len(df),
        'frequency': detect_frequency(df),
        'missing_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)))
    }
    
    st.markdown("---")
    st.markdown("## 🔍 Analysis Process")
    
    if st.button("🚀 Run Selection & Validation Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing dataset..."):
            # Run analysis
            result = service.analyze_dataset_with_knowledge_base(dataset_info)
            
            # Store in session state
            st.session_state.analysis_result = result
    
    # Display results if available
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        # SECTION 1: Validation Results
        st.markdown("### ✅ Step 1: Data Validation")
        
        validation = result['rule_analysis']['validation']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if validation['valid']:
                st.success("**Status:** ✅ Valid")
            else:
                st.error("**Status:** ❌ Invalid")
        
        with col2:
            st.metric("Data Points", dataset_info['row_count'])
        
        with col3:
            st.metric("Missing %", f"{dataset_info['missing_percentage']:.1%}")
        
        # Errors and warnings
        if validation.get('errors'):
            with st.expander("❌ Validation Errors", expanded=True):
                for error in validation['errors']:
                    st.error(f"• {error}")
        
        if validation.get('warnings'):
            with st.expander("⚠️ Warnings"):
                for warning in validation['warnings']:
                    st.warning(f"• {warning}")
        
        st.markdown("---")
        
        # SECTION 2: Model Selection Process
        st.markdown("### 🤖 Step 2: Model Selection (Hybrid Approach)")
        
        selection = result['rule_analysis']['selection_analysis']
        compatible_models = selection['analysis'].get('compatible_models', [])
        incompatible_models = selection['analysis'].get('incompatible_models', [])
        
        # Show compatible models
        st.markdown("#### ✅ Compatible Models")
        
        if compatible_models:
            # Create DataFrame for display
            models_df = pd.DataFrame([
                {
                    'Model': m['model_name'],
                    'Score': f"{m.get('compatibility_score', 0):.1f}%",
                    'Status': m.get('status', 'compatible'),
                    'Source': m.get('source', 'rule_engine')
                }
                for m in compatible_models
            ])
            
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            
            # Visualize scores
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[m['model_name'] for m in compatible_models],
                y=[m.get('compatibility_score', 0) for m in compatible_models],
                marker_color='#28a745',
                text=[f"{m.get('compatibility_score', 0):.1f}%" for m in compatible_models],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Model Compatibility Scores',
                xaxis_title='Model',
                yaxis_title='Compatibility Score (%)',
                yaxis_range=[0, 110],
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No compatible models found")
        
        # Show incompatible models
        if incompatible_models:
            with st.expander("❌ Incompatible Models"):
                for m in incompatible_models:
                    reason = m.get('reason', 'Requirements not met')
                    st.write(f"**{m['model_name']}:** {reason}")
        
        st.markdown("---")
        
        # SECTION 3: Final Selection
        st.markdown("### 🏆 Step 3: Final Model Selection")
        
        selected = result['rule_analysis']['model_selection']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 1rem; color: white;'>
                <h2 style='margin: 0; color: white;'>🎯 {selected.get('selected_model', 'No Model Selected')}</h2>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>{selected.get('reason', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = selected.get('confidence', 0)
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.metric("Source", selected.get('source', 'Unknown'))
        
        st.markdown("---")
        
        # SECTION 4: Knowledge Base Insights
        st.markdown("### 📚 Step 4: Knowledge Base Recommendations")
        
        kb_recs = result.get('knowledge_base_recommendations', [])
        
        if kb_recs:
            for i, rec in enumerate(kb_recs[:5], 1):
                with st.container():
                    st.markdown(f"""
                    <div style='background: #f8f9fa; padding: 1rem; 
                                border-left: 4px solid #17a2b8; border-radius: 0.5rem; margin: 0.5rem 0;'>
                        <strong>{i}.</strong> {rec if isinstance(rec, str) else rec.get('message', str(rec))}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No additional recommendations from knowledge base")
        
        st.markdown("---")
        
        # SECTION 5: Combined Summary
        st.markdown("### 📋 Final Summary")
        
        summary = result['combined_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if summary['can_proceed']:
                st.success("**Can Proceed:** ✅ Yes")
            else:
                st.error("**Can Proceed:** ❌ No")
        
        with col2:
            st.metric("Confidence", f"{summary['confidence']*100:.1f}%")
        
        with col3:
            st.metric("Compatible Models", len(compatible_models))
        
        with col4:
            st.metric("Issues Found", len(validation.get('errors', [])))
        
        # Decision explanation
        with st.expander("📊 Decision Logic"):
            st.json({
                'validation_passed': validation['valid'],
                'models_found': len(compatible_models),
                'selection_method': selected.get('source', 'hybrid'),
                'confidence_factors': {
                    'data_quality': f"{(1 - dataset_info['missing_percentage'])*100:.1f}%",
                    'data_volume': 'sufficient' if dataset_info['row_count'] >= 50 else 'limited',
                    'compatible_models': len(compatible_models)
                }
            })
        
        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export summary as JSON
            import json
            json_str = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json_str,
                file_name="selection_validation_report.json",
                mime="application/json"
            )
        
        with col2:
            # Export models as CSV
            if compatible_models:
                models_export_df = pd.DataFrame([
                    {
                        'Model Name': m['model_name'],
                        'Compatibility Score': m.get('compatibility_score', 0),
                        'Status': m.get('status', ''),
                        'Source': m.get('source', ''),
                        'Requirements': ', '.join(m.get('requirements_met', []))
                    }
                    for m in compatible_models
                ])
                
                csv = models_export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Models (CSV)",
                    data=csv,
                    file_name="compatible_models.csv",
                    mime="text/csv"
                )

else:
    # Welcome screen
    st.info("👆 Upload a dataset to see the selection and validation process")
    
    st.markdown("""
    ## How It Works
    
    This dashboard shows our **hybrid model selection approach**:
    
    ### 1️⃣ Data Validation (Rule Engine)
    - Checks minimum data requirements
    - Validates data quality
    - Identifies issues and warnings
    
    ### 2️⃣ Model Compatibility (Hybrid)
    - **Knowledge Base:** Queries relational database for model capabilities
    - **Rule Engine:** Validates against data characteristics
    - Scores each model's compatibility
    
    ### 3️⃣ Final Selection (Decision Logic)
    - Ranks compatible models
    - Considers confidence scores
    - Selects best fit with explanation
    
    ### 4️⃣ Recommendations
    - Provides actionable insights
    - Suggests improvements
    - Explains selection reasoning
    
    **Upload a dataset to see the analysis in action!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>🎯 Model Selection & Validation Engine | Hybrid Approach: Knowledge Base + Rules</small>
</div>
""", unsafe_allow_html=True)