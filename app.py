import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add project root for imports
sys.path.append(os.path.dirname(__file__))

from Agent_AI.agent import app as agent_app, AgentState

# Page configuration
st.set_page_config(
    page_title="ArymaEdge AI - Marketing Optimization Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 ArymaEdge AI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Marketing Spends Optimization Agent</h2>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Configuration")
        
        # Target Sales Input
        target_sales = st.number_input(
            "Target Sales Amount",
            min_value=1000000,
            max_value=10000000,
            value=5000000,
            step=100000,
            help="Enter your target sales amount"
        )
        
        # Optimization Type
        optimization_type = st.selectbox(
            "Optimization Type",
            ["Default Optimization", "ROI-based Optimization"],
            help="Choose the type of optimization algorithm"
        )
        
        use_roi = optimization_type == "ROI-based Optimization"
        
        # Run Button
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            st.session_state.run_optimization = True
            st.session_state.target_sales = target_sales
            st.session_state.use_roi = use_roi
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Optimization Results")
        
        if st.session_state.get('run_optimization', False):
            with st.spinner("Running optimization..."):
                try:
                    # Create initial state
                    initial_state = {
                        "target_sales": st.session_state.target_sales,
                        "use_roi": st.session_state.use_roi,
                        "spends": {},
                        "warnings": [],
                        "results": {},
                        "wants_report": False,
                        "report_data": {}
                    }
                    
                    # Run the agent
                    result = agent_app.invoke(initial_state)
                    
                    # Display results
                    if result.get("results"):
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.subheader("✅ Optimization Complete!")
                        
                        # Display spends
                        st.write("**Optimized Spends:**")
                        spends_df = pd.DataFrame([
                            {"Channel": k, "Spend": f"${v:,.2f}"} 
                            for k, v in result["results"]["spends"].items()
                        ])
                        st.dataframe(spends_df, use_container_width=True)
                        
                        # Display additional results
                        if "predicted_sales" in result["results"]:
                            st.write(f"**Predicted Sales:** ${result['results']['predicted_sales']:,.2f}")
                        
                        if "seasonality" in result["results"]:
                            st.write("**Seasonality Factors:**")
                            for season, value in result["results"]["seasonality"].items():
                                st.write(f"- {season}: {value}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display warnings
                        if result.get("warnings"):
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.subheader("⚠️ Warnings")
                            for warning in result["warnings"]:
                                st.write(f"• {warning}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Report generation
                        if st.button("📋 Generate Detailed Report", type="secondary"):
                            with st.spinner("Generating report..."):
                                # Create state for report generation
                                report_state = result.copy()
                                report_state["wants_report"] = True
                                
                                # Run report generation
                                report_result = agent_app.invoke(report_state)
                                
                                if report_result.get("report_data"):
                                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                    st.subheader("📊 Detailed Report")
                                    st.markdown(report_result["report_data"]["content"])
                                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    else:
                        st.error("❌ Optimization failed. Please check your inputs and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error during optimization: {str(e)}")
                    st.exception(e)
    
    with col2:
        st.header("ℹ️ Information")
        
        # About the agent
        st.info("""
        **About this Agent:**
        
        This AI agent optimizes marketing spends across multiple channels to achieve your target sales.
        
        **Features:**
        - 📊 Multi-channel optimization
        - 🎯 Target-based planning
        - ⚠️ Risk assessment
        - 📋 Detailed reporting
        - 📈 Historical data analysis
        """)
        
        # Data sources
        st.info("""
        **Data Sources:**
        - Historical marketing data
        - ROI performance metrics
        - Seasonal patterns
        - Competitive analysis
        """)
        
        # Instructions
        st.info("""
        **How to use:**
        1. Set your target sales amount
        2. Choose optimization type
        3. Click "Run Optimization"
        4. Review results and warnings
        5. Generate detailed report (optional)
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'run_optimization' not in st.session_state:
        st.session_state.run_optimization = False
    
    main()
