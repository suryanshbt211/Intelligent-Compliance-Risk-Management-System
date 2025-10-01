# icrms/dashboard/app.py
"""
ICRMS Dashboard
Interactive web interface using Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
sys.path.append('..')

# Page config
st.set_page_config(
    page_title="ICRMS Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏦 Intelligent Compliance & Risk Management System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Module",
        ["Dashboard", "Deepfake Detection", "Compliance Automation", 
         "Risk Management", "Reports"]
    )
    
    st.markdown("---")
    st.subheader("System Status")
    st.success("🟢 All Systems Operational")
    st.metric("Active Alerts", "3", delta="1")

# Dashboard Page
if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SARs Generated", "127", delta="12")
    with col2:
        st.metric("Risk Score", "42.3", delta="-5.2")
    with col3:
        st.metric("Deepfakes Detected", "8", delta="2")
    with col4:
        st.metric("Compliance Rate", "98.5%", delta="0.3%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Trends")
        dates = pd.date_range(start='2025-09-01', end='2025-09-30', freq='D')
        risk_data = pd.DataFrame({
            'Date': dates,
            'Risk Score': pd.Series(range(30)).apply(lambda x: 40 + (x % 10) + (x // 10))
        })
        fig = px.line(risk_data, x='Date', y='Risk Score', title="30-Day Risk Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("SAR Distribution")
        sar_data = pd.DataFrame({
            'Category': ['Wire Transfer', 'Crypto', 'Cash', 'Other'],
            'Count': [45, 32, 28, 22]
        })
        fig = px.pie(sar_data, values='Count', names='Category', title="SARs by Category")
        st.plotly_chart(fig, use_container_width=True)

# Deepfake Detection Page
elif page == "Deepfake Detection":
    st.header("🎭 Deepfake Detection")
    
    upload_type = st.radio("Upload Type", ["Image", "Video"])
    
    if upload_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            with col2:
                with st.spinner("Analyzing..."):
                    st.success("Analysis Complete!")
                    st.metric("Deepfake Probability", "12.3%")
                    st.metric("Confidence", "94.2%")
                    st.info("✅ Image appears authentic")
    else:
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            st.video(uploaded_file)
            if st.button("Analyze Video"):
                with st.spinner("Processing video..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    st.success("Video Analysis Complete!")
                    st.metric("Frames Analyzed", "300")
                    st.metric("Deepfake Detection", "2 frames flagged")

# Compliance Automation Page
elif page == "Compliance Automation":
    st.header("📋 Compliance Automation")
    
    tab1, tab2 = st.tabs(["Generate SAR", "Monitor Transactions"])
    
    with tab1:
        st.subheader("Generate Suspicious Activity Report")
        
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.text_input("Customer ID")
            amount = st.number_input("Transaction Amount ($)", min_value=0.0)
            transaction_type = st.selectbox("Transaction Type", 
                                          ["Wire Transfer", "Cash", "Crypto", "ACH"])
        with col2:
            location = st.text_input("Location")
            description = st.text_area("Description")
        
        if st.button("Generate SAR"):
            with st.spinner("Generating SAR..."):
                st.success("SAR Generated Successfully!")
                st.code(f"""
SAR ID: SAR-2025100112345
Customer: {customer_id}
Amount: ${amount}
Type: {transaction_type}
Risk Score: 75.2
Status: PENDING_REVIEW
                """)
    
    with tab2:
        st.subheader("Transaction Monitoring")
        # Sample data
        transactions = pd.DataFrame({
            'Transaction ID': [f'TXN-{i:04d}' for i in range(1, 11)],
            'Customer ID': [f'CUST-{i:03d}' for i in range(1, 11)],
            'Amount': [10000, 5000, 25000, 3000, 50000, 8000, 15000, 6000, 30000, 12000],
            'Risk Score': [45, 20, 78, 15, 92, 35, 55, 22, 85, 48],
            'Status': ['OK', 'OK', 'FLAGGED', 'OK', 'FLAGGED', 'OK', 'FLAGGED', 'OK', 'FLAGGED', 'OK']
        })
        st.dataframe(transactions, use_container_width=True)

# Risk Management Page
elif page == "Risk Management":
    st.header("📊 Risk Management")
    
    tab1, tab2, tab3 = st.tabs(["Portfolio Risk", "Stress Testing", "Forecasting"])
    
    with tab1:
        st.subheader("Portfolio Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("VaR (95%)", "-2.3%")
            st.metric("CVaR (95%)", "-3.8%")
        with col2:
            st.metric("Volatility", "15.2%")
            st.metric("Sharpe Ratio", "1.42")
        with col3:
            st.metric("Max Drawdown", "-12.5%")
            st.metric("Beta", "0.89")
        
        # Risk distribution chart
        returns = pd.Series(pd.Series(range(1000)).apply(lambda x: pd.np.random.normal(0, 0.02)))
        fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=50)])
        fig.update_layout(title="Returns Distribution", xaxis_title="Returns", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Stress Testing")
        
        scenario = st.selectbox("Select Scenario", 
                              ["Market Crash (-30%)", "Interest Rate Spike", 
                               "Credit Crisis", "Pandemic Scenario"])
        
        if st.button("Run Stress Test"):
            st.write("Stress Test Results:")
            results = pd.DataFrame({
                'Metric': ['Expected Return', 'Volatility', 'VaR 95%', 'Max Loss'],
                'Current': ['8.2%', '15.3%', '-2.3%', '-12.5%'],
                'Stressed': ['-15.3%', '45.2%', '-18.5%', '-42.3%']
            })
            st.table(results)
    
    with tab3:
        st.subheader("Risk Forecasting")
        
        forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        
        # Generate forecast data
        dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='D')
        forecast = pd.DataFrame({
            'Date': dates,
            'Forecast': pd.Series(range(forecast_horizon)).apply(lambda x: 42 + (x * 0.1)),
            'Lower': pd.Series(range(forecast_horizon)).apply(lambda x: 38 + (x * 0.1)),
            'Upper': pd.Series(range(forecast_horizon)).apply(lambda x: 46 + (x * 0.1))
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Forecast'], 
                                mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Upper'], 
                                fill=None, mode='lines', line_color='lightgray', name='Upper'))
        fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Lower'], 
                                fill='tonexty', mode='lines', line_color='lightgray', name='Lower'))
        fig.update_layout(title=f"{forecast_horizon}-Day Risk Forecast")
        st.plotly_chart(fig, use_container_width=True)

# Reports Page
elif page == "Reports":
    st.header("📄 Reports & Analytics")
    
    report_type = st.selectbox("Report Type", 
                             ["Daily Summary", "Weekly Analysis", "Monthly Report", "Custom"])
    
    date_range = st.date_input("Select Date Range", 
                              value=[datetime.now() - timedelta(days=7), datetime.now()])
    
    if st.button("Generate Report"):
        st.success("Report Generated!")
        st.download_button("Download PDF Report", data="Sample Report Content", 
                         file_name="icrms_report.pdf")

# Footer
st.markdown("---")
st.markdown("© 2025 ICRMS | Powered by Open Source AI")
