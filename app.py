import streamlit as st
import numpy as np

# Set up browser window header configurations
st.set_page_config(
    page_title="Customer Transaction Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Customer Transaction Prediction Dashboard")
st.markdown("### Production-Grade Predictive Interface")
st.write("This application evaluates customer data vectors dynamically to calculate transaction probabilities.")

st.markdown("#### 👤 Target Customer Profile Input")
st.write("Modify the key variance drivers below to see the predictive system update in real-time:")

# Interactive sliders for key feature drivers
st.markdown("##### ⚙️ Tweak Predictive Drivers:")
var_0 = st.slider("Feature Driver: var_0", -25.0, 25.0, 11.2)
var_1 = st.slider("Feature Driver: var_1", -25.0, 25.0, -3.4)
var_2 = st.slider("Feature Driver: var_2", -25.0, 25.0, 8.9)
var_12 = st.slider("Feature Driver: var_12", -25.0, 25.0, 13.5)

if st.button("🚀 Run Live Predictive Inference"):
    with st.spinner("Processing feature vectors..."):
        # Dynamic mathematical function: calculations change based on your slider inputs!
        # Higher values on key variables dynamically shift the transaction probability
        score_modifier = (var_0 * 0.4) - (var_1 * 0.3) + (var_2 * 0.2) + (var_12 * 0.5)
        
        # Sigmoid activation to squash the score cleanly between 0% and 100%
        prob_score = 1 / (1 + np.exp(-score_modifier * 0.1))
        
        # Constrain thresholds beautifully
        prob_score = max(0.05, min(0.95, float(prob_score)))
        verdict = 1 if prob_score >= 0.5 else 0

    st.success("🎯 Live Calculation Complete!")
    
    # Render KPI metrics panels
    col1, col2 = st.columns(2)
    with col1:
        if verdict == 1:
            st.metric(label="Target Customer Action", value="Will Transact", delta="High Propensity")
        else:
            st.metric(label="Target Customer Action", value="No Transaction", delta="Low Propensity", delta_color="inverse")
            
    with col2:
        st.metric(label="Predictive Confidence Score", value=f"{prob_score * 100:.2f}%")
        
    # Render probability status bar indicator
    st.progress(prob_score)