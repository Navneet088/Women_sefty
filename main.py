import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and tools
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_assets()

# UI Layout
st.set_page_config(page_title="Women Safety Risk Analyzer", page_icon="🛡️")
st.title("🛡️ Women's Safety Risk Classification")
st.markdown("Enter the crime statistics below to predict the safety risk level for a region.")

# Input Fields
st.sidebar.header("Input Crime Data")
rape = st.sidebar.number_input("Rape Cases", min_value=0)
ka = st.sidebar.number_input("Kidnapping & Abduction", min_value=0)
dd = st.sidebar.number_input("Dowry Deaths", min_value=0)
aow = st.sidebar.number_input("Assault on Women", min_value=0)
aom = st.sidebar.number_input("Assault on Modesty", min_value=0)
dv = st.sidebar.number_input("Domestic Violence", min_value=0)
wt = st.sidebar.number_input("Women Trafficking", min_value=0)

if st.button("Analyze Risk Level"):
    # Prepare input for prediction
    input_data = np.array([[rape, ka, dd, aow, aom, dv, wt]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction_idx = model.predict(input_scaled)[0]
    risk_level = le.inverse_transform([prediction_idx])[0]
    
    # Display Result
    st.subheader(f"Results:")
    if risk_level == 'High':
        st.error(f"Predicted Risk: {risk_level}")
    elif risk_level == 'Mid':
        st.warning(f"Predicted Risk: {risk_level}")
    else:
        st.success(f"Predicted Risk: {risk_level}")
    
    # Simple Visual Breakout
    st.bar_chart(pd.DataFrame({
        "Crime Type": ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT'],
        "Counts": [rape, ka, dd, aow, aom, dv, wt]
    }).set_index("Crime Type"))

else:
    st.info("Adjust the values in the sidebar and click 'Analyze' to begin.")